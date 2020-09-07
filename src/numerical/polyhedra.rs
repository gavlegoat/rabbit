#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use nalgebra::{DMatrix, DVector, RowDVector};
use replace_with::replace_with_or_abort;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::os::raw::{c_double, c_int};

use crate::numerical::{AffineTransform, LinearConstraint, NumericalDomain};
use crate::AbstractDomain;

// Include generated bindings for GLPK and add a safe interface.
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

struct GLPProblem {
    prob: *mut glp_prob,
}

#[derive(Eq, PartialEq, Debug, Clone)]
enum GLPStatus {
    Infeasible,
    Unbounded,
    Unsolved,
}

impl GLPProblem {
    fn new() -> GLPProblem {
        GLPProblem {
            prob: unsafe { glp_create_prob() },
        }
    }

    fn load_matrix(&mut self, mat: &DMatrix<f64>, vec: &DVector<f64>) {
        println!("{} {}", mat, vec);
        let nr = mat.nrows();
        let nc = mat.ncols();
        unsafe {
            glp_add_rows(self.prob, nr as c_int);
            glp_add_cols(self.prob, nc as c_int);
        }
        let mut ia: Vec<c_int> = vec![0];
        let mut ja: Vec<c_int> = vec![0];
        let mut ar: Vec<c_double> = vec![0.];
        for (j, c) in mat.column_iter().enumerate() {
            for (i, v) in c.iter().enumerate() {
                ia.push((i + 1) as c_int);
                ja.push((j + 1) as c_int);
                ar.push(*v);
            }
        }
        let ne = (ar.len() - 1) as c_int;
        println!("{:?}", ia);
        println!("{:?}", ja);
        println!("{:?}", ar);
        println!("{}", ne);
        unsafe {
            glp_load_matrix(self.prob, ne, ia.as_ptr(), ja.as_ptr(), ar.as_ptr());
            for (i, v) in vec.iter().enumerate() {
                glp_set_row_bnds(
                    self.prob,
                    (i + 1) as c_int,
                    GLP_UP as c_int,
                    0. as c_double,
                    *v as c_double,
                );
            }
        }
    }

    fn set_objective(&mut self, vec: &[f64], cst: f64) {
        unsafe {
            glp_set_obj_coef(self.prob, 0 as c_int, cst as c_double);
            for (i, v) in vec.iter().enumerate() {
                glp_set_obj_coef(self.prob, (i + 1) as c_int, *v as c_double);
            }
            glp_set_obj_dir(self.prob, GLP_MAX as c_int);
        }
    }

    fn solve(&mut self) -> Result<f64, GLPStatus> {
        unsafe {
            let mut params = {
                let mut x = MaybeUninit::uninit();
                glp_init_smcp(x.as_mut_ptr());
                x.assume_init()
            };
            params.msg_lev = GLP_MSG_OFF as c_int;
            let simplex_res = glp_simplex(self.prob, &params);
            if simplex_res != 0 {
                return Err(GLPStatus::Unsolved);
            }
            let solve_res = glp_get_status(self.prob) as u32;
            if solve_res == GLP_NOFEAS {
                Err(GLPStatus::Infeasible)
            } else if solve_res == GLP_UNBND {
                Err(GLPStatus::Unbounded)
            } else if solve_res == GLP_OPT {
                let ret = glp_get_obj_val(self.prob);
                Ok(ret as f64)
            } else {
                println!("solve_res: {}", solve_res);
                Err(GLPStatus::Unsolved)
            }
        }
    }
}

impl Drop for GLPProblem {
    fn drop(&mut self) {
        unsafe {
            glp_delete_prob(self.prob);
        }
    }
}

// Based on http://www2.in.tum.de/bib/files/simon05exploiting.pdf

/// A polyhedron, i.e., a set of linear constraints.
#[derive(Clone, Debug)]
pub struct Polyhedron {
    // Constraint representation: a x <= b
    a: DMatrix<f64>,
    b: DVector<f64>,
}

impl PartialEq for Polyhedron {
    fn eq(&self, other: &Polyhedron) -> bool {
        if self.dims() != other.dims() {
            return false;
        }
        self.includes(other) && other.includes(self)
    }
}

impl Eq for Polyhedron {}

fn eliminate_column(a: &mut DMatrix<f64>, b: &mut DVector<f64>, col: usize) {
    let mut cs: Vec<RowDVector<f64>> = Vec::new();
    let mut bs: Vec<f64> = Vec::new();
    for (i, ri) in a.row_iter().enumerate() {
        if ri[col] == 0. {
            cs.push(RowDVector::from(ri).remove_column(col));
            bs.push(b[i]);
        }
        for (k, rj) in a.rows_range(i + 1..).row_iter().enumerate() {
            let j = k + i + 1;
            if ri[col] * rj[col] < 0. {
                cs.push((rj[col].abs() * ri + ri[col].abs() * rj).remove_column(col));
                bs.push(rj[col].abs() * b[i] + ri[col].abs() * b[j]);
            }
        }
    }
    *a = DMatrix::from_rows(&cs);
    *b = DVector::from_vec(bs);
}

impl Polyhedron {
    fn minimize(&mut self) {
        loop {
            if self.a.nrows() == 0 {
                break;
            }
            let mut elim = false;
            for (i, r) in self.a.row_iter().enumerate() {
                let mut prob = GLPProblem::new();
                let mat = self.a.clone().remove_row(i);
                let vec = self.b.clone().remove_row(i);
                prob.load_matrix(&mat, &vec);
                // TODO: This can't be the best way to pass the row.
                prob.set_objective(&r.iter().cloned().collect::<Vec<f64>>(), 0.);
                let s = prob.solve();
                match s {
                    Err(_) => continue,
                    Ok(_) => {
                        elim = true;
                        replace_with_or_abort(&mut self.a, |a| a.remove_row(i));
                        replace_with_or_abort(&mut self.b, |b| b.remove_row(i));
                        break;
                    }
                };
            }
            if !elim {
                break;
            }
        }
    }

    fn includes(&self, other: &Polyhedron) -> bool {
        if self.a.nrows() == 0 {
            // This polyhedron is top, so it includes everything.
            return true;
        } else if other.a.nrows() == 0 {
            // Other is top and this isn't
            return false;
        }
        for (i, r) in self.a.row_iter().enumerate() {
            let mut prob = GLPProblem::new();
            prob.load_matrix(&other.a, &other.b);
            // TODO: This can't be the best way to pass the row.
            prob.set_objective(&r.iter().cloned().collect::<Vec<f64>>(), 0.);
            let s = prob.solve();
            match s {
                Err(_) => return false,
                Ok(v) => {
                    if v > self.b[i] {
                        return false;
                    }
                }
            };
        }
        true
    }
}

impl AbstractDomain for Polyhedron {
    fn top(dims: usize) -> Polyhedron {
        Polyhedron {
            a: DMatrix::from_vec(0, dims, vec![]),
            b: DVector::from_vec(vec![]),
        }
    }

    fn bottom(dims: usize) -> Polyhedron {
        Polyhedron {
            a: DMatrix::from_vec(1, dims, vec![0.; dims]),
            b: DVector::from_vec(vec![-1.]),
        }
    }

    fn join(&self, other: &Polyhedron) -> Polyhedron {
        if self.dims() != other.dims() {
            panic!("Mismatched dimensionality in Polyhedron::join");
        }
        // Introduce new variables s1, s2 in R, and y1, y2 in R^n and construct the system
        // A y1 - s1 b <= 0 and A y2 - s2 b <= 0 and -s1 <= 0 and -s2 <= 0 and x - y1 - y2 = 0
        // and s1 + s2 = 1, then project out y1, y2, s1, and s1. This yields the following system,
        // from which we must project out columns Y1 - s2
        //  X Y1 Y2 s1 s2  |  b
        // --------------- | ---
        //  0  A  0 -b  0  |  0
        //  0  0  A  0 -b  |  0
        //  0  0  0 -1  0  |  0
        //  0  0  0  0 -1  |  0
        //  I  I  I  0  0  |  0
        // -I -I -I  0  0  |  0
        //  0  0  0  1  1  |  1
        //  0  0  0 -1 -1  | -1
        let dims = self.a.ncols();
        let cs = self.a.nrows();
        let i = DMatrix::identity(dims, dims);
        let neg_i = -1. * i.clone();
        let neg_b = -1. * self.b.clone();
        let mut a = DMatrix::from_element(2 * cs + 2 * dims + 4, 3 * dims + 2, 0.);
        let mut b = DVector::from_element(2 * cs + 2 * dims + 4, 0.);
        a.slice_mut((0, dims), (cs, dims))
            .copy_from_slice(self.a.as_slice());
        a.slice_mut((0, 3 * dims), (cs, 1))
            .copy_from_slice(neg_b.as_slice());
        a.slice_mut((cs, 2 * dims), (cs, dims))
            .copy_from_slice(self.a.as_slice());
        a.slice_mut((cs, 3 * dims + 2), (cs, 1))
            .copy_from_slice(neg_b.as_slice());
        a[(2 * cs, 3 * dims)] = -1.;
        a[(2 * cs + 1, 3 * dims + 1)] = -1.;
        a.slice_mut((2 * cs + 2, 0), (dims, dims))
            .copy_from_slice(i.as_slice());
        a.slice_mut((2 * cs + 2, dims), (dims, dims))
            .copy_from_slice(i.as_slice());
        a.slice_mut((2 * cs + 2, 2 * dims), (dims, dims))
            .copy_from_slice(i.as_slice());
        a.slice_mut((2 * cs + 2 + dims, 0), (dims, dims))
            .copy_from_slice(neg_i.as_slice());
        a.slice_mut((2 * cs + 2 + dims, dims), (dims, dims))
            .copy_from_slice(neg_i.as_slice());
        a.slice_mut((2 * cs + 2 + dims, 2 * dims), (dims, dims))
            .copy_from_slice(neg_i.as_slice());
        a[(2 * cs + 2 + 2 * dims, 3 * dims)] = 1.;
        a[(2 * cs + 2 + 2 * dims, 3 * dims + 1)] = 1.;
        a[(2 * cs + 2 + 3 * dims, 3 * dims)] = -1.;
        a[(2 * cs + 2 + 3 * dims, 3 * dims + 1)] = -1.;
        b[2 * cs + 2 + 3 * dims] = 1.;
        b[2 * cs + 3 + 3 * dims] = -1.;
        for k in (dims..3 * dims + 2).rev() {
            eliminate_column(&mut a, &mut b, k);
        }
        let mut p = Polyhedron { a, b };
        p.minimize();
        p
    }

    fn meet(&self, other: &Polyhedron) -> Polyhedron {
        if self.dims() != other.dims() {
            panic!("Mismatched dimensionality in Polyhedron::meet");
        }
        let mut a = self.a.clone();
        a.resize_vertically_mut(other.a.nrows(), 0.);
        a.slice_mut((self.a.nrows(), 0), (other.a.nrows(), other.a.ncols()))
            .copy_from_slice(other.a.as_slice());
        let b = DVector::from_iterator(
            self.b.nrows() + other.b.nrows(),
            self.b
                .column(0)
                .iter()
                .cloned()
                .chain(other.b.column(0).iter().cloned()),
        );
        let mut p = Polyhedron { a, b };
        p.minimize();
        p
    }

    fn is_top(&self) -> bool {
        // If there are any constraints then this is not top.
        self.a.nrows() == 0
    }

    fn is_bottom(&self) -> bool {
        if self.a.nrows() == 0 {
            // This is top.
            return false;
        }
        let mut prob = GLPProblem::new();
        prob.load_matrix(&self.a, &self.b);
        prob.set_objective(&vec![1.; self.a.ncols()], 0.);
        let sol = prob.solve();
        println!("{:?}", sol);
        match sol {
            Ok(_) => false,
            Err(e) => match e {
                GLPStatus::Infeasible => true,
                _ => false,
            },
        }
    }

    fn remove_dims<I>(&self, dims: I) -> Polyhedron
    where
        I: IntoIterator<Item = usize>,
    {
        let mut to_remove: Vec<usize> = dims.into_iter().collect();
        to_remove.sort();
        to_remove.reverse();
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        for d in to_remove {
            eliminate_column(&mut a, &mut b, d);
        }
        let mut p = Polyhedron { a, b };
        p.minimize();
        p
    }

    fn add_dims<I>(&self, dims: I) -> Polyhedron
    where
        I: IntoIterator<Item = usize>,
    {
        // To add an unconstrained variable, we simply add a new column of zeros so that it is not
        // restricted by any existing constraint.
        let mut to_add: Vec<usize> = dims.into_iter().collect();
        to_add.sort();
        for (i, d) in to_add.iter_mut().enumerate() {
            *d += i;
        }
        let mut a = self.a.clone();
        for d in to_add {
            let t = if d > a.ncols() { a.ncols() } else { d };
            a = a.insert_column(t, 0.);
        }
        Polyhedron {
            a,
            b: self.b.clone(),
        }
    }

    fn dims(&self) -> usize {
        self.a.ncols()
    }
}

impl NumericalDomain for Polyhedron {
    fn assign(&self, trans: &HashMap<usize, AffineTransform>) -> Polyhedron {
        // A x <= b then x <- C x + d. We need to find A', b' such that
        // A x <= b <-> A' (C x + d) <= b'.
        //          <-> A' C x + A' d <= b'
        //          <-> A' C x <= b' - A' d
        // So we should have A = A' C and b = b' - A' d
        // In general if C is not invertible this might be difficult, so we'll rely on existing
        // tools instead. Specifically, for each transformation (c, d) we introduce a new variable
        // e and add the constraint e = c^T x + d. Then we project out the old values of each
        // variable to be eliminated and move e for each variable into the appropriate place.
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        // Keep a map from locations to the new variables which will be used to fill them.
        let mut map: HashMap<usize, usize> = HashMap::new();
        for (dim, at) in trans {
            let n = a.nrows();
            let c = a.ncols();
            a = a.insert_rows(n, 2, 0.);
            a.row_mut(n).copy_from_slice(&at.coeffs);
            a.row_mut(n)
                .component_mul_assign(&RowDVector::from_element(c, -1.));
            a.row_mut(n + 1).copy_from_slice(&at.coeffs);
            a = a.insert_column(c, 0.);
            map.insert(*dim, c);
            a[(n, c)] = 1.;
            a[(n + 1, c)] = -1.;
            b = b.insert_rows(n, 2, 0.);
            b[n] = at.cst;
            b[n + 1] = -at.cst;
        }
        // Project out the old variables
        let mut ks: Vec<usize> = map.keys().cloned().collect();
        ks.sort();
        ks.reverse();
        for k in ks {
            eliminate_column(&mut a, &mut b, k);
        }
        let dims = self.dims();
        // Swap the new columns back to where they're supposed to be.
        for (i, (d, v)) in map.iter().enumerate() {
            a = a.insert_column(*d, 0.);
            a.swap_columns(*d, i + v - dims);
            a = a.remove_column(i + v - dims);
        }
        let mut p = Polyhedron { a, b };
        p.minimize();
        p
    }

    fn constrain<'a, I>(&self, cnts: I) -> Polyhedron
    where
        I: IntoIterator<Item = &'a LinearConstraint>,
        I::IntoIter: Clone,
    {
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        for lc in cnts {
            let n = a.nrows();
            a = a.insert_row(n, 0.);
            b = b.insert_row(n, 0.);
            a.row_mut(n).copy_from_slice(&lc.coeffs);
            b[n] = lc.cst;
        }
        let mut p = Polyhedron { a, b };
        p.minimize();
        p
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_eliminate() {
        let mut a = DMatrix::<f64>::from_row_slice(
            5,
            4,
            &vec![
                0., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 1., -1., 0., -2., -1., 1., 0., 4.,
            ],
        );
        let mut b = DVector::<f64>::from_vec(vec![0., 1., 0., 0., 2.]);
        eliminate_column(&mut a, &mut b, 3);
        let x1 = DVector::<f64>::from_vec(vec![1., 1., 1.]);
        let x2 = DVector::<f64>::from_vec(vec![1., 1., -1.]);
        let x3 = DVector::<f64>::from_vec(vec![3., 0., 1.]);
        let x4 = DVector::<f64>::from_vec(vec![0., 3., 1.]);
        assert!(&a * x1 <= b);
        assert!(!(&a * x2 <= b));
        assert!(!(&a * x3 <= b));
        assert!(!(&a * x4 <= b));
    }

    #[test]
    fn test_top_bottom() {
        let pt: Polyhedron = AbstractDomain::top(3);
        let pb: Polyhedron = AbstractDomain::bottom(3);
        assert!(pt.is_top());
        assert!(!pb.is_top());
        assert!(pb.is_bottom());
        assert!(!pt.is_bottom());
    }

    #[test]
    fn minimize() {
        let a = DMatrix::from_vec(3, 2, vec![1., 0., 0., 1., 2., 0.]);
        let b = DVector::from_vec(vec![2., 2., 2.]);
        let mut p = Polyhedron { a, b };
        p.minimize();
        assert_eq!(p.a.nrows(), 2);
    }

    #[test]
    fn test_eq() {
        let a = DMatrix::from_vec(3, 2, vec![1., 0., 0., 1., 2., 0.]);
        let b = DVector::from_vec(vec![2., 2., 2.]);
        let mut p = Polyhedron { a, b };
        let p1 = p.clone();
        p.minimize();
        assert_eq!(p, p1);
    }

    #[test]
    fn test_dims() {
        let a: Polyhedron = AbstractDomain::top(4);
        assert_eq!(a.dims(), 4);
        let b: Polyhedron = AbstractDomain::bottom(2);
        assert_eq!(b.dims(), 2);
    }

    #[test]
    fn test_add_dims() {
        let a: Polyhedron = AbstractDomain::top(4);
        let b: Polyhedron = AbstractDomain::top(3);
        let c = Polyhedron {
            a: DMatrix::from_vec(1, 2, vec![1., 1.]),
            b: DVector::from_vec(vec![2.]),
        };
        let d = Polyhedron {
            a: DMatrix::from_vec(1, 4, vec![0., 1., 1., 0.]),
            b: DVector::from_vec(vec![2.]),
        };
        assert_eq!(b.add_dims(vec![0]).dims(), 4);
        assert_eq!(b.add_dims(vec![4]), a);
        assert_eq!(c.add_dims(vec![0, 3]), d);
    }
}
