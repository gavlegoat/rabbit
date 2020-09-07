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

// This polyhedra implementation is based on http://www2.in.tum.de/bib/files/simon05exploiting.pdf

// Include generated bindings for GLPK
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Define a safe interface for GLPK problems.
struct GLPProblem {
    prob: *mut glp_prob,
}

// For our purposes it is sufficient to distinguish between infeasibility, unboundedness, unsolved
// (for any other reason) or finite solutions.
#[derive(PartialEq, Debug, Clone)]
enum GLPResult {
    Infeasible,
    Unbounded,
    Unsolved,
    Solved(f64),
}

impl GLPProblem {
    fn new() -> GLPProblem {
        GLPProblem {
            prob: unsafe { glp_create_prob() },
        }
    }

    // Load a problem defined as A x <= b into GLPK.
    fn load_matrix(&mut self, mat: &DMatrix<f64>, vec: &DVector<f64>) {
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
            for i in 1..nc + 1 {
                glp_set_col_bnds(
                    self.prob,
                    i as c_int,
                    GLP_FR as c_int,
                    0. as c_double,
                    0. as c_double,
                );
            }
        }
    }

    // Set the objective function to vec^T x + cst
    fn set_objective(&mut self, vec: &[f64], cst: f64) {
        unsafe {
            glp_set_obj_coef(self.prob, 0 as c_int, cst as c_double);
            for (i, v) in vec.iter().enumerate() {
                glp_set_obj_coef(self.prob, (i + 1) as c_int, *v as c_double);
            }
            glp_set_obj_dir(self.prob, GLP_MAX as c_int);
        }
    }

    // Solve a problem
    fn solve(&mut self) -> GLPResult {
        unsafe {
            let mut params = {
                let mut x = MaybeUninit::uninit();
                glp_init_smcp(x.as_mut_ptr());
                x.assume_init()
            };
            params.msg_lev = GLP_MSG_OFF as c_int;
            let simplex_res = glp_simplex(self.prob, &params);
            if simplex_res != 0 {
                return GLPResult::Unsolved;
            }
            let solve_res = glp_get_status(self.prob) as u32;
            if solve_res == GLP_NOFEAS {
                GLPResult::Infeasible
            } else if solve_res == GLP_UNBND {
                GLPResult::Unbounded
            } else if solve_res == GLP_OPT {
                let ret = glp_get_obj_val(self.prob);
                GLPResult::Solved(ret as f64)
            } else {
                GLPResult::Unsolved
            }
        }
    }
}

// We need to be sure to free the problem object since it was allocated from within C.
impl Drop for GLPProblem {
    fn drop(&mut self) {
        unsafe {
            glp_delete_prob(self.prob);
        }
    }
}

/// A polyhedron, i.e., a set of linear constraints.
#[derive(Clone, Debug)]
pub struct Polyhedron {
    // Constraint representation: a x <= b
    a: DMatrix<f64>,
    b: DVector<f64>,
}

// Equality is handled by double inclusion and in general checking equality can be slow.
impl PartialEq for Polyhedron {
    fn eq(&self, other: &Polyhedron) -> bool {
        if self.dims() != other.dims() {
            return false;
        }
        (self.is_bottom() && other.is_bottom()) || (self.includes(other) && other.includes(self))
    }
}

impl Eq for Polyhedron {}

// Project out a variable from a system of inequalities. See Fourier-Motzkin method.
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
    // Minimize the representation of this polyhedron by removing redundant constraints. In general
    // many operations on polyhedra can introduce new constraints that are already implied by
    // others, and these extra constraints can slow down analysis.
    fn minimize(&mut self) {
        loop {
            if self.a.nrows() <= 1 {
                break;
            } else if self.is_top() {
                replace_with_or_abort(&mut self.a, |a| DMatrix::zeros(0, a.ncols()));
                replace_with_or_abort(&mut self.b, |_| DVector::zeros(0));
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
                    GLPResult::Solved(v) => {
                        if v <= self.b[i] {
                            elim = true;
                            replace_with_or_abort(&mut self.a, |a| a.remove_row(i));
                            replace_with_or_abort(&mut self.b, |b| b.remove_row(i));
                            break;
                        }
                    }
                    _ => continue,
                };
            }
            if !elim {
                break;
            }
        }
    }

    // Determine whether this polyhedron includes anothe by checking whether all of the constraints
    // of this polyhedron are implied by other.
    fn includes(&self, other: &Polyhedron) -> bool {
        if self.is_top() {
            return true;
        } else if other.is_top() {
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
                GLPResult::Solved(v) => {
                    if v > self.b[i] {
                        return false;
                    }
                }
                _ => return false,
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
        if self.is_top() || other.is_top() {
            // Polyhedra with zero constraints seem to mess up the algorithm
            return AbstractDomain::top(self.dims());
        }
        if self.is_bottom() {
            return other.clone();
        } else if other.is_bottom() {
            return self.clone();
        }
        // Introduce new variables s1, s2 in R, and y1, y2 in R^n and construct the system
        // A1 y1 - s1 b1 <= 0 and A2 y2 - s2 b2 <= 0 and -s1 <= 0 and -s2 <= 0 and x - y1 - y2 = 0
        // and s1 + s2 = 1, then project out y1, y2, s1, and s1. This yields the following system,
        // from which we must project out columns Y1 - s2
        //  X Y1 Y2 s1 s2  |  b
        // --------------- | ---
        //  0 A1  0 -b1 0  |  0
        //  0  0 A2  0 -b2 |  0
        //  0  0  0 -1  0  |  0
        //  0  0  0  0 -1  |  0
        //  I -I -I  0  0  |  0
        // -I  I  I  0  0  |  0
        //  0  0  0  1  1  |  1
        //  0  0  0 -1 -1  | -1
        //
        // Total size: s_cs + o_cs + 2 + 2 * dims + 2

        let dims = self.a.ncols();
        let s_cs = self.a.nrows();
        let o_cs = other.a.nrows();
        let i = DMatrix::identity(dims, dims);
        let neg_i = -1. * i.clone();
        let s_neg_b = -1. * self.b.clone();
        let o_neg_b = -1. * other.b.clone();
        let mut a = DMatrix::from_element(s_cs + o_cs + 2 * dims + 4, 3 * dims + 2, 0.);
        let mut b = DVector::from_element(s_cs + o_cs + 2 * dims + 4, 0.);
        a.slice_mut((0, dims), (s_cs, dims))
            .copy_from_slice(self.a.as_slice());
        a.slice_mut((0, 3 * dims), (s_cs, 1))
            .copy_from_slice(s_neg_b.as_slice());
        a.slice_mut((s_cs, 2 * dims), (o_cs, dims))
            .copy_from_slice(other.a.as_slice());
        a.slice_mut((s_cs, 3 * dims + 1), (o_cs, 1))
            .copy_from_slice(o_neg_b.as_slice());
        a[(s_cs + o_cs, 3 * dims)] = -1.;
        a[(s_cs + o_cs + 1, 3 * dims + 1)] = -1.;
        a.slice_mut((s_cs + o_cs + 2, 0), (dims, dims))
            .copy_from_slice(i.as_slice());
        a.slice_mut((s_cs + o_cs + 2, dims), (dims, dims))
            .copy_from_slice(neg_i.as_slice());
        a.slice_mut((s_cs + o_cs + 2, 2 * dims), (dims, dims))
            .copy_from_slice(neg_i.as_slice());
        a.slice_mut((s_cs + o_cs + 2 + dims, 0), (dims, dims))
            .copy_from_slice(neg_i.as_slice());
        a.slice_mut((s_cs + o_cs + 2 + dims, dims), (dims, dims))
            .copy_from_slice(i.as_slice());
        a.slice_mut((s_cs + o_cs + 2 + dims, 2 * dims), (dims, dims))
            .copy_from_slice(i.as_slice());
        a[(s_cs + o_cs + 2 + 2 * dims, 3 * dims)] = 1.;
        a[(s_cs + o_cs + 2 + 2 * dims, 3 * dims + 1)] = 1.;
        a[(s_cs + o_cs + 2 + 2 * dims + 1, 3 * dims)] = -1.;
        a[(s_cs + o_cs + 2 + 2 * dims + 1, 3 * dims + 1)] = -1.;
        b[s_cs + o_cs + 2 + 2 * dims] = 1.;
        b[s_cs + o_cs + 2 + 2 * dims + 1] = -1.;
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
        a.resize_vertically_mut(self.a.nrows() + other.a.nrows(), 0.);
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
        if self.a.nrows() == 0 {
            return true;
        }
        // This polyhedron can still be top if each coefficient in the constraint matrix is zero
        // and each element of the constraint vector is non-negative
        for v in self.a.iter() {
            if *v != 0. {
                return false;
            }
        }
        for v in self.b.iter() {
            if *v < 0. {
                return false;
            }
        }
        true
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
        match sol {
            GLPResult::Infeasible => true,
            _ => false,
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
        let dims = self.dims();
        for (dim, at) in trans {
            if at.coeffs.len() != self.dims() {
                panic!("Mismatched dimensionality in Polyhedron::assign");
            }
            if *dim >= self.dims() {
                panic!("Dimension too high in Polyhedron::assign");
            }
            let n = a.nrows();
            let c = a.ncols();
            a = a.insert_rows(n, 2, 0.);
            a.row_part_mut(n, dims).copy_from_slice(&at.coeffs);
            a.row_mut(n)
                .component_mul_assign(&RowDVector::from_element(c, -1.));
            a.row_part_mut(n + 1, dims).copy_from_slice(&at.coeffs);
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
        let mut removed = 0;
        println!("Constraints with new variables: {} {}", a, b);
        for k in &ks {
            eliminate_column(&mut a, &mut b, *k);
            removed += 1;
        }
        println!("Constraints after elimination: {} {}", a, b);
        // Swap the new columns back to where they're supposed to be.
        for d in ks {
            a = a.insert_column(d, 0.);
            a.swap_columns(d, map[&d] - removed);
            a = a.remove_column(map[&d] - removed);
        }
        println!("Constraints after swapping: {} {}", a, b);
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
    use crate::numerical::*;
    use crate::AbstractDomain;

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
    fn dims() {
        let a: Polyhedron = AbstractDomain::top(4);
        assert_eq!(a.dims(), 4);
        let b: Polyhedron = AbstractDomain::bottom(2);
        assert_eq!(b.dims(), 2);
    }

    #[test]
    fn add_dims() {
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

    #[test]
    fn test_from_lincons() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 2.),
                LinearConstraint::from_coeffs(vec![1., 0.], 3.),
            ],
        );
        let b: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![2., 0.], 6.),
                LinearConstraint::from_coeffs(vec![0., 3.], 6.),
            ],
        );
        assert_eq!(a, b);
        let t: Polyhedron = from_lincons(2, &vec![LinearConstraint::from_coeffs(vec![0., 0.], 1.)]);
        assert!(t.is_top());
        assert_eq!(t, AbstractDomain::top(2));
        let b: Polyhedron =
            from_lincons(2, &vec![LinearConstraint::from_coeffs(vec![0., 0.], -3.)]);
        assert!(b.is_bottom());
        assert_eq!(b, AbstractDomain::bottom(2));
    }

    #[test]
    fn join_bounded() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 2.),
                LinearConstraint::from_coeffs(vec![1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![0., -1.], 0.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 0.),
            ],
        );
        let b: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        // (0, 0), (3, 0), (0, 2), (3, 2) join (-3, -2), (-3, 4), (1, -2), (1, 4)
        //   -3-2-1 0 1 2 3
        //  4 +-------+ . .
        //  3 | . . . | . .
        //  2 | . . +-----+
        //  1 | . . | | . |
        //  0 | . . +-----+
        // -1 | . . . | . .
        // -2 +-------+ . .
        let c: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 1.], 5.),
                LinearConstraint::from_coeffs(vec![1., -1.], 3.),
            ],
        );
        assert_eq!(a.join(&b), c);
    }

    #[test]
    fn join_open() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 2.),
                LinearConstraint::from_coeffs(vec![1., 0.], 3.),
            ],
        );
        let b: Polyhedron = from_lincons(2, &vec![LinearConstraint::from_coeffs(vec![1., 1.], 0.)]);
        let c: Polyhedron = from_lincons(2, &vec![LinearConstraint::from_coeffs(vec![1., 1.], 5.)]);
        assert_eq!(a.join(&b), c);
        let d: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., -1.], 0.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 0.),
            ],
        );
        assert!(a.join(&d).is_top());
        let e: Polyhedron = from_lincons(2, &vec![LinearConstraint::from_coeffs(vec![1., 1.], 6.)]);
        assert_eq!(a.join(&e), e);
    }

    #[test]
    fn join_top() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 2.),
                LinearConstraint::from_coeffs(vec![1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![0., -1.], 0.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 0.),
            ],
        );
        let t: Polyhedron = AbstractDomain::top(2);
        assert!(a.join(&t).is_top());
        assert_eq!(t.join(&a), t);
    }

    #[test]
    fn join_bottom() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 2.),
                LinearConstraint::from_coeffs(vec![1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![0., -1.], 0.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 0.),
            ],
        );
        let b: Polyhedron = AbstractDomain::bottom(2);
        assert_eq!(a.join(&b), a);
    }

    #[test]
    fn meet_normal() {
        let a: Polyhedron =
            from_lincons(2, &vec![LinearConstraint::from_coeffs(vec![1., 1.], -1.)]);
        let b: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        let c: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![1., 1.], -1.),
            ],
        );
        assert_eq!(a.meet(&b), c);
    }

    #[test]
    fn meet_top() {
        let b: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        let t: Polyhedron = AbstractDomain::top(2);
        assert_eq!(b.meet(&t), b);
    }

    #[test]
    fn meet_bottom() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        let b: Polyhedron = AbstractDomain::bottom(2);
        assert!(b.meet(&a).is_bottom());
    }

    #[test]
    fn remove_vars() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        let b: Polyhedron = from_lincons(
            1,
            &vec![
                LinearConstraint::from_coeffs(vec![1.], 1.),
                LinearConstraint::from_coeffs(vec![-1.], 3.),
            ],
        );
        assert_eq!(a.remove_dims(vec![1]), b);
        let c: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![1., 1.], 2.),
                LinearConstraint::from_coeffs(vec![1., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., -1.], 2.),
            ],
        );
        let d: Polyhedron = from_lincons(
            1,
            &vec![
                LinearConstraint::from_coeffs(vec![1.], 2.),
                LinearConstraint::from_coeffs(vec![-1.], 2.),
            ],
        );
        assert_eq!(c.remove_dims(vec![0]), d);
    }

    #[test]
    fn assign() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        let mut tr: HashMap<usize, AffineTransform> = HashMap::new();
        tr.insert(1, AffineTransform::from_coeffs(vec![1., 1.], 0.));
        let b: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![1., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 1.], 4.),
            ],
        );
        assert_eq!(a.assign(&tr), b);
        tr.insert(0, AffineTransform::from_coeffs(vec![1., -1.], 0.));
        // -3 <= x <= 1, -2 <= y <= 4. Then we assign
        // x := x - y and y := x + y  in parallel
        // The bounding box is now -7 <= x <= 3, -5 <= y <= 5. Then we get vertices
        // (-7, 1), (3, -1), (-1, -5), and (-3, 5)
        let m: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![1., 1.], 2.),
                LinearConstraint::from_coeffs(vec![1., -1.], 4.),
                LinearConstraint::from_coeffs(vec![-1., 1.], 8.),
                LinearConstraint::from_coeffs(vec![-1., -1.], 6.),
            ],
        );
        assert_eq!(a.assign(&tr), m);
    }

    #[test]
    fn constrain_normal() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        let lc = LinearConstraint::from_coeffs(vec![1., 1.], -1.);
        let c: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
                LinearConstraint::from_coeffs(vec![1., 1.], -1.),
            ],
        );
        assert_eq!(a.constrain(&vec![lc]), c);
    }

    #[test]
    fn constrain_unsat() {
        let a: Polyhedron = from_lincons(
            2,
            &vec![
                LinearConstraint::from_coeffs(vec![0., 1.], 4.),
                LinearConstraint::from_coeffs(vec![1., 0.], 1.),
                LinearConstraint::from_coeffs(vec![0., -1.], 2.),
                LinearConstraint::from_coeffs(vec![-1., 0.], 3.),
            ],
        );
        let lc = LinearConstraint::from_coeffs(vec![0., 0.], -1.);
        assert!(a.constrain(&vec![lc]).is_bottom());
    }

    #[test]
    fn constrain_top() {
        let t: Polyhedron = AbstractDomain::top(2);
        let lc = LinearConstraint::from_coeffs(vec![1., 1.], -1.);
        let b: Polyhedron =
            from_lincons(2, &vec![LinearConstraint::from_coeffs(vec![1., 1.], -1.)]);
        assert_eq!(t.constrain(&vec![lc]), b);
    }
}
