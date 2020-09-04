use minilp::{ComparisonOp, OptimizationDirection, Problem};
use nalgebra::{DMatrix, DVector, RowDVector};
use replace_with::replace_with_or_abort;
use std::collections::HashMap;

use crate::numerical::{AffineTransform, LinearConstraint, NumericalDomain};
use crate::AbstractDomain;

// Based on http://www2.in.tum.de/bib/files/simon05exploiting.pdf

/// A polyhedron, i.e., a set of linear constraints.
pub struct Polyhedron {
    // Constraint representation: a x <= b
    a: DMatrix<f64>,
    b: DVector<f64>,
}

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
            let mut elim = false;
            for (i, r) in self.a.row_iter().enumerate() {
                let mut prob = Problem::new(OptimizationDirection::Maximize);
                let mut vars = Vec::new();
                for c in &r {
                    vars.push(prob.add_var(*c, (f64::NEG_INFINITY, f64::INFINITY)));
                }
                for (j, s) in self.a.row_iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    prob.add_constraint(
                        vars.iter().cloned().zip(s.iter().cloned()),
                        ComparisonOp::Le,
                        self.b[j],
                    );
                }
                let s = prob.solve();
                if s.is_err() {
                    continue;
                }
                if s.unwrap().objective() <= self.b[i] {
                    elim = true;
                    replace_with_or_abort(&mut self.a, |a| a.remove_row(i));
                    break;
                }
            }
            if !elim {
                break;
            }
        }
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
        let mut prob = Problem::new(OptimizationDirection::Maximize);
        let mut vars = Vec::new();
        for _ in self.a.column_iter() {
            vars.push(prob.add_var(1., (f64::NEG_INFINITY, f64::INFINITY)));
        }
        for (i, r) in self.a.row_iter().enumerate() {
            prob.add_constraint(
                vars.iter().cloned().zip(r.iter().cloned()),
                ComparisonOp::Le,
                self.b[i],
            );
        }
        match prob.solve() {
            Ok(_) => false,
            Err(e) => match e {
                minilp::Error::Infeasible => true,
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
            a = a.insert_column(d, 0.);
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
}
