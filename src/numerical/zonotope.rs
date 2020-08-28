// This is a modified version of the zonotope domain which is more intuitive to me.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};

use crate::AbstractDomain;
use crate::numerical::*;

/// TODO
pub struct Zonotope {
    mat: DMatrix<f64>,
    off: DVector<f64>,
    itv: Interval,
}

impl AbstractDomain for Zonotope {

    fn top(dims: usize) -> Zonotope {
        Zonotope {
            mat: DMatrix::from_row_slice(0, dims, &vec![]),
            off: DVector::zeros(0),
            itv: AbstractDomain::top(0),
        }
    }

    fn bottom(dims: usize) -> Zonotope {
        Zonotope {
            mat: DMatrix::from_row_slice(0, dims, &vec![]),
            off: DVector::zeros(0),
            itv: AbstractDomain::bottom(0),
        }
    }

    fn join(&self, other: &Zonotope) -> Zonotope {
        // TODO
    }

    fn meet(&self, other: &Zonotope) -> Zonotope {
        // TODO
    }

    fn is_top(&self) -> bool {
        self.itv.is_top()
    }

    fn is_bottom(&self) -> bool {
        self.itv.is_bottom()
    }

    fn remove_dims<I>(&self, dims: I) -> Zonotope
        where I: IntoIterator<Item = usize>
    {
        // We can simply remove the associated rows of the matrix and offset  (and maybe check for
        // noise symbols which are no longer used).
        // TODO: We can check to see if itv has any noise symbols which are no longer used.
        let d: Vec<usize> = dims.into_iter().collect();
        Zonotope {
            mat: self.mat.remove_rows_at(&d),
            off: self.off.remove_rows_at(&d),
            itv: self.itv,
        }
    }

    fn dims(&self) -> usize {
        self.itv.dims()
    }
}

impl NumericalDomain for Zonotope {

    fn assign(&self, trans: &HashMap<usize, AffineTransform>) -> Zonotope {
        // Suppose we expand trans into T x + d for a matrix T and vector d. Then we should have a
        // zonotope  T (M e + b) + d = (T M) e + (T b + d)
        let n = self.dims();
        let mut t = DMatrix::<f64>::identity(n, n);
        let mut d = DVector::<f64>::zeros(n);
        for (k, v) in trans {
            t.row_mut(*k).copy_from_slice(&v.coeffs);
            d[*k] = v.cst;
        }
        Zonotope {
            mat: t * self.mat,
            off: t * self.off + d,
            itv: self.itv,
        }
    }

    fn constrain<'a, I>(&self, cnts: I) -> Zonotope
        where I: Iterator<Item = &'a LinearConstraint> + Clone
    {
        // We need M e + o to satisfy A x <= d. Then A (M e + o) <= d --> (A M) e + A o <= d
        // --> (A M) e <= d - A o. Since e comes from an interval, we can apply the linear
        // constraint (A M) x <= (d - A o) to the underlying interval.
        let n = cnts.clone().count();
        let a = DMatrix::from_iterator(
            n, self.dims(), cnts.clone().map(|x| x.coeffs.iter()).flatten().cloned());
        let am = a * self.mat;
        let d = DVector::from_iterator(n, cnts.map(|x| x.cst));
        let b = d - a * self.off;
        let lcs = Vec::new();
        for (x, y) in am.row_iter().zip(b.iter()) {
            lcs.push(LinearConstraint::from_coeffs(x.iter().cloned(), *y));
        }
        Zonotope {
            mat: self.mat,
            off: self.off,
            itv: self.itv.constrain(lcs.iter()),
        }
    }
}
