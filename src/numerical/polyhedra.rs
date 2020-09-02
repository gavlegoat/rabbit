use nalgebra::{DMatrix, DVector, RowDVector};

// Based on https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotHalbwachs-POPL-78-ACM-p84--97-1978.pdf

/// TODO
pub struct Polyhedron {
    // Constraint representation: a x <= b
    a: DMatrix<f64>,
    b: DVector<f64>,
    // Frame representation
    vertices: Vec<DVector<f64>>,
    rays: Vec<DVector<f64>>,
    lines: Vec<DVector<f64>>,
    // Keep track of which representation is current. This facilitates lazy conversion between the
    // frame and constraint representations as needed. At least one of these two should always be
    // true.
    constraints_updated: bool,
    frame_updated: bool,
    // The number of dimensions in the constrained space.
    dims: usize,
}

fn eliminate_column(a: &mut DMatrix<f64>, b: &mut DVector<f64>, col: usize) {
    let mut cs: Vec<RowDVector<f64>> = Vec::new();
    let mut bs: Vec<f64> = Vec::new();
    for (i, ri) in a.row_iter().enumerate() {
        if ri[col] == 0. {
            cs.push(RowDVector::from(ri).remove_column(col));
            bs.push(b[i]);
        }
        for (k, rj) in a.rows_range(i+1..).row_iter().enumerate() {
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
    /// TODO: This doesn't actually need to be public.
    pub fn generate_constraints(&mut self) {
        if self.constraints_updated {
            return;
        }
        self.constraints_updated = true;
        if self.vertices.len() == 0 {
            // This is an empty polytope
            self.a = DMatrix::zeros(1, self.dims);
            self.b = DVector::from_element(1, -1.);
            return;
        }
        let mut a: DMatrix<f64> = DMatrix::from_fn(2 * self.dims, self.dims,
            |i, j| if i == 2 * j { 1.0 } else if i == 2 * j + 1 { -1.0 } else { 0.0 });
        let mut b: DVector<f64> = DVector::from_fn(2 * self.dims,
            |i, _| if i % 2 == 0 { self.vertices[0][i/2] } else { -self.vertices[0][i/2] });
        // Now the i'th pair of rows in a and b define constraints x_i <= v_i and -x_i <= -v_i.
        for v in &self.vertices {
            // Add constraints 0 <= lambda <= 1, convert the system to
            // A x + lambda (A v - b) <= A v, then eliminate lambda.
            a.extend(std::iter::once(&a * v - &b));
            let i = a.nrows();
            a = a.insert_rows(i, 2, 0.);
            let rs = a.nrows();
            let cs = a.ncols();
            a[(rs - 2, cs - 1)] = 1.;
            a[(rs - 1, cs - 1)] = -1.;
            b = b.insert_rows(i, 2, 0.);
            b[rs - 2] = 1.;
            eliminate_column(&mut a, &mut b, cs - 1);
        }
        for r in &self.rays {
            // Add constraints 0 <= mu, convert to A x - mu A r <= b, then eliminate mu.
            a.extend(std::iter::once(- &a * r));
            let i = a.nrows();
            a = a.insert_row(i, 0.);
            let rs = a.nrows();
            let cs = a.ncols();
            a[(rs - 1, cs - 1)] = -1.;
            b = b.insert_row(i, 0.);
            eliminate_column(&mut a, &mut b, cs - 1);
        }
        for d in &self.lines {
            // Convert to A x - nu A d <= b then eliminate nu.
            a.extend(std::iter::once(- &a * d));
            let cs = a.ncols();
            eliminate_column(&mut a, &mut b, cs - 1);
        }

        // Simplify the system of inequalities.
        // We can remove any constraints which are not saturated for some vertex.
        let mut to_remove: Vec<usize> = Vec::new();
        for (i, (r, c)) in a.row_iter().zip(b.iter()).enumerate() {
            let mut remove = true;
            for v in &self.vertices {
                if (r.dot(&v) - c).abs() < 1e-7 {
                    remove = false;
                    break;
                }
            }
            if remove {
                to_remove.push(i);
            }
        }
        a = a.remove_rows_at(&to_remove);
        b = b.remove_rows_at(&to_remove);
        // For rows i, j, C_i <= C_j if for all vertices v, a_i s = b_i implies a_j s = b_j and for
        // all rays r, a_i r = 0 implies a_j r = 0. Then if C_i <= C_j and not C_j <= C_i, we can
        // remove C_i. If C_i <= C_j and C_j <= C_i, we can remove either one of C_i or C_j. We
        // will look for pairs i, j satisfying either C_i <= C_j or C_j <= C_i and remove then as
        // appropriate until we reach a fixed point.
        loop {
            let mut dim: Option<usize> = None;
            for (i, ri) in a.row_iter().enumerate() {
                for (j, rj) in a.row_iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    // Determine whether C_i <= C_j or C_j <= C_i
                    let mut i_lt_j = true;
                    let mut j_lt_i = true;
                    for v in &self.vertices {
                        if rj.dot(v) == b[j] && ri.dot(v) != b[i] {
                            i_lt_j = false;
                        }
                        if ri.dot(v) == b[i] && rj.dot(v) != b[j] {
                            j_lt_i = false;
                        }
                        if !i_lt_j && !j_lt_i {
                            break;
                        }
                    }
                    for r in &self.rays {
                        if rj.dot(r).abs() < 1e-7 && ri.dot(r).abs() < 1e-7 {
                            i_lt_j = false;
                            break;
                        }
                        if ri.dot(r).abs() < 1e-7 && rj.dot(r).abs() < 1e-7 {
                            j_lt_i = false;
                            break;
                        }
                        if !i_lt_j && !j_lt_i {
                            break;
                        }
                    }
                    if i_lt_j {
                        dim = Some(i);
                    } else if j_lt_i {
                        dim = Some(j);
                    }
                }
                if dim.is_some() {
                    break;
                }
            }
            if dim.is_none() {
                break;
            }
            a = a.remove_row(dim.unwrap());
            b = b.remove_row(dim.unwrap());
        }

        self.a = a;
        self.b = b;
    }

    /// TODO: This doesn't actually need to be public.
    pub fn generate_frame(&mut self) {
        if self.frame_updated {
            return;
        }
        self.frame_updated = true;
        // TODO
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_eliminate() {
        let mut a = DMatrix::<f64>::from_row_slice(5, 4, &vec![ 0.,  0.,  0., -1.,
                                                                0.,  0.,  0.,  1.,
                                                                0.,  0., -1.,  0.,
                                                                1., -1.,  0., -2.,
                                                               -1.,  1.,  0.,  4.]);
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
