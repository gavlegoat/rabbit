//! Provides abstractions over numerical, real-valued spaces.
//!
//! This module includes extra utilities for working with programs over real-valued data.
//! Specifically, it includes types and functions for working with linear transformations and
//! constraints. Note that currently this module uses floating point numbers to represent reals and
//! in general the components of this module are _not_ safe with respect to floating point errors.
//! Therefore this module is _not_ ready to be used in safety-critical contexts.

pub use crate::numerical::interval::Interval;
use crate::AbstractDomain;
use std::collections::HashMap;
pub mod interval;

/// An affine transformation is a linear combination of the program variables plus a constant. If
/// the program variables form a vector `x`, then the result of applying the affine transformation
/// is `x^T coeffs + cst`.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineTransform {
    /// The coefficients of the affine transformation.
    coeffs: Vec<f64>,
    /// The constant for the affine transformation.
    cst: f64,
}

impl AffineTransform {
    /// Create a new AffineTransform which maps all inputs to zero.
    ///
    /// # Arguments
    /// * `dims` - The number of dimensions in the space.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// let a = AffineTransform::from_coeffs(vec![0.; 3], 0.);
    /// assert_eq!(a, AffineTransform::zero(3));
    /// ```
    pub fn zero(dims: usize) -> AffineTransform {
        AffineTransform {
            coeffs: vec![0.; dims],
            cst: 0.,
        }
    }

    /// Create an AffineTransform from a given set of coefficients and a constant.
    ///
    /// # Arguments
    /// * `cfs` - The coefficients for each program variable.
    /// * `ct` - The constant to add.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// let a = AffineTransform::from_coeffs(vec![0.; 3], 0.);
    /// assert_eq!(a, AffineTransform::zero(3));
    /// ```
    pub fn from_coeffs<I>(cfs: I, ct: f64) -> AffineTransform
    where
        I: IntoIterator<Item = f64>,
    {
        AffineTransform {
            coeffs: cfs.into_iter().collect(),
            cst: ct,
        }
    }

    /// Modify one of the coefficients of the AffineTransform.
    ///
    /// # Arguments
    /// * `d` - The dimension to modify.
    /// * `c` - The new value of the coefficient.
    ///
    /// # Panics
    /// Panics if `d` is greater than the number of dimensions in `self`.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// let mut a = AffineTransform::from_coeffs(vec![0., 1., 0.], 0.);
    /// a.update_coeff(1, 0.);
    /// assert_eq!(a, AffineTransform::zero(3));
    /// ```
    /// ```should_panic
    /// # use rabbit::numerical::*;
    /// # let mut a = AffineTransform::from_coeffs(vec![0., 1., 0.], 0.);
    /// // Should panic
    /// a.update_coeff(3, 0.);
    /// ```
    pub fn update_coeff(&mut self, d: usize, c: f64) {
        self.coeffs[d] = c;
    }

    /// Modify the constant of this AffineTransform.
    ///
    /// # Arguments
    /// * `c` - The new value of the constant.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// let mut a = AffineTransform::from_coeffs(vec![0.; 3], 2.);
    /// a.update_cst(0.);
    /// assert_eq!(a, AffineTransform::zero(3));
    /// ```
    pub fn update_cst(&mut self, c: f64) {
        self.cst = c;
    }

    /// Get the dimension of the space this transformation operates on.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// assert_eq!(AffineTransform::zero(3).dims(), 3);
    /// ```
    pub fn dims(&self) -> usize {
        self.coeffs.len()
    }
}

/// A linear constraint is a predicate of the form `x^T c <= b`.
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    coeffs: Vec<f64>,
    cst: f64,
}

impl PartialEq for LinearConstraint {
    fn eq(&self, other: &LinearConstraint) -> bool {
        // If coeffs is all zeros for both transformations and the constants have the same sign,
        // then these two constraints are equal. Otherwise, constraints are equal if they are a
        // constant multiple of each other.
        let mut all_zero = true;
        if self.coeffs.len() != other.coeffs.len() {
            return false;
        }
        for c in self.coeffs.iter().chain(other.coeffs.iter()) {
            if *c != 0. {
                all_zero = false;
                break;
            }
        }
        if all_zero {
            (self.cst >= 0. && other.cst >= 0.) || (self.cst < 0. && other.cst < 0.)
        } else {
            let rat = self.coeffs[0] / other.coeffs[0];
            for (a, b) in self.coeffs.iter().zip(other.coeffs.iter()) {
                if a / b != rat {
                    return false;
                }
            }
            self.cst / other.cst == rat
        }
    }
}

impl LinearConstraint {
    /// Create a new `LinearConstraint` which is satisfied for every point. Specifically, the
    /// constructs the linear constraint 0 <= 1.
    ///
    /// # Arguments
    /// * `dims` - The number of dimensions in the constrained space.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// let lc = LinearConstraint::from_coeffs(vec![0.; 3], 2.);
    /// assert_eq!(lc, LinearConstraint::unconstrained(3));
    /// ```
    pub fn unconstrained(dims: usize) -> LinearConstraint {
        LinearConstraint {
            coeffs: vec![0.; dims],
            cst: 1.,
        }
    }

    /// Construct a new `LinearConstraint` given coefficients and a constant. The constraint will
    /// be satisfied whenever `x^T cfs <= ct`.
    ///
    /// # Arguments
    /// * `cfs` - The coefficients of the linear constraint.
    /// * `ct` - The constant of the linear constraint.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// let lc = LinearConstraint::from_coeffs(vec![0.; 3], 2.);
    /// assert_eq!(lc, LinearConstraint::unconstrained(3));
    /// ```
    pub fn from_coeffs<I>(cfs: I, ct: f64) -> LinearConstraint
    where
        I: IntoIterator<Item = f64>,
    {
        LinearConstraint {
            coeffs: cfs.into_iter().collect(),
            cst: ct,
        }
    }

    /// Get the dimension of the space this constraint operates on.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::*;
    /// assert_eq!(LinearConstraint::unconstrained(3).dims(), 3);
    /// ```
    pub fn dims(&self) -> usize {
        self.coeffs.len()
    }
}

/// Abstract domains over a numerical space.
pub trait NumericalDomain: AbstractDomain {
    /// Modify the space by applying a set of affine transformations in parallel. The
    /// transformations are given as a HashMap associating dimensions with transformations. Each
    /// dimensions which is present in the HashMap is replaced by the evaluation of the associated
    /// transformation. These assignments are all performed in parallel, i.e., using the old values
    /// for every transformation.
    ///
    /// # Arguments
    /// * `trans` - The set of transformations to apply.
    ///
    /// # Panics
    /// Panics if any of the keys in `trans` are greater than the dimension of `self`, or if any of
    /// the transformations have dimension different from `self`.
    ///
    /// # Example
    /// ```
    /// # use std::collections::HashMap;
    /// # use rabbit::numerical::*;
    /// # use rabbit::numerical::interval::*;
    /// let mut trans = HashMap::new();
    /// trans.insert(0, AffineTransform::zero(2));
    /// let a = Interval::from_bounds(vec![-1., 1.], vec![1., 2.]);
    /// let res = Interval::from_bounds(vec![0., 1.], vec![0., 2.]);
    /// assert_eq!(a.assign(&trans), res);
    /// ```
    fn assign(&self, trans: &HashMap<usize, AffineTransform>) -> Self;

    /// Meet the abstract element with a set of linear constraints. This returns an
    /// overapproximation of the states in the current abstract element which also satisfy all of
    /// the given linear constraints.
    ///
    /// # Arguments
    /// * `cnts` - The constraints to restrict the space with.
    ///
    /// # Panics
    /// Panics if any of the linear constraints have different dimension from `self`.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::*;
    /// # use rabbit::numerical::*;
    /// # use rabbit::numerical::interval::*;
    /// let a: Interval = AbstractDomain::top(2);
    /// let lc1 = LinearConstraint::from_coeffs(vec![0., 1.], 2.);
    /// let res1 = Interval::from_vec(
    ///     vec![Itv::from_bounds(LowerBound::NegInf, UpperBound::PosInf),
    ///          Itv::from_bounds(LowerBound::NegInf, UpperBound::Value(2.))]);
    /// assert_eq!(a.constrain(vec![lc1].iter()), res1);
    /// let lc2 = LinearConstraint::from_coeffs(vec![0., 0.], -1.);
    /// assert!(a.constrain(vec![lc2].iter()).is_bottom());
    /// ```
    fn constrain<'a, I>(&self, cnts: I) -> Self
    where
        I: Iterator<Item = &'a LinearConstraint> + Clone;
}

/// Create a new element of a numerical domain from a given set of linear constraints. This is
/// constructed by first creating top, then constraining the element.
///
/// # Arguments
/// * `dims` - The dimension of the constrained space.
/// * `cnts` - The constraints to satisfy.
///
/// # Panics
/// Panics if any of the linear constraints have dimension different from `dims`.
///
/// # Example
///
/// ```
/// # use rabbit::numerical::*;
/// # use rabbit::numerical::interval::*;
/// let lc = vec![LinearConstraint::from_coeffs(vec![1., 0.], 2.),
///               LinearConstraint::from_coeffs(vec![0., -1.], -3.)];
/// let itv: Interval = from_lincons(2, lc.iter());
/// let res = Interval::from_vec(
///     vec![Itv::from_bounds(LowerBound::NegInf, UpperBound::Value(2.)),
///          Itv::from_bounds(LowerBound::Value(3.), UpperBound::PosInf)]);
/// assert_eq!(itv, res);
/// ```
pub fn from_lincons<'a, D, I>(dims: usize, cnts: I) -> D
where
    I: Iterator<Item = &'a LinearConstraint> + Clone,
    D: NumericalDomain,
{
    D::top(dims).constrain(cnts)
}
