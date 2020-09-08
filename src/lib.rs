#![crate_name = "rabbit"]
#![crate_type = "lib"]
#![warn(missing_docs)]

//! A library for abstract interpretation.
//!
//! The high-level interface is through the [`AbstractDomain`] trait, which includes the necessary
//! components for abstract interpretation (top, bottom, meet, join, etc.). This library provides a
//! few predefined abstract domains, or you can implement the [`AbstractDomain`] trait to get
//! access to abstract interpretation utilities over your own domains.
//!
//! [`AbstractDomain`]: ./trait.AbstractDomain.html

pub mod numerical;

pub use crate::disjunctive::Disjunction;
mod disjunctive;

/// Defines all of the basic functionality needed to perform abstract interpretation.
pub trait AbstractDomain {
    /// Create a new top element. The top element includes every possible value.
    ///
    /// # Arguments
    /// * `dims` - The number of dimensions in the constrained space.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let t: Interval = AbstractDomain::top(2);
    /// let a = Interval::from_vec(vec![ Itv::unbounded(), Itv::unbounded() ]);
    /// assert_eq!(t, a);
    /// ```
    fn top(dims: usize) -> Self;

    /// Create a new bottom element. The bottom element includes no values.
    ///
    /// # Arguments
    /// * `dims` - The number of dimensions in the constrained space.
    ///
    /// #Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let b: Interval = AbstractDomain::bottom(1);
    /// let a = Interval::from_vec(vec![ Itv::empty() ]);
    /// assert_eq!(a, b);
    /// ```
    fn bottom(dims: usize) -> Self;

    /// Compute the join of two abstract elements. The join is the smallest element in the abstract
    /// domain which includes all of the states from `self` and `other`.
    ///
    /// # Arguments
    /// * `other` - Another abstract element to join with.
    ///
    /// # Panics
    /// Panics if `self` and `other` do not have the same number of dimensions.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let a = Interval::from_doubles(vec![0., 0.], vec![2., 2.], true);
    /// let b = Interval::from_doubles(vec![1., 0.], vec![1., 3.], true);
    /// assert_eq!(a.join(&b), Interval::from_doubles(vec![0., 0.], vec![2., 3.], true));
    /// ```
    fn join(&self, other: &Self) -> Self;

    /// Compute the meet of two abstract elements. The meet is the largest element in the abstract
    /// domain which includes only states present in both `self` and `other`.
    ///
    /// # Arguments
    /// * `other` - Another abstract element to meet with.
    ///
    /// # Panics
    /// Panics if `self` and `other` do not have the same number of dimensions.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let a = Interval::from_doubles(vec![0., 0.], vec![2., 2.], true);
    /// let b = Interval::from_doubles(vec![1., 0.], vec![1., 3.], true);
    /// assert_eq!(a.meet(&b), Interval::from_doubles(vec![1., 0.], vec![1., 2.], true));
    /// ```
    fn meet(&self, other: &Self) -> Self;

    /// Determine whether this abstract element is top.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let t: Interval = AbstractDomain::top(3);
    /// let b: Interval = AbstractDomain::bottom(3);
    /// assert!(t.is_top());
    /// assert!(!b.is_top());
    /// ```
    fn is_top(&self) -> bool;

    /// Determine whether this abstract element is bottom.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let t: Interval = AbstractDomain::top(3);
    /// let b: Interval = AbstractDomain::bottom(3);
    /// assert!(b.is_bottom());
    /// assert!(!t.is_bottom());
    /// ```
    fn is_bottom(&self) -> bool;

    /// Add new variables to the constrained space. The new variables are assumed to be
    /// unconstrained. That is, the result will be an extrusion of the existing abstract value
    /// along the new variables.
    ///
    /// # Argumets
    /// * `dims` - The dimensions before which to add new dimensions. If a dimension appears
    ///   multiple times, multiple dimensions will be added in the same space. If an element of
    ///   dims is greater than the number of dimensions of the constrained space, then the new
    ///   dimension is added at the end.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let a = Interval::from_doubles(vec![0.], vec![1.], false);
    /// assert_eq!(a.add_dims(vec![0, 0]),
    ///     Interval::from_vec(
    ///         vec![Itv::unbounded(),
    ///              Itv::unbounded(),
    ///              Itv::from_double_open(0., 1.)]));
    /// assert_eq!(a.add_dims(vec![2]),
    ///     Interval::from_vec(
    ///         vec![Itv::from_double_open(0., 1.),
    ///              Itv::unbounded()]));
    /// ```
    fn add_dims<I>(&self, dims: I) -> Self
    where
        I: IntoIterator<Item = usize>;

    /// Remove some set of variables from the constrained space. This is a projection onto the
    /// remaining dimensions.
    ///
    /// # Arguments
    /// * `dims` - The set of dimensions to remove.
    ///
    /// # Panics
    /// Panics if `dims` contains an element greater than the number of dimensions in `self`.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let a = Interval::from_doubles(vec![1., 0.], vec![1., 3.], false);
    /// assert_eq!(a.remove_dims(vec![0]), Interval::from_doubles(vec![0.], vec![3.], false));
    /// ```
    fn remove_dims<I>(&self, dims: I) -> Self
    where
        I: IntoIterator<Item = usize>;

    /// Get the number of variables constrained by this element.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::*;
    /// let a: Interval = AbstractDomain::top(4);
    /// assert_eq!(a.dims(), 4);
    /// ```
    fn dims(&self) -> usize;
}

/// Join a set of abstract elements.
///
/// # Arguments
/// * `dims` - The number of dimensions in the abstract domain.
/// * `es` - The set of elements to find the join of.
///
/// # Panics
/// Panics if any element of `es` does not have `dims` dimensions.
///
/// # Examples
/// ```
/// # use rabbit::*;
/// # use rabbit::numerical::*;
/// let a = Interval::from_doubles(vec![0., 0.], vec![2., 2.], true);
/// let b = Interval::from_doubles(vec![1., 0.], vec![1., 3.], true);
/// let c = Interval::from_doubles(vec![-1., 1.], vec![1., 3.], true);
/// let res = Interval::from_doubles(vec![-1., 0.], vec![2., 3.], true);
/// assert_eq!(join_all(2, vec![a, b, c].iter()), res);
/// ```
pub fn join_all<'a, D: 'a, I>(dims: usize, es: I) -> D
where
    D: AbstractDomain,
    I: IntoIterator<Item = &'a D>,
{
    es.into_iter().fold(D::bottom(dims), |acc, x| acc.join(&x))
}

/// Meet a set of abstract elements.
///
/// # Arguments
/// * `dims` - The number of dimensions in the abstract domain.
/// * `es` - The set of elements to find the meet of.
///
/// # Panics
/// Panics if any element of `es` does not have `dims` dimensions.
///
/// # Examples
/// ```
/// # use rabbit::*;
/// # use rabbit::numerical::*;
/// let a = Interval::from_doubles(vec![0., 0.], vec![2., 2.], true);
/// let b = Interval::from_doubles(vec![1., 0.], vec![1., 3.], true);
/// let c = Interval::from_doubles(vec![-1., 1.], vec![1., 3.], true);
/// let res = Interval::from_doubles(vec![1., 1.], vec![1., 2.], true);
/// assert_eq!(meet_all(2, vec![a, b, c].iter()), res);
/// ```
pub fn meet_all<'a, D: 'a, I>(dims: usize, es: I) -> D
where
    D: AbstractDomain,
    I: IntoIterator<Item = &'a D>,
{
    es.into_iter().fold(D::top(dims), |acc, x| acc.meet(&x))
}
