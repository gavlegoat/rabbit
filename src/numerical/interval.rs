//! Utilities for working with intervals.
//!
//! The interval abstract domain keeps track of separate bounds on each program variable. The
//! bounds for a particular variable are held in the [`Itv`] type, while the [`Interval`] type
//! holds bounds for all of the program variables together. Most of the time it will be more
//! convenient to work with the higher-level interface rather than generating intervals directly.
//! Specifically, generating a specific abstract element for numerical domains should usually be
//! done either by starting with top or bottom, or using the [`from_lincons`] function in
//! [`rabbit::numerical`]. This allows you to easily switch to a different domain if you decide to.
//! For example:
//! ```
//! use rabbit::numerical::*;
//!
//! // If we start with a constraint x0 + x1 <= 3
//! let lc1 = LinearConstraint::from_coeffs(vec![1., 1.], 3.);
//! let t: Interval = from_lincons(2, vec![lc1].iter());
//! // Now perform your analysis using functions that can accept any numerical domain.
//! ```
//!
//! [`Itv`]: ./struct.Itv.html
//! [`Interval`]: ./struct.Interval.html
//! [`from_lincons`]: ../fn.from_lincons.html
//! [`rabbit::numerical`]: ../index.html

use crate::numerical::{AffineTransform, LinearConstraint, NumericalDomain};
use crate::AbstractDomain;
use std::collections::BinaryHeap;
use std::collections::HashMap;

/// A lower bound may be either negative infinity or a real number.
#[derive(PartialEq, PartialOrd, Copy, Clone, Debug)]
pub enum LowerBound {
    /// Represents negative infinity, i.e., there is no lower bound here.
    NegInf,
    /// Represents a real-valued lower bound.
    Value(f64),
}

impl LowerBound {
    /// Add another lower bound to this one.
    fn add(&self, b: &LowerBound) -> LowerBound {
        match self {
            LowerBound::NegInf => LowerBound::NegInf,
            LowerBound::Value(x) => match b {
                LowerBound::NegInf => LowerBound::NegInf,
                LowerBound::Value(y) => LowerBound::Value(x + y),
            },
        }
    }

    /// Multiply this bound by a positive constant. This function doesn't handle negative
    /// constants well because generally if you want to multiply a lower bound by a negative
    /// constant you should get an upper bound as a result.
    fn mult(&self, c: f64) -> LowerBound {
        match self {
            LowerBound::NegInf => LowerBound::NegInf,
            LowerBound::Value(x) => LowerBound::Value(x * c),
        }
    }
}

/// An upper bound may be either a real number or positive infinity.
#[derive(PartialEq, PartialOrd, Copy, Clone, Debug)]
pub enum UpperBound {
    /// Represents a real-valued upper bound.
    Value(f64),
    /// Represents positive infinity, i.e., there is no upper bound.
    PosInf,
}

impl UpperBound {
    /// Add another upper bound to this one.
    fn add(&self, b: &UpperBound) -> UpperBound {
        match self {
            UpperBound::PosInf => UpperBound::PosInf,
            UpperBound::Value(x) => match b {
                UpperBound::PosInf => UpperBound::PosInf,
                UpperBound::Value(y) => UpperBound::Value(x + y),
            },
        }
    }
}

/// A one dimensional interval over real numbers.
#[derive(Copy, Clone, Debug)]
pub struct Itv {
    /// The lower bound of this interval.
    lower: LowerBound,
    /// The upper bound of this interval.
    upper: UpperBound,
    /// Determines whether the lower bound is included in the interval.
    lower_incl: bool,
    /// Determines whether the lower bound is included in the interval.
    upper_incl: bool,
}

impl Itv {
    /// Construct an interval including all reals.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::unbounded(),
    ///     Itv::from_bounds_closed(LowerBound::NegInf, UpperBound::PosInf));
    /// ```
    pub fn unbounded() -> Itv {
        Itv {
            lower: LowerBound::NegInf,
            upper: UpperBound::PosInf,
            lower_incl: false,
            upper_incl: false,
        }
    }

    /// Construct an empty interval.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::empty(),
    ///     Itv::from_bounds_closed(LowerBound::Value(3.), UpperBound::Value(1.)));
    /// ```
    pub fn empty() -> Itv {
        Itv {
            lower: LowerBound::Value(1.),
            upper: UpperBound::Value(0.),
            lower_incl: false,
            upper_incl: false,
        }
    }

    /// Construct an interval with a single value, [a, a].
    ///
    /// # Arguments
    /// * `a` - The only value in the interval.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::precise(2.),
    ///     Itv::from_bounds_closed(LowerBound::Value(2.), UpperBound::Value(2.)));
    /// ```
    pub fn precise(a: f64) -> Itv {
        Itv {
            lower: LowerBound::Value(a),
            upper: UpperBound::Value(a),
            lower_incl: true,
            upper_incl: true,
        }
    }

    /// Construct a closed interval given two real-valued bounds.
    ///
    /// # Arguments
    /// * `l` - The lower bound.
    /// * `u` - The upper bound.
    ///
    /// Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::from_double_closed(0., 2.),
    ///     Itv::from_bounds_closed(LowerBound::Value(0.), UpperBound::Value(2.)));
    /// ```
    pub fn from_double_closed(l: f64, u: f64) -> Itv {
        Itv {
            lower: LowerBound::Value(l),
            upper: UpperBound::Value(u),
            lower_incl: true,
            upper_incl: true,
        }
    }

    /// Construct an open interval given two real-valued bounds.
    ///
    /// # Arguments
    /// * `l` - The lower bound.
    /// * `u` - The upper bound.
    ///
    /// Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::from_double_open(0., 2.),
    ///     Itv::from_bounds_open(LowerBound::Value(0.), UpperBound::Value(2.)));
    /// ```
    pub fn from_double_open(l: f64, u: f64) -> Itv {
        Itv {
            lower: LowerBound::Value(l),
            upper: UpperBound::Value(u),
            lower_incl: false,
            upper_incl: false,
        }
    }

    /// Construct an interval from given bounds where non-infinite bounds are included.
    ///
    /// # Arguments
    /// * `l` - The lower bound.
    /// * `u` - The upper bound.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::unbounded(),
    ///     Itv::from_bounds_closed(LowerBound::NegInf, UpperBound::PosInf));
    /// ```
    pub fn from_bounds_closed(l: LowerBound, u: UpperBound) -> Itv {
        Itv {
            lower: l,
            upper: u,
            lower_incl: match l {
                LowerBound::NegInf => false,
                _ => true,
            },
            upper_incl: match u {
                UpperBound::PosInf => false,
                _ => true,
            },
        }
    }

    /// Construct an interval from given bounds where the bounds are excluded.
    ///
    /// # Arguments
    /// * `l` - The lower bound.
    /// * `u` - The upper bound.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::unbounded(),
    ///     Itv::from_bounds_closed(LowerBound::NegInf, UpperBound::PosInf));
    /// ```
    pub fn from_bounds_open(l: LowerBound, u: UpperBound) -> Itv {
        Itv {
            lower: l,
            upper: u,
            lower_incl: false,
            upper_incl: false,
        }
    }

    /// Construct an interval where bounds may or may not be included.
    ///
    /// # Arguments
    /// * `l` - The lower bound.
    /// * `incl_l` - Whether to include the lower bound in the interval.
    /// * `u` - The upper bound.
    /// * `incl_u` - Whether to include the upper bound in the interval.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::from_double_closed(0., 1.),
    ///     Itv::from_bounds(LowerBound::Value(0.), true, UpperBound::Value(1.), true));
    /// assert_eq!(Itv::from_double_open(0., 1.),
    ///     Itv::from_bounds(LowerBound::Value(0.), false, UpperBound::Value(1.), false));
    /// ```
    pub fn from_bounds(l: LowerBound, incl_l: bool, u: UpperBound, incl_u: bool) -> Itv {
        Itv {
            lower: l,
            upper: u,
            lower_incl: incl_l && l != LowerBound::NegInf,
            upper_incl: incl_u && u != UpperBound::PosInf,
        }
    }

    /// Add two intervals. If either interval is empty, return empty.
    ///
    /// # Arguments
    /// * `b` - Another interval to add to this one.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// let a = Itv::from_double_closed(0., 2.);
    /// let b = Itv::from_double_closed(1., 4.);
    /// assert_eq!(a.add(&b), Itv::from_double_closed(1., 6.));
    /// ```
    pub fn add(&self, b: &Itv) -> Itv {
        if self.is_empty() {
            return self.clone();
        } else if b.is_empty() {
            return b.clone();
        }
        Itv {
            lower: self.lower.add(&b.lower),
            upper: self.upper.add(&b.upper),
            lower_incl: self.lower_incl && b.lower_incl,
            upper_incl: self.upper_incl && b.upper_incl,
        }
    }

    /// Multiply this interval by a constant.
    ///
    /// # Arguments
    /// * `b` - The constant to multiply by.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// let a = Itv::from_double_closed(-1., 3.);
    /// assert_eq!(a.mult(2.), Itv::from_double_closed(-2., 6.));
    /// assert_eq!(a.mult(-2.), Itv::from_double_closed(-6., 2.));
    /// assert_eq!(a.mult(0.), Itv::precise(0.));
    /// let b = Itv::from_bounds_closed(LowerBound::NegInf, UpperBound::Value(2.));
    /// assert_eq!(b.mult(-2.),
    ///     Itv::from_bounds_closed(LowerBound::Value(-4.), UpperBound::PosInf));
    /// ```
    pub fn mult(&self, b: f64) -> Itv {
        if self.is_empty() {
            return self.clone();
        }
        if b == 0.0 {
            Itv::precise(0.0)
        } else if b < 0.0 {
            Itv {
                lower: match self.upper {
                    UpperBound::Value(x) => LowerBound::Value(b * x),
                    UpperBound::PosInf => LowerBound::NegInf,
                },
                upper: match self.lower {
                    LowerBound::NegInf => UpperBound::PosInf,
                    LowerBound::Value(x) => UpperBound::Value(b * x),
                },
                lower_incl: self.upper_incl,
                upper_incl: self.lower_incl,
            }
        } else {
            Itv {
                lower: match self.lower {
                    LowerBound::NegInf => LowerBound::NegInf,
                    LowerBound::Value(x) => LowerBound::Value(x * b),
                },
                upper: match self.upper {
                    UpperBound::Value(x) => UpperBound::Value(x * b),
                    UpperBound::PosInf => UpperBound::PosInf,
                },
                lower_incl: self.lower_incl,
                upper_incl: self.upper_incl,
            }
        }
    }

    /// Expand this interval to include all the points in another interval.
    ///
    /// # Arguments
    /// * `b` - Another interval to join with.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::from_double_closed(0., 2.).join(&Itv::from_double_closed(1., 3.)),
    ///     Itv::from_double_closed(0., 3.));
    /// assert_eq!(Itv::from_double_closed(0., 1.).join(&Itv::from_double_closed(2., 3.)),
    ///     Itv::from_double_closed(0., 3.));
    /// ```
    pub fn join(&self, b: &Itv) -> Itv {
        if self.is_empty() {
            return b.clone();
        } else if b.is_empty() {
            return self.clone();
        }
        Itv {
            lower: if self.lower < b.lower {
                self.lower
            } else {
                b.lower
            },
            upper: if self.upper > b.upper {
                self.upper
            } else {
                b.upper
            },
            lower_incl: if self.lower < b.lower {
                self.lower_incl
            } else if self.lower == b.lower {
                self.lower_incl || b.lower_incl
            } else {
                b.lower_incl
            },
            upper_incl: if self.upper > b.upper {
                self.upper_incl
            } else if self.upper == b.upper {
                self.upper_incl || b.upper_incl
            } else {
                b.upper_incl
            }
        }
    }

    /// Constrict this interval so that it only includes points also present in another interval.
    ///
    /// # Arguments
    /// * `b` - Another interval to meet with.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::from_double_closed(0., 2.).meet(&Itv::from_double_closed(1., 3.)),
    ///     Itv::from_double_closed(1., 2.));
    /// assert_eq!(Itv::unbounded().meet(&Itv::from_double_closed(0., 2.)),
    ///     Itv::from_double_closed(0., 2.));
    /// ```
    pub fn meet(&self, b: &Itv) -> Itv {
        if self.is_empty() {
            return self.clone();
        } else if b.is_empty() {
            return b.clone();
        }
        Itv {
            lower: if self.lower > b.lower {
                self.lower
            } else {
                b.lower
            },
            upper: if self.upper < b.upper {
                self.upper
            } else {
                b.upper
            },
            lower_incl: if self.lower > b.lower {
                self.lower_incl
            } else if self.lower == b.lower {
                self.lower_incl && b.lower_incl
            } else {
                b.lower_incl
            },
            upper_incl: if self.upper < b.upper {
                self.upper_incl
            } else if self.upper == b.upper {
                self.upper_incl && b.upper_incl
            } else {
                b.upper_incl
            },
        }
    }

    /// Determine whether this interval includes any points.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert!(Itv::from_double_closed(1., 0.).is_empty());
    /// assert!(!Itv::unbounded().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        match self.lower {
            LowerBound::NegInf => false,
            LowerBound::Value(x) => match self.upper {
                UpperBound::PosInf => false,
                UpperBound::Value(y) => y < x || (y == x && !self.lower_incl && !self.upper_incl),
            },
        }
    }

    /// Determine whether this interval includes all reals.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert!(Itv::unbounded().is_unbounded());
    /// assert!(!Itv::empty().is_unbounded());
    /// ```
    pub fn is_unbounded(&self) -> bool {
        match self.lower {
            LowerBound::NegInf => match self.upper {
                UpperBound::PosInf => true,
                _ => false,
            }
            _ => false,
        }
    }

    /// Remove the lower bound from this interval.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::from_double_closed(0., 2.).remove_lower_bound(),
    ///     Itv::from_bounds_closed(LowerBound::NegInf, UpperBound::Value(2.)));
    /// ```
    pub fn remove_lower_bound(&self) -> Itv {
        Itv {
            lower: LowerBound::NegInf,
            upper: self.upper,
            lower_incl: false,
            upper_incl: self.upper_incl,
        }
    }

    /// Remove the upper bound from this interval.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// assert_eq!(Itv::from_double_closed(0., 2.).remove_upper_bound(),
    ///     Itv::from_bounds_closed(LowerBound::Value(0.), UpperBound::PosInf));
    /// ```
    pub fn remove_upper_bound(&self) -> Itv {
        Itv {
            lower: self.lower,
            upper: UpperBound::PosInf,
            lower_incl: self.lower_incl,
            upper_incl: false,
        }
    }
}

impl PartialEq for Itv {
    fn eq(&self, other: &Itv) -> bool {
        // We can ignore lower_incl and upper_incl when the bounds are infinite.
        (self.is_empty() && other.is_empty())
            || (self.lower == other.lower && self.upper == other.upper &&
                (self.lower_incl == other.lower_incl && self.upper_incl == other.upper_incl) ||
                (self.lower == LowerBound::NegInf && self.upper_incl == other.upper_incl) ||
                (self.upper == UpperBound::PosInf && self.lower_incl == other.lower_incl) ||
                (self.lower == LowerBound::NegInf && self.upper == UpperBound::PosInf))
    }
}

/// A hyperinterval.
#[derive(Clone, Debug)]
pub struct Interval {
    bounds: Vec<Itv>,
}

impl Interval {
    /// Construct an interval from a vector of constraints on each dimension.
    ///
    /// # Arguments
    /// * `v` - The bounds on each dimension.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::AbstractDomain;
    /// # use rabbit::numerical::interval::*;
    /// let i = Interval::from_vec(vec![Itv::unbounded(); 3]);
    /// assert!(i.is_top());
    /// ```
    pub fn from_vec(v: Vec<Itv>) -> Interval {
        Interval { bounds: v }
    }

    /// Construct an interval from a set of lower and upper bounds.
    ///
    /// # Arguments
    /// * `ls` - The lower bound of each interval.
    /// * `us` - The upper bound of each interval.
    /// * `closed` - Determines whether the bounds are included in the interval.
    ///
    /// # Panics
    /// Panics if the two bounds vectors do not have the same length.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// let i = Interval::from_doubles(vec![0., -1.], vec![1., 2.], true);
    /// ```
    pub fn from_doubles<I>(ls: I, us: I, closed: bool) -> Interval
    where
        I: IntoIterator<Item = f64>,
    {
        let mut li = ls.into_iter();
        let mut ui = us.into_iter();
        let mut bs = Vec::new();
        loop {
            let l = li.next();
            let u = ui.next();
            if l.is_none() && u.is_none() {
                break;
            }
            if l.is_none() || u.is_none() {
                panic!("Interval::from_doubles given bounds of different lengths");
            }
            if closed {
                bs.push(Itv::from_double_closed(l.unwrap(), u.unwrap()));
            } else {
                bs.push(Itv::from_double_open(l.unwrap(), u.unwrap()));
            }
        }
        Interval { bounds: bs }
    }

    /// Modify the bounds of a particular dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to modify.
    /// * `b` - The new bounds.
    ///
    /// # Panics
    /// Panics if `dim` is greater than the number of dimensions in `self`.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// let mut i = Interval::from_doubles(vec![0., -1.], vec![1., 2.], true);
    /// i.update_dim(0, Itv::from_double_closed(1., 2.));
    /// assert_eq!(i, Interval::from_doubles(vec![1., -1.], vec![2., 2.], true));
    /// ```
    pub fn update_dim(&mut self, dim: usize, b: Itv) {
        if dim >= self.bounds.len() {
            panic!("Attempting to update do high a dimension in Interval::update_dim");
        }
        self.bounds[dim] = b;
    }

    /// Remove the upper bound from some dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to modify.
    ///
    /// # Panics
    /// Panics if `dim` is greater than the number of dimensions in `self`.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// let mut i = Interval::from_doubles(vec![0., -1.], vec![1., 2.], true);
    /// i.remove_upper_bound(1);
    /// assert_eq!(i, Interval::from_vec(
    ///     vec![Itv::from_double_closed(0., 1.),
    ///          Itv::from_bounds_closed(LowerBound::Value(-1.), UpperBound::PosInf)]));
    /// ```
    pub fn remove_upper_bound(&mut self, dim: usize) {
        if dim >= self.bounds.len() {
            panic!("Attempting to update too high a dimension in Interval::remove_upper_bound");
        }
        self.bounds[dim] = self.bounds[dim].remove_upper_bound();
    }

    /// Remove the lower bound from some dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to modify.
    ///
    /// # Panics
    /// Panics if `dim` is greater than the number of dimensions in `self`.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::numerical::interval::*;
    /// let mut i = Interval::from_doubles(vec![0., -1.], vec![1., 2.], true);
    /// i.remove_lower_bound(1);
    /// assert_eq!(i, Interval::from_vec(
    ///     vec![Itv::from_double_closed(0., 1.),
    ///          Itv::from_bounds_closed(LowerBound::NegInf, UpperBound::Value(2.))]));
    /// ```
    pub fn remove_lower_bound(&mut self, dim: usize) {
        if dim >= self.bounds.len() {
            panic!("Attempting to update too high a dimension in Interval::remove_lower_bound");
        }
        self.bounds[dim] = self.bounds[dim].remove_lower_bound();
    }
}

impl PartialEq for Interval {
    fn eq(&self, other: &Interval) -> bool {
        (self.is_bottom() && other.is_bottom()) || self.bounds == other.bounds
    }
}

impl AbstractDomain for Interval {
    fn top(dims: usize) -> Interval {
        Interval::from_vec(vec![Itv::unbounded(); dims])
    }

    fn bottom(dims: usize) -> Interval {
        Interval::from_vec(vec![Itv::empty(); dims])
    }

    fn join(&self, other: &Interval) -> Interval {
        if self.bounds.len() != other.bounds.len() {
            panic!("Mismatched dimensionality in Interval::join");
        }
        Interval::from_vec(
            self.bounds
                .iter()
                .zip(other.bounds.iter())
                .map(|(a, b)| a.join(b))
                .collect(),
        )
    }

    fn meet(&self, other: &Interval) -> Interval {
        if self.bounds.len() != other.bounds.len() {
            panic!("Mismatched dimensionality in Interval::meet");
        }
        Interval::from_vec(
            self.bounds
                .iter()
                .zip(other.bounds.iter())
                .map(|(a, b)| a.meet(b))
                .collect(),
        )
    }

    fn is_top(&self) -> bool {
        for v in &self.bounds {
            if !v.is_unbounded() {
                return false;
            }
        }
        true
    }

    fn is_bottom(&self) -> bool {
        for v in &self.bounds {
            if v.is_empty() {
                return true;
            }
        }
        false
    }

    fn add_dims<I>(&self, dims: I) -> Interval
        where I: IntoIterator<Item = usize>
    {
        let mut ret = self.clone();
        // Convert ds from indices into the current vector to the indices that need to be added.
        let mut ds: Vec<usize> = dims.into_iter().collect();
        ds.sort();
        for i in 0..ds.len() {
            ds[i] += i;
        }
        for d in ds {
            let ind = if d > ret.bounds.len() { ret.bounds.len() } else { d };
            ret.bounds.insert(ind, Itv::unbounded());
        }
        ret
    }

    fn remove_dims<I>(&self, dims: I) -> Interval
    where
        I: IntoIterator<Item = usize>,
    {
        let mut ret = self.clone();
        let mut set = BinaryHeap::new();
        for n in dims {
            set.push(n);
        }
        let mut ds = set.into_sorted_vec();
        ds.reverse();
        for n in ds {
            if n >= ret.bounds.len() {
                panic!("Attempting to remove to high a dimension in Interval::remove_dims");
            }
            ret.bounds.remove(n);
        }
        ret
    }

    fn dims(&self) -> usize {
        self.bounds.len()
    }
}

/// Get bounds on the value of an affine expression over a given interval.
fn eval_affine_itv(itv: &Interval, trans: &AffineTransform) -> Itv {
    if trans.dims() != itv.dims() {
        panic!("Transformation has the wrong dimension in Interval::assign")
    }
    let mut ret = Itv::precise(0.0);
    for (i, c) in trans.coeffs.iter().enumerate() {
        ret = ret.add(&itv.bounds[i].mult(*c));
    }
    ret = ret.add(&Itv::precise(trans.cst));
    ret
}

impl NumericalDomain for Interval {
    fn assign(&self, trans: &HashMap<usize, AffineTransform>) -> Interval {
        let mut ret = self.clone();
        for (k, v) in trans {
            if *k >= self.bounds.len() {
                panic!("Dimension too high in Interval::assign");
            }
            ret.update_dim(*k, eval_affine_itv(self, v));
        }
        ret
    }

    fn constrain<'a, I>(&self, cnts: I) -> Interval
    where
        I: Iterator<Item = &'a LinearConstraint> + Clone,
    {
        let mut ret = self.clone();
        for lc in cnts.clone() {
            if lc.dims() != self.bounds.len() {
                panic!("Linear constraint has wrong dimenion in Interval::constrain");
            }
        }
        for _ in 0..10 {
            for dim in 0..self.bounds.len() {
                let mut update = Itv::unbounded();
                for lc in cnts.clone() {
                    let mut bounds = Itv::precise(0.0);
                    for (i, a) in lc.coeffs.iter().enumerate() {
                        if i != dim {
                            bounds = bounds.add(&ret.bounds[i].mult(*a));
                        }
                    }
                    bounds = bounds.add(&Itv::precise(-lc.cst));
                    let dim_bnd = if lc.coeffs[dim] == 0.0 {
                        if bounds.lower <= LowerBound::Value(0.0) {
                            Itv::unbounded()
                        } else {
                            Itv::empty()
                        }
                    } else if lc.coeffs[dim] < 0.0 {
                        Itv::from_bounds_closed(
                            bounds.lower.mult(1.0 / (-lc.coeffs[dim])),
                            UpperBound::PosInf)
                    } else {
                        match bounds.lower {
                            LowerBound::NegInf => Itv::empty(),
                            LowerBound::Value(c) => Itv::from_bounds_closed(
                                LowerBound::NegInf,
                                UpperBound::Value(c / (-lc.coeffs[dim]))),
                        }
                    };
                    update = update.meet(&dim_bnd);
                }
                ret.update_dim(dim, ret.bounds[dim].meet(&update));
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::numerical::NumericalDomain;
    use crate::AbstractDomain;

    fn itv1() -> Interval {
        Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::NegInf,
                UpperBound::Value(2.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::PosInf),
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(4.))])
    }

    fn itv3() -> Interval {
        Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::Value(3.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::Value(4.)),
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(4.))])
    }

    #[test]
    fn top_is_top() {
        let a: Interval = AbstractDomain::top(3);
        assert!(a.is_top());
    }

    #[test]
    fn unconstrained_is_top() {
        let a = Interval::from_vec(vec![Itv::unbounded(); 3]);
        assert!(a.is_top());
    }

    #[test]
    fn top_is_not_bottom() {
        let a: Interval = AbstractDomain::top(3);
        assert!(!a.is_bottom());
    }

    #[test]
    fn top_join_any_is_top() {
        let a: Interval = AbstractDomain::top(3);
        let b: Interval = AbstractDomain::bottom(3);
        assert!(a.join(&b).is_top());
        assert!(a.join(&itv1()).is_top());
    }

    #[test]
    fn top_meet_other() {
        let a: Interval = AbstractDomain::top(3);
        let b: Interval = AbstractDomain::bottom(3);
        assert_eq!(a.meet(&b), b);
        assert_eq!(a.meet(&itv1()), itv1());
    }

    #[test]
    fn bottom_is_bottom() {
        let a: Interval = AbstractDomain::bottom(3);
        assert!(a.is_bottom());
    }

    #[test]
    fn bottom_is_not_top() {
        let a: Interval = AbstractDomain::bottom(3);
        assert!(!a.is_top());
    }

    #[test]
    fn bottom_join_other() {
        let a: Interval = AbstractDomain::top(3);
        let b: Interval = AbstractDomain::bottom(3);
        assert_eq!(b.join(&a), a);
        assert_eq!(b.join(&itv1()), itv1());
    }

    #[test]
    fn bottom_meet_other() {
        let a: Interval = AbstractDomain::top(3);
        let b: Interval = AbstractDomain::bottom(3);
        assert_eq!(b.meet(&a), b);
        assert_eq!(b.meet(&itv1()), b);
    }

    #[test]
    fn eq_is_reflexive() {
        let a: Interval = AbstractDomain::top(3);
        let b: Interval = AbstractDomain::bottom(3);
        assert_eq!(a, a);
        assert_eq!(b, b);
        assert_eq!(itv1(), itv1());
    }

    #[test]
    fn is_top_top_only() {
        let b: Interval = AbstractDomain::bottom(3);
        assert!(!b.is_top());
        assert!(!itv1().is_top());
    }

    #[test]
    fn is_bottom_bottom_only() {
        let a: Interval = AbstractDomain::top(3);
        assert!(!a.is_bottom());
        assert!(!itv1().is_bottom());
    }

    #[test]
    fn is_bottom_unsat_constraints() {
        let unsat = Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::NegInf,
                UpperBound::Value(2.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::Value(0.)),
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(4.))]);
        assert!(unsat.is_bottom());
    }

    #[test]
    fn join_normal() {
        let c = Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::NegInf,
                UpperBound::Value(3.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::PosInf),
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(4.))]);
        assert_eq!(itv1().join(&itv3()), c);
    }

    #[test]
    fn meet_normal() {
        let c = Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::Value(2.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::Value(4.)),
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(4.))]);
        assert_eq!(itv1().meet(&itv3()), c);
    }

    #[test]
    fn remove_vars_normal() {
        let b = Interval::from_vec(vec![Itv::from_double_closed(0., 4.)]);
        assert_eq!(itv1().remove_dims(vec![0, 1]).dims(), 1);
        assert_eq!(itv1().remove_dims(vec![0, 1]), b);
    }

    #[test]
    fn assign_zero_gives_zero() {
        let mut zero_trans = HashMap::new();
        zero_trans.insert(0, AffineTransform::zero(3));
        let c = Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(0.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::PosInf),
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(4.))]);
        assert_eq!(itv1().assign(&zero_trans), c);
    }

    #[test]
    fn assign_unbounded() {
        let mut trans = HashMap::new();
        trans.insert(0, AffineTransform::from_coeffs(vec![1., 1., 1.], 0.));
        let c = Interval::from_vec(vec![
            Itv::unbounded(),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::PosInf),
            Itv::from_double_closed(0., 4.)]);
        assert_eq!(itv1().assign(&trans), c);
        trans.insert(1, AffineTransform::from_coeffs(vec![1., 1., 1.], 0.));
        trans.insert(2, AffineTransform::from_coeffs(vec![1., 1., 1.], 0.));
        let top = Interval::top(3);
        assert!(itv1().assign(&trans).is_top());
        assert_eq!(itv1().assign(&trans), top);
    }

    #[test]
    fn assign_normal() {
        let mut trans = HashMap::new();
        trans.insert(0, AffineTransform::from_coeffs(vec![1., 0., 1.], 2.));
        let c = Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::NegInf,
                UpperBound::Value(8.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::PosInf),
            Itv::from_double_closed(0., 4.)]);
        assert_eq!(itv1().assign(&trans), c);
        let mut n_trans = HashMap::new();
        n_trans.insert(0, AffineTransform::from_coeffs(vec![0., -1., -1.], -1.));
        let d = Interval::from_vec(vec![
            Itv::from_bounds_closed(
                LowerBound::NegInf,
                UpperBound::Value(-2.)),
            Itv::from_bounds_closed(
                LowerBound::Value(1.),
                UpperBound::PosInf),
            Itv::from_bounds_closed(
                LowerBound::Value(0.),
                UpperBound::Value(4.))]);
        assert_eq!(itv1().assign(&n_trans), d);
    }

    #[test]
    fn constrain_unsat() {
        let cnstr = LinearConstraint::from_coeffs(vec![0.; 3], -1.);
        assert!(itv1().constrain(vec![cnstr].iter()).is_bottom());
    }

    #[test]
    fn constrain_no_constraints() {
        assert_eq!(itv1().constrain(Vec::new().iter()), itv1());
    }

    #[test]
    fn constrain_normal() {
        let lc1 = LinearConstraint::from_coeffs(vec![1., 0., 0.], 1.);
        let lc2 = vec![
            LinearConstraint::from_coeffs(vec![0., 0., -1.], -1.),
            LinearConstraint::from_coeffs(vec![1., 1., 0.], 1.),
        ];
        let itv4 = Interval::from_vec(vec![Itv::from_double_closed(0., 2.); 3]);
        let res1 = Interval::from_vec(vec![
            Itv::from_double_closed(1., 1.),
            Itv::from_double_closed(1., 4.),
            Itv::from_double_closed(0., 4.)]);
        let res2 = Interval::from_vec(vec![
            Itv::from_double_closed(0., 1.),
            Itv::from_double_closed(0., 1.),
            Itv::from_double_closed(1., 2.)]);
        assert_eq!(itv3().constrain(vec![lc1].iter()), res1);
        assert_eq!(itv4.constrain(lc2.iter()), res2);
    }

    #[test]
    fn join_bounds() {
        let a = Interval::from_doubles(vec![0.], vec![1.], true);
        let b = Interval::from_doubles(vec![0.], vec![2.], false);
        assert_eq!(a.join(&b),
            Interval::from_vec(
                vec![Itv::from_bounds(LowerBound::Value(0.), true,
                                      UpperBound::Value(2.), false)]));
    }

    #[test]
    fn meet_bounds() {
        let a = Interval::from_doubles(vec![0.], vec![1.], true);
        let b = Interval::from_doubles(vec![0.], vec![2.], false);
        assert_eq!(a.meet(&b),
            Interval::from_vec(
                vec![Itv::from_bounds(LowerBound::Value(0.), false,
                                      UpperBound::Value(1.), true)]));
    }
}
