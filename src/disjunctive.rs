use crate::AbstractDomain;

/// A disjunction of elements in an underlying abstract domain.
///
/// An element of a disjunctive domain represents the union of the states represented by each of
/// it's elements. This provides added flexibility compared to using a single abstract element to
/// capture all possible program states. For example, joining the interval [0, 1] with [3, 4]
/// yields the interval [0, 4]. However the values in (1, 3) cannot actually be attained, so this
/// join is not particularly precise. In the disjunctive interval domain, these two pieces are kept
/// separately, allowing more precision.
///
/// While disjunctive domains allow much greater expressivity than non-disjunctive domains, it
/// comes at a cost in terms of computational efficiency. Specifically, the number of disjuncts in
/// the output of each join is the sum of the disjuncts in its inputs. This means that the number
/// of disjuncts can increase rapidly and lead to slow analysis. To ameliorate this problem, it is
/// generally advisable to call the [`reduce`] function periodically in order to reduce
/// the complexity.
///
/// [`reduce`]: ./struct.Disjunction.html#method.reduce

#[derive(Clone, Debug)]
pub struct Disjunction<D> {
    // disj should always contain at least one element.
    disj: Vec<D>,
}

impl<D: AbstractDomain + Clone + PartialEq> PartialEq for Disjunction<D> {
    fn eq(&self, other: &Disjunction<D>) -> bool {
        for v in &self.disj {
            let mut found = false;
            for w in &other.disj {
                if v == w {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        for v in &other.disj {
            let mut found = false;
            for w in &self.disj {
                if v == w {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        true
    }
}

impl<D: AbstractDomain> Disjunction<D> {

    /// Construct a disjunction from a set of disjuncts.
    ///
    /// # Arguments
    /// * `parts` - The pieces of the disjunction to build.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::*;
    /// # use rabbit::numerical::Interval;
    /// let d = Disjunction::from_parts(
    ///     vec![Interval::from_doubles(vec![0.], vec![1.], true),
    ///          Interval::from_doubles(vec![3.], vec![4.], true)]);
    /// assert_eq!(d.num_parts(), 2);
    /// ```
    pub fn from_parts<I>(parts: I) -> Disjunction<D>
        where I: IntoIterator<Item = D>
    {
        Disjunction {
            disj: parts.into_iter().collect()
        }
    }

    /// Reduce the number of disjunctions in an abstract element. The function `f` is applied to
    /// the vector of disjunctions and the resulting pair of elements are joined. This process
    /// continues until `f` returns `None`.
    ///
    /// # Arguments
    /// * `f` - A function to determine which elements to join.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::*;
    /// # use rabbit::numerical::Interval;
    /// let mut d = Disjunction::from_parts(
    ///     vec![Interval::from_doubles(vec![0.], vec![1.], true),
    ///          Interval::from_doubles(vec![3.], vec![4.], true)]);
    /// // Join elements until there is only one left.
    /// d.reduce(|v| if v.len() > 1 { Some((0, 1)) } else { None });
    /// assert_eq!(d.num_parts(), 1);
    /// assert_eq!(d.get_parts(), vec![Interval::from_doubles(vec![0.], vec![4.], true)]);
    /// ```
    pub fn reduce<F>(&mut self, f: F)
    where
        F: Fn(&Vec<D>) -> Option<(usize, usize)>,
    {
        loop {
            match f(&self.disj) {
                Some((i, j)) => {
                    let a = i.max(j);
                    let b = i.min(j);
                    let x = self.disj.remove(a);
                    let y = self.disj.remove(b);
                    self.disj.push(x.join(&y));
                }
                None => break,
            }
        }
    }

    /// Get the number of disjuncts in this abstract element.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::*;
    /// # use rabbit::numerical::Interval;
    /// let d = Disjunction::from_parts(
    ///     vec![Interval::from_doubles(vec![0.], vec![1.], true),
    ///          Interval::from_doubles(vec![3.], vec![4.], true)]);
    /// assert_eq!(d.num_parts(), 2);
    /// ```
    pub fn num_parts(&self) -> usize {
        self.disj.len()
    }

    /// Get the pieces out of this abstract element.
    ///
    /// # Examples
    /// ```
    /// # use rabbit::*;
    /// # use rabbit::numerical::Interval;
    /// let d = Disjunction::from_parts(
    ///     vec![Interval::from_doubles(vec![0.], vec![1.], true)]);
    /// assert_eq!(d.get_parts(), vec![Interval::from_doubles(vec![0.], vec![1.], true)]);
    /// ```
    pub fn get_parts(self) -> Vec<D> {
        self.disj
    }

}

impl<D: AbstractDomain + Clone> AbstractDomain for Disjunction<D> {
    fn top(dims: usize) -> Disjunction<D> {
        Disjunction {
            disj: vec![AbstractDomain::top(dims)],
        }
    }

    fn bottom(dims: usize) -> Disjunction<D> {
        Disjunction {
            disj: vec![AbstractDomain::bottom(dims)],
        }
    }

    fn join(&self, other: &Disjunction<D>) -> Disjunction<D> {
        let mut ret = self.disj.clone();
        ret.extend(other.disj.clone());
        Disjunction { disj: ret }
    }

    fn meet(&self, other: &Disjunction<D>) -> Disjunction<D> {
        let mut ret = Vec::new();
        for a in &self.disj {
            for b in &other.disj {
                let res = a.meet(&b);
                if !res.is_bottom() {
                    ret.push(res);
                }
            }
        }
        Disjunction {
            disj: if ret.is_empty() {
                vec![AbstractDomain::bottom(self.dims())]
            } else {
                ret
            },
        }
    }

    fn is_top(&self) -> bool {
        self.disj.iter().all(|x| x.is_top())
    }

    fn is_bottom(&self) -> bool {
        self.disj.iter().all(|x| x.is_bottom())
    }

    fn add_dims<I>(&self, dims: I) -> Disjunction<D>
    where
        I: IntoIterator<Item = usize>,
    {
        let i: Vec<usize> = dims.into_iter().collect();
        Disjunction {
            disj: self
                .disj
                .iter()
                .map(|x| x.add_dims(i.iter().cloned()))
                .collect(),
        }
    }

    fn remove_dims<I>(&self, dims: I) -> Disjunction<D>
    where
        I: IntoIterator<Item = usize>,
    {
        let i: Vec<usize> = dims.into_iter().collect();
        Disjunction {
            disj: self
                .disj
                .iter()
                .map(|x| x.remove_dims(i.iter().cloned()))
                .collect(),
        }
    }

    fn dims(&self) -> usize {
        if self.disj.is_empty() {
            panic!("Empty disjunction encountered");
        }
        self.disj[0].dims()
    }
}

