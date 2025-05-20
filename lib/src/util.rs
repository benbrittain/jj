//! Utilities

use core::hash::Hash;

use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

/// Create a **hashbrown::HashMap** from a list of key-value pairs.
#[macro_export(local_inner_macros)]
macro_rules! hashmap {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(hashmap!(@single $rest)),*]));

    ($($key:expr => $value:expr,)+) => { hashmap!($($key => $value),+) };
    ($($key:expr => $value:expr),*) => {
        {
            let _cap = hashmap!(@count $($key),*);
            let mut _map = ::hashbrown::HashMap::with_capacity(_cap);
            $(
                let _ = _map.insert($key, $value);
            )*
            _map
        }
    };
}
pub use hashmap;

/// Create a **hashbrown::HashSet** from a list of elements.
#[macro_export(local_inner_macros)]
macro_rules! hashset {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(hashset!(@single $rest)),*]));

    ($($key:expr,)+) => { hashset!($($key),+) };
    ($($key:expr),*) => {
        {
            let _cap = hashset!(@count $($key),*);
            let mut _set = ::hashbrown::HashSet::with_capacity(_cap);
            $(
                let _ = _set.insert($key);
            )*
            _set
        }
    };
}
pub use hashset;

/// Create a **std::collections::BTreeMap** from a list of key-value pairs.
#[macro_export(local_inner_macros)]
macro_rules! btreemap {
    // trailing comma case
    ($($key:expr => $value:expr,)+) => (btreemap!($($key => $value),+));

    ( $($key:expr => $value:expr),* ) => {
        {
            let mut _map = ::std::collections::BTreeMap::new();
            $(
                let _ = _map.insert($key, $value);
            )*
            _map
        }
    };
}

pub use btreemap;

/// An iterator adapter to filter out duplicate elements.
///
/// See [`.unique_by()`](crate::Itertools::unique) for more information.
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct UniqueBy<I: Iterator, V, F> {
    iter: I,
    // Use a Hashmap for the Entry API in order to prevent hashing twice.
    // This can maybe be replaced with a HashSet once `get_or_insert_with`
    // or a proper Entry API for Hashset is stable and meets this msrv
    used: HashMap<V, ()>,
    f: F,
}

// count the number of new unique keys in iterable (`used` is the set already
// seen)
fn count_new_keys<I, K>(mut used: HashMap<K, ()>, iterable: I) -> usize
where
    I: IntoIterator<Item = K>,
    K: Hash + Eq,
{
    let iter = iterable.into_iter();
    let current_used = used.len();
    used.extend(iter.map(|key| (key, ())));
    used.len() - current_used
}

impl<I, V, F> Iterator for UniqueBy<I, V, F>
where
    I: Iterator,
    V: Eq + Hash,
    F: FnMut(&I::Item) -> V,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let Self { iter, used, f } = self;
        iter.find(|v| used.insert(f(v), ()).is_none())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, hi) = self.iter.size_hint();
        ((low > 0 && self.used.is_empty()) as usize, hi)
    }

    fn count(self) -> usize {
        let mut key_f = self.f;
        count_new_keys(self.used, self.iter.map(move |elt| key_f(&elt)))
    }
}

impl<I> Iterator for Unique<I>
where
    I: Iterator,
    I::Item: Eq + Hash + Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let UniqueBy { iter, used, .. } = &mut self.iter;
        iter.find_map(|v| {
            if let Entry::Vacant(entry) = used.entry(v) {
                let elt = entry.key().clone();
                entry.insert(());
                return Some(elt);
            }
            None
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, hi) = self.iter.iter.size_hint();
        ((low > 0 && self.iter.used.is_empty()) as usize, hi)
    }

    fn count(self) -> usize {
        count_new_keys(self.iter.used, self.iter.iter)
    }
}

#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub(crate) struct Unique<I>
where
    I: Iterator,
    I::Item: Eq + Hash + Clone,
{
    iter: UniqueBy<I, I::Item, ()>,
}

pub(crate) fn unique<I>(iter: I) -> Unique<I>
where
    I: Iterator,
    I::Item: Eq + Hash + Clone,
{
    Unique {
        iter: UniqueBy {
            iter,
            used: HashMap::new(),
            f: (),
        },
    }
}
