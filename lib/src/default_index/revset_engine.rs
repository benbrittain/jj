// Copyright 2023 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::convert::Infallible;
use std::fmt;
use std::iter;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use bstr::BString;
use futures::Stream;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use itertools::Itertools as _;

use super::composite::AsCompositeIndex;
use super::composite::CompositeIndex;
use super::entry::GlobalCommitPosition;
use super::rev_walk::EagerRevWalk;
use super::rev_walk::PeekableRevWalk;
use super::rev_walk::RevWalk;
use super::rev_walk::RevWalkBuilder;
use super::revset_graph_iterator::RevsetGraphWalk;
use crate::backend::BackendResult;
use crate::backend::ChangeId;
use crate::backend::CommitId;
use crate::backend::MillisSinceEpoch;
use crate::commit::Commit;
use crate::conflict_labels::ConflictLabels;
use crate::conflicts::MaterializedTreeValue;
use crate::conflicts::materialize_tree_value;
use crate::default_index::bit_set::AncestorsBitSet;
use crate::diff::ContentDiff;
use crate::diff::DiffHunkKind;
use crate::files;
use crate::graph::GraphNode;
use crate::matchers::FilesMatcher;
use crate::matchers::Matcher;
use crate::matchers::Visit;
use crate::merge::Merge;
use crate::object_id::HexPrefix;
use crate::object_id::ObjectId as _;
use crate::object_id::PrefixResolution;
use crate::repo_path::RepoPath;
use crate::revset::GENERATION_RANGE_FULL;
use crate::revset::ResolvedExpression;
use crate::revset::ResolvedPredicateExpression;
use crate::revset::Revset;
use crate::revset::RevsetContainingFn;
use crate::revset::RevsetEvaluationError;
use crate::revset::RevsetFilterPredicate;
use crate::rewrite;
use crate::store::Store;
use crate::str_util::StringMatcher;
use crate::tree_merge::MergeOptions;
use crate::tree_merge::resolve_file_values;
use crate::union_find;

pub(super) type BoxedRevWalk<'a> = Box<
    dyn RevWalk<CompositeIndex, Item = Result<GlobalCommitPosition, RevsetEvaluationError>> + 'a,
>;

// Use a local stream type without Send bound since we're single-threaded
type PositionsStream<'a> =
    std::pin::Pin<Box<dyn Stream<Item = Result<GlobalCommitPosition, RevsetEvaluationError>> + 'a>>;

/// Helper to drive a stream to completion synchronously.
/// This repeatedly polls until all items are collected.
/// Only safe for streams that don't actually perform async I/O.
fn collect_positions_sync(
    mut stream: PositionsStream<'_>,
) -> Result<Vec<GlobalCommitPosition>, RevsetEvaluationError> {
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut results = Vec::new();

    loop {
        match stream.as_mut().poll_next(&mut cx) {
            Poll::Ready(Some(Ok(pos))) => results.push(pos),
            Poll::Ready(Some(Err(e))) => return Err(e),
            Poll::Ready(None) => return Ok(results),
            Poll::Pending => {
                // Keep polling - our streams should always make progress
                continue;
            }
        }
    }
}

/// Helper to get the next item from a stream synchronously.
#[allow(dead_code)]
fn next_position_sync(
    stream: &mut PositionsStream<'_>,
) -> Option<Result<GlobalCommitPosition, RevsetEvaluationError>> {
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);

    loop {
        match stream.as_mut().poll_next(&mut cx) {
            Poll::Ready(item) => return item,
            Poll::Pending => {
                // Keep polling - our streams should always make progress
                continue;
            }
        }
    }
}

/// A predicate that can asynchronously test whether a commit position should be included.
#[async_trait(?Send)]
trait AsyncPredicate {
    /// Tests if the given entry is included in the set.
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError>;
}

trait ToAsyncPredicateFn: fmt::Debug {
    /// Creates an async predicate that tests if the given entry is included in the set.
    ///
    /// The predicate function is evaluated in order of the positions stream.
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_>;
}

impl<T: ToAsyncPredicateFn + ?Sized> ToAsyncPredicateFn for Box<T> {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        <T as ToAsyncPredicateFn>::to_async_predicate_fn(self)
    }
}

trait InternalRevset: fmt::Debug + ToAsyncPredicateFn {
    // All revsets currently iterate in order of descending index position
    fn positions_stream<'a>(&'a self, index: &'a CompositeIndex) -> PositionsStream<'a>;
}

impl<T: InternalRevset + ?Sized> InternalRevset for Box<T> {
    fn positions_stream<'a>(&'a self, index: &'a CompositeIndex) -> PositionsStream<'a> {
        <T as InternalRevset>::positions_stream(self, index)
    }
}

pub(super) struct RevsetImpl<I> {
    inner: Box<dyn InternalRevset>,
    index: I,
}

impl<I: AsCompositeIndex + Clone> RevsetImpl<I> {
    fn new(inner: Box<dyn InternalRevset>, index: I) -> Self {
        Self { inner, index }
    }

    fn positions_stream(&self) -> PositionsStream<'_> {
        self.inner.positions_stream(self.index.as_composite())
    }
}

impl<I> fmt::Debug for RevsetImpl<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RevsetImpl")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl<I: AsCompositeIndex + Clone> Revset for RevsetImpl<I> {
    async fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = Result<CommitId, RevsetEvaluationError>> + 'a> {
        let index = self.index.clone();
        let positions: Vec<_> = self.positions_stream().collect().await;
        Box::new(positions.into_iter().map(move |pos| {
            Ok(index.as_composite().commits().entry_by_pos(pos?).commit_id())
        }))
    }

    async fn commit_change_ids<'a>(&'a self) -> Box<dyn Iterator<Item = Result<(CommitId, ChangeId), RevsetEvaluationError>> + 'a> {
        let index = self.index.clone();
        let positions: Vec<_> = self.positions_stream().collect().await;
        Box::new(positions.into_iter().map(move |pos| {
            let entry = index.as_composite().commits().entry_by_pos(pos?);
            Ok((entry.commit_id(), entry.change_id()))
        }))
    }

    async fn iter_graph<'a>(&'a self) -> Box<dyn Iterator<Item = Result<GraphNode<CommitId>, RevsetEvaluationError>> + 'a> {
        self.iter_graph_impl(true).await
    }

    async fn is_empty(&self) -> bool {
        self.positions_stream().next().await.is_none()
    }

    async fn count_estimate(&self) -> Result<(usize, Option<usize>), RevsetEvaluationError> {
        if cfg!(feature = "testing") {
            // Exercise the estimation feature in tests. (If we ever have a Revset
            // implementation in production code that returns estimates, we can probably
            // remove this and rewrite the associated tests.)
            let positions: Vec<_> = self.positions_stream().take(10).collect().await;
            let count = positions.into_iter().process_results(|iter| iter.count())?;
            if count < 10 {
                Ok((count, Some(count)))
            } else {
                Ok((10, None))
            }
        } else {
            let positions: Vec<_> = self.positions_stream().collect().await;
            let count = positions.into_iter().process_results(|iter| iter.count())?;
            Ok((count, Some(count)))
        }
    }

    async fn containing_fn<'a>(&'a self) -> Box<RevsetContainingFn<'a>> {
        let positions: Vec<_> = self.positions_stream().collect().await;
        let positions =
            PositionsAccumulator::new(self.index.clone(), Box::new(EagerRevWalk::new(positions.into_iter())));
        Box::new(move |commit_id| positions.contains(commit_id))
    }

    fn stream<'a>(
        &self,
    ) -> Box<dyn futures::Stream<Item = Result<CommitId, RevsetEvaluationError>> + Unpin + 'a>
    where
        Self: 'a,
    {
        // We need to eagerly collect positions first since positions_stream() borrows self,
        // but we need the returned stream to outlive the borrow with lifetime 'a.
        // This collects using manual polling since we can't await in a non-async method.
        let index = self.index.clone();
        let positions = collect_positions_sync(self.positions_stream());
        Box::new(futures::stream::iter(positions.into_iter().flatten().map(
            move |pos| Ok(index.as_composite().commits().entry_by_pos(pos).commit_id()),
        )))
    }
}

impl<I: AsCompositeIndex + Clone> RevsetImpl<I> {
    pub(super) async fn iter_graph_impl<'a>(
        &'a self,
        skip_transitive_edges: bool,
    ) -> Box<dyn Iterator<Item = Result<GraphNode<CommitId>, RevsetEvaluationError>> + 'a> {
        let index = self.index.clone();
        let positions: Vec<_> = self.positions_stream().collect().await;
        let walk = EagerRevWalk::new(positions.into_iter());
        let mut graph_walk = RevsetGraphWalk::new(Box::new(walk), skip_transitive_edges);
        Box::new(iter::from_fn(move || {
            graph_walk.next(index.as_composite())
        }))
    }
}

/// Incrementally consumes `RevWalk` of the revset collecting positions.
struct PositionsAccumulator<'a, I> {
    index: I,
    inner: RefCell<PositionsAccumulatorInner<'a>>,
}

impl<'a, I: AsCompositeIndex> PositionsAccumulator<'a, I> {
    fn new(index: I, walk: BoxedRevWalk<'a>) -> Self {
        let inner = RefCell::new(PositionsAccumulatorInner {
            walk,
            consumed_positions: Vec::new(),
        });
        Self { index, inner }
    }

    /// Checks whether the commit is in the revset.
    fn contains(&self, commit_id: &CommitId) -> Result<bool, RevsetEvaluationError> {
        let index = self.index.as_composite();
        let Some(position) = index.commits().commit_id_to_pos(commit_id) else {
            return Ok(false);
        };

        let mut inner = self.inner.borrow_mut();
        inner.consume_to(index, position)?;
        let found = inner
            .consumed_positions
            .binary_search_by(|p| p.cmp(&position).reverse())
            .is_ok();
        Ok(found)
    }

    #[cfg(test)]
    fn consumed_len(&self) -> usize {
        self.inner.borrow().consumed_positions.len()
    }
}

/// Helper struct for [`PositionsAccumulator`] to simplify interior mutability.
struct PositionsAccumulatorInner<'a> {
    walk: BoxedRevWalk<'a>,
    consumed_positions: Vec<GlobalCommitPosition>,
}

impl PositionsAccumulatorInner<'_> {
    /// Consumes `RevWalk` to a desired position but not deeper.
    fn consume_to(
        &mut self,
        index: &CompositeIndex,
        desired_position: GlobalCommitPosition,
    ) -> Result<(), RevsetEvaluationError> {
        let last_position = self.consumed_positions.last();
        if last_position.is_some_and(|&pos| pos <= desired_position) {
            return Ok(());
        }
        while let Some(position) = self.walk.next(index).transpose()? {
            self.consumed_positions.push(position);
            if position <= desired_position {
                return Ok(());
            }
        }
        Ok(())
    }
}

/// Adapter for precomputed `GlobalCommitPosition`s.
#[derive(Debug)]
struct EagerRevset {
    positions: Vec<GlobalCommitPosition>,
}

impl EagerRevset {
    pub const fn empty() -> Self {
        Self {
            positions: Vec::new(),
        }
    }
}

impl InternalRevset for EagerRevset {
    fn positions_stream<'a>(&'a self, _index: &'a CompositeIndex) -> PositionsStream<'a> {
        Box::pin(futures::stream::iter(self.positions.iter().copied().map(Ok)))
    }
}

struct EagerRevsetPredicate {
    positions: Vec<GlobalCommitPosition>,
}

#[async_trait(?Send)]
impl AsyncPredicate for EagerRevsetPredicate {
    async fn matches(
        &mut self,
        _index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        Ok(self
            .positions
            .binary_search_by(|p| p.cmp(&pos).reverse())
            .is_ok())
    }
}

impl ToAsyncPredicateFn for EagerRevset {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(EagerRevsetPredicate {
            positions: self.positions.clone(),
        })
    }
}

/// Adapter for infallible `RevWalk` of `GlobalCommitPosition`s.
struct RevWalkRevset<W> {
    walk: W,
}

impl<W> fmt::Debug for RevWalkRevset<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RevWalkRevset").finish_non_exhaustive()
    }
}

impl<W> InternalRevset for RevWalkRevset<W>
where
    W: RevWalk<CompositeIndex, Item = GlobalCommitPosition> + Clone,
{
    fn positions_stream<'a>(&'a self, index: &'a CompositeIndex) -> PositionsStream<'a> {
        let walk = self.walk.clone();
        Box::pin(futures::stream::unfold(
            (walk, index),
            |(mut walk, index)| async move {
                walk.next(index).map(|pos| (Ok(pos), (walk, index)))
            },
        ))
    }
}

struct RevWalkPredicate<W: RevWalk<CompositeIndex, Item = GlobalCommitPosition>> {
    walk: PeekableRevWalk<CompositeIndex, W>,
}

#[async_trait(?Send)]
impl<W> AsyncPredicate for RevWalkPredicate<W>
where
    W: RevWalk<CompositeIndex, Item = GlobalCommitPosition>,
{
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        while self.walk.next_if(index, |&p| p > pos).is_some() {
            continue;
        }
        Ok(self.walk.next_if(index, |&p| p == pos).is_some())
    }
}

impl<W> ToAsyncPredicateFn for RevWalkRevset<W>
where
    W: RevWalk<CompositeIndex, Item = GlobalCommitPosition> + Clone,
{
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(RevWalkPredicate {
            walk: self.walk.clone().peekable(),
        })
    }
}

#[derive(Debug)]
struct FilterRevset<S, P> {
    candidates: S,
    predicate: P,
}

impl<S, P> InternalRevset for FilterRevset<S, P>
where
    S: InternalRevset,
    P: ToAsyncPredicateFn,
{
    fn positions_stream<'a>(&'a self, index: &'a CompositeIndex) -> PositionsStream<'a> {
        let candidates = &self.candidates;
        let predicate = &self.predicate;

        struct FilterState<'a> {
            stream: PositionsStream<'a>,
            pred: Box<dyn AsyncPredicate + 'a>,
            index: &'a CompositeIndex,
        }

        let state = FilterState {
            stream: candidates.positions_stream(index),
            pred: predicate.to_async_predicate_fn(),
            index,
        };

        Box::pin(futures::stream::unfold(state, |mut state| async move {
            loop {
                match state.stream.next().await {
                    Some(Ok(pos)) => {
                        match state.pred.matches(state.index, pos).await {
                            Ok(true) => return Some((Ok(pos), state)),
                            Ok(false) => continue,
                            Err(e) => return Some((Err(e), state)),
                        }
                    }
                    Some(Err(e)) => return Some((Err(e), state)),
                    None => return None,
                }
            }
        }))
    }
}

struct FilterPredicate<'a> {
    p1: Box<dyn AsyncPredicate + 'a>,
    p2: Box<dyn AsyncPredicate + 'a>,
}

#[async_trait(?Send)]
impl AsyncPredicate for FilterPredicate<'_> {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        Ok(self.p1.matches(index, pos).await? && self.p2.matches(index, pos).await?)
    }
}

impl<S, P> ToAsyncPredicateFn for FilterRevset<S, P>
where
    S: ToAsyncPredicateFn,
    P: ToAsyncPredicateFn,
{
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(FilterPredicate {
            p1: self.candidates.to_async_predicate_fn(),
            p2: self.predicate.to_async_predicate_fn(),
        })
    }
}

#[derive(Debug)]
struct NotInPredicate<S>(S);

struct NotInPredicateImpl<'a> {
    inner: Box<dyn AsyncPredicate + 'a>,
}

#[async_trait(?Send)]
impl AsyncPredicate for NotInPredicateImpl<'_> {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        Ok(!self.inner.matches(index, pos).await?)
    }
}

impl<S: ToAsyncPredicateFn> ToAsyncPredicateFn for NotInPredicate<S> {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(NotInPredicateImpl {
            inner: self.0.to_async_predicate_fn(),
        })
    }
}

#[derive(Debug)]
struct UnionRevset<S1, S2> {
    set1: S1,
    set2: S2,
}

impl<S1, S2> InternalRevset for UnionRevset<S1, S2>
where
    S1: InternalRevset,
    S2: InternalRevset,
{
    fn positions_stream<'a>(&'a self, index: &'a CompositeIndex) -> PositionsStream<'a> {
        union_stream(
            self.set1.positions_stream(index),
            self.set2.positions_stream(index),
        )
    }
}

struct UnionPredicate<'a> {
    p1: Box<dyn AsyncPredicate + 'a>,
    p2: Box<dyn AsyncPredicate + 'a>,
}

#[async_trait(?Send)]
impl AsyncPredicate for UnionPredicate<'_> {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        Ok(self.p1.matches(index, pos).await? || self.p2.matches(index, pos).await?)
    }
}

impl<S1, S2> ToAsyncPredicateFn for UnionRevset<S1, S2>
where
    S1: ToAsyncPredicateFn,
    S2: ToAsyncPredicateFn,
{
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(UnionPredicate {
            p1: self.set1.to_async_predicate_fn(),
            p2: self.set2.to_async_predicate_fn(),
        })
    }
}

/// `RevWalk` node that merges two sorted walk nodes.
///
/// The input items should be sorted in ascending order by the `cmp` function.
#[allow(dead_code)]
struct UnionRevWalk<I: ?Sized, W1: RevWalk<I>, W2: RevWalk<I>, C> {
    walk1: PeekableRevWalk<I, W1>,
    walk2: PeekableRevWalk<I, W2>,
    cmp: C,
}

impl<I, T, E, W1, W2, C> RevWalk<I> for UnionRevWalk<I, W1, W2, C>
where
    I: ?Sized,
    W1: RevWalk<I, Item = Result<T, E>>,
    W2: RevWalk<I, Item = Result<T, E>>,
    C: FnMut(&T, &T) -> Ordering,
{
    type Item = W1::Item;

    fn next(&mut self, index: &I) -> Option<Self::Item> {
        match (self.walk1.peek(index), self.walk2.peek(index)) {
            (None, _) => self.walk2.next(index),
            (_, None) => self.walk1.next(index),
            (Some(Ok(item1)), Some(Ok(item2))) => match (self.cmp)(item1, item2) {
                Ordering::Less => self.walk1.next(index),
                Ordering::Equal => {
                    self.walk2.next(index);
                    self.walk1.next(index)
                }
                Ordering::Greater => self.walk2.next(index),
            },
            (Some(Err(_)), _) => self.walk1.next(index),
            (_, Some(Err(_))) => self.walk2.next(index),
        }
    }
}

#[allow(dead_code)]
fn union_by<I, T, E, W1, W2, C>(walk1: W1, walk2: W2, cmp: C) -> UnionRevWalk<I, W1, W2, C>
where
    I: ?Sized,
    W1: RevWalk<I, Item = Result<T, E>>,
    W2: RevWalk<I, Item = Result<T, E>>,
    C: FnMut(&T, &T) -> Ordering,
{
    UnionRevWalk {
        walk1: walk1.peekable(),
        walk2: walk2.peekable(),
        cmp,
    }
}

/// Stream that merges two sorted streams, yielding elements from both in sorted order.
fn union_stream<'a>(
    stream1: PositionsStream<'a>,
    stream2: PositionsStream<'a>,
) -> PositionsStream<'a> {
    struct UnionState<'a> {
        stream1: PositionsStream<'a>,
        stream2: PositionsStream<'a>,
        buf1: Option<Result<GlobalCommitPosition, RevsetEvaluationError>>,
        buf2: Option<Result<GlobalCommitPosition, RevsetEvaluationError>>,
        initialized: bool,
    }

    let state = UnionState {
        stream1,
        stream2,
        buf1: None,
        buf2: None,
        initialized: false,
    };

    Box::pin(futures::stream::unfold(state, |mut state| async move {
        if !state.initialized {
            state.buf1 = state.stream1.next().await;
            state.buf2 = state.stream2.next().await;
            state.initialized = true;
        }
        loop {
            match (&state.buf1, &state.buf2) {
                (None, None) => return None,
                (Some(_), None) => {
                    let item = state.buf1.take().unwrap();
                    state.buf1 = state.stream1.next().await;
                    return Some((item, state));
                }
                (None, Some(_)) => {
                    let item = state.buf2.take().unwrap();
                    state.buf2 = state.stream2.next().await;
                    return Some((item, state));
                }
                (Some(Ok(pos1)), Some(Ok(pos2))) => match pos1.cmp(pos2).reverse() {
                    Ordering::Less => {
                        let item = state.buf1.take().unwrap();
                        state.buf1 = state.stream1.next().await;
                        return Some((item, state));
                    }
                    Ordering::Equal => {
                        state.buf2 = state.stream2.next().await;
                        let item = state.buf1.take().unwrap();
                        state.buf1 = state.stream1.next().await;
                        return Some((item, state));
                    }
                    Ordering::Greater => {
                        let item = state.buf2.take().unwrap();
                        state.buf2 = state.stream2.next().await;
                        return Some((item, state));
                    }
                },
                (Some(Err(_)), _) => {
                    let item = state.buf1.take().unwrap();
                    state.buf1 = state.stream1.next().await;
                    return Some((item, state));
                }
                (_, Some(Err(_))) => {
                    let item = state.buf2.take().unwrap();
                    state.buf2 = state.stream2.next().await;
                    return Some((item, state));
                }
            }
        }
    }))
}

/// Stream that yields elements present in both sorted streams.
fn intersection_stream<'a>(
    stream1: PositionsStream<'a>,
    stream2: PositionsStream<'a>,
) -> PositionsStream<'a> {
    struct IntersectionState<'a> {
        stream1: PositionsStream<'a>,
        stream2: PositionsStream<'a>,
        buf1: Option<Result<GlobalCommitPosition, RevsetEvaluationError>>,
        buf2: Option<Result<GlobalCommitPosition, RevsetEvaluationError>>,
        initialized: bool,
    }

    let state = IntersectionState {
        stream1,
        stream2,
        buf1: None,
        buf2: None,
        initialized: false,
    };

    Box::pin(futures::stream::unfold(state, |mut state| async move {
        if !state.initialized {
            state.buf1 = state.stream1.next().await;
            state.buf2 = state.stream2.next().await;
            state.initialized = true;
        }
        loop {
            match (&state.buf1, &state.buf2) {
                (None, _) | (_, None) => return None,
                (Some(Ok(pos1)), Some(Ok(pos2))) => match pos1.cmp(pos2).reverse() {
                    Ordering::Less => {
                        state.buf1 = state.stream1.next().await;
                    }
                    Ordering::Equal => {
                        state.buf2 = state.stream2.next().await;
                        let item = state.buf1.take().unwrap();
                        state.buf1 = state.stream1.next().await;
                        return Some((item, state));
                    }
                    Ordering::Greater => {
                        state.buf2 = state.stream2.next().await;
                    }
                },
                (Some(Err(_)), _) => {
                    let item = state.buf1.take().unwrap();
                    state.buf1 = state.stream1.next().await;
                    return Some((item, state));
                }
                (_, Some(Err(_))) => {
                    let item = state.buf2.take().unwrap();
                    state.buf2 = state.stream2.next().await;
                    return Some((item, state));
                }
            }
        }
    }))
}

/// Stream that yields elements from stream1 that are not in stream2.
fn difference_stream<'a>(
    stream1: PositionsStream<'a>,
    stream2: PositionsStream<'a>,
) -> PositionsStream<'a> {
    struct DifferenceState<'a> {
        stream1: PositionsStream<'a>,
        stream2: PositionsStream<'a>,
        buf1: Option<Result<GlobalCommitPosition, RevsetEvaluationError>>,
        buf2: Option<Result<GlobalCommitPosition, RevsetEvaluationError>>,
        initialized: bool,
    }

    let state = DifferenceState {
        stream1,
        stream2,
        buf1: None,
        buf2: None,
        initialized: false,
    };

    Box::pin(futures::stream::unfold(state, |mut state| async move {
        if !state.initialized {
            state.buf1 = state.stream1.next().await;
            state.buf2 = state.stream2.next().await;
            state.initialized = true;
        }
        loop {
            match (&state.buf1, &state.buf2) {
                (None, _) => return None,
                (Some(_), None) => {
                    let item = state.buf1.take().unwrap();
                    state.buf1 = state.stream1.next().await;
                    return Some((item, state));
                }
                (Some(Ok(pos1)), Some(Ok(pos2))) => match pos1.cmp(pos2).reverse() {
                    Ordering::Less => {
                        let item = state.buf1.take().unwrap();
                        state.buf1 = state.stream1.next().await;
                        return Some((item, state));
                    }
                    Ordering::Equal => {
                        state.buf1 = state.stream1.next().await;
                        state.buf2 = state.stream2.next().await;
                    }
                    Ordering::Greater => {
                        state.buf2 = state.stream2.next().await;
                    }
                },
                (Some(Err(_)), _) => {
                    let item = state.buf1.take().unwrap();
                    state.buf1 = state.stream1.next().await;
                    return Some((item, state));
                }
                (_, Some(Err(_))) => {
                    let item = state.buf2.take().unwrap();
                    state.buf2 = state.stream2.next().await;
                    return Some((item, state));
                }
            }
        }
    }))
}

#[derive(Debug)]
struct IntersectionRevset<S1, S2> {
    set1: S1,
    set2: S2,
}

impl<S1, S2> InternalRevset for IntersectionRevset<S1, S2>
where
    S1: InternalRevset,
    S2: InternalRevset,
{
    fn positions_stream<'a>(&'a self, index: &'a CompositeIndex) -> PositionsStream<'a> {
        intersection_stream(
            self.set1.positions_stream(index),
            self.set2.positions_stream(index),
        )
    }
}

struct IntersectionPredicate<'a> {
    p1: Box<dyn AsyncPredicate + 'a>,
    p2: Box<dyn AsyncPredicate + 'a>,
}

#[async_trait(?Send)]
impl AsyncPredicate for IntersectionPredicate<'_> {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        Ok(self.p1.matches(index, pos).await? && self.p2.matches(index, pos).await?)
    }
}

impl<S1, S2> ToAsyncPredicateFn for IntersectionRevset<S1, S2>
where
    S1: ToAsyncPredicateFn,
    S2: ToAsyncPredicateFn,
{
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(IntersectionPredicate {
            p1: self.set1.to_async_predicate_fn(),
            p2: self.set2.to_async_predicate_fn(),
        })
    }
}

/// `RevWalk` node that intersects two sorted walk nodes.
///
/// The input items should be sorted in ascending order by the `cmp` function.
#[allow(dead_code)]
struct IntersectionRevWalk<I: ?Sized, W1: RevWalk<I>, W2: RevWalk<I>, C> {
    walk1: PeekableRevWalk<I, W1>,
    walk2: PeekableRevWalk<I, W2>,
    cmp: C,
}

impl<I, T, E, W1, W2, C> RevWalk<I> for IntersectionRevWalk<I, W1, W2, C>
where
    I: ?Sized,
    W1: RevWalk<I, Item = Result<T, E>>,
    W2: RevWalk<I, Item = Result<T, E>>,
    C: FnMut(&T, &T) -> Ordering,
{
    type Item = W1::Item;

    fn next(&mut self, index: &I) -> Option<Self::Item> {
        loop {
            match (self.walk1.peek(index), self.walk2.peek(index)) {
                (None, _) => {
                    return None;
                }
                (_, None) => {
                    return None;
                }
                (Some(Ok(item1)), Some(Ok(item2))) => match (self.cmp)(item1, item2) {
                    Ordering::Less => {
                        self.walk1.next(index);
                    }
                    Ordering::Equal => {
                        self.walk2.next(index);
                        return self.walk1.next(index);
                    }
                    Ordering::Greater => {
                        self.walk2.next(index);
                    }
                },
                (Some(Err(_)), _) => {
                    return self.walk1.next(index);
                }
                (_, Some(Err(_))) => {
                    return self.walk2.next(index);
                }
            }
        }
    }
}

#[allow(dead_code)]
fn intersection_by<I, T, E, W1, W2, C>(
    walk1: W1,
    walk2: W2,
    cmp: C,
) -> IntersectionRevWalk<I, W1, W2, C>
where
    I: ?Sized,
    W1: RevWalk<I, Item = Result<T, E>>,
    W2: RevWalk<I, Item = Result<T, E>>,
    C: FnMut(&T, &T) -> Ordering,
{
    IntersectionRevWalk {
        walk1: walk1.peekable(),
        walk2: walk2.peekable(),
        cmp,
    }
}

#[derive(Debug)]
struct DifferenceRevset<S1, S2> {
    // The minuend (what to subtract from)
    set1: S1,
    // The subtrahend (what to subtract)
    set2: S2,
}

impl<S1, S2> InternalRevset for DifferenceRevset<S1, S2>
where
    S1: InternalRevset,
    S2: InternalRevset,
{
    fn positions_stream<'a>(&'a self, index: &'a CompositeIndex) -> PositionsStream<'a> {
        difference_stream(
            self.set1.positions_stream(index),
            self.set2.positions_stream(index),
        )
    }
}

struct DifferencePredicate<'a> {
    p1: Box<dyn AsyncPredicate + 'a>,
    p2: Box<dyn AsyncPredicate + 'a>,
}

#[async_trait(?Send)]
impl AsyncPredicate for DifferencePredicate<'_> {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        Ok(self.p1.matches(index, pos).await? && !self.p2.matches(index, pos).await?)
    }
}

impl<S1, S2> ToAsyncPredicateFn for DifferenceRevset<S1, S2>
where
    S1: ToAsyncPredicateFn,
    S2: ToAsyncPredicateFn,
{
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(DifferencePredicate {
            p1: self.set1.to_async_predicate_fn(),
            p2: self.set2.to_async_predicate_fn(),
        })
    }
}

/// `RevWalk` node that subtracts `walk2` items from `walk1`.
///
/// The input items should be sorted in ascending order by the `cmp` function.
struct DifferenceRevWalk<I: ?Sized, W1: RevWalk<I>, W2: RevWalk<I>, C> {
    walk1: PeekableRevWalk<I, W1>,
    walk2: PeekableRevWalk<I, W2>,
    cmp: C,
}

impl<I, T, E, W1, W2, C> RevWalk<I> for DifferenceRevWalk<I, W1, W2, C>
where
    I: ?Sized,
    W1: RevWalk<I, Item = Result<T, E>>,
    W2: RevWalk<I, Item = Result<T, E>>,
    C: FnMut(&T, &T) -> Ordering,
{
    type Item = W1::Item;

    fn next(&mut self, index: &I) -> Option<Self::Item> {
        loop {
            match (self.walk1.peek(index), self.walk2.peek(index)) {
                (None, _) => {
                    return None;
                }
                (_, None) => {
                    return self.walk1.next(index);
                }
                (Some(Ok(item1)), Some(Ok(item2))) => match (self.cmp)(item1, item2) {
                    Ordering::Less => {
                        return self.walk1.next(index);
                    }
                    Ordering::Equal => {
                        self.walk2.next(index);
                        self.walk1.next(index);
                    }
                    Ordering::Greater => {
                        self.walk2.next(index);
                    }
                },
                (Some(Err(_)), _) => {
                    return self.walk1.next(index);
                }
                (_, Some(Err(_))) => {
                    return self.walk2.next(index);
                }
            }
        }
    }
}

fn difference_by<I, T, E, W1, W2, C>(
    walk1: W1,
    walk2: W2,
    cmp: C,
) -> DifferenceRevWalk<I, W1, W2, C>
where
    I: ?Sized,
    W1: RevWalk<I, Item = Result<T, E>>,
    W2: RevWalk<I, Item = Result<T, E>>,
    C: FnMut(&T, &T) -> Ordering,
{
    DifferenceRevWalk {
        walk1: walk1.peekable(),
        walk2: walk2.peekable(),
        cmp,
    }
}

pub(super) async fn evaluate<I: AsCompositeIndex + Clone>(
    expression: ResolvedExpression,
    store: Arc<Store>,
    index: I,
) -> Result<RevsetImpl<I>, RevsetEvaluationError> {
    let context = EvaluationContext {
        store: store.clone(),
        index: index.as_composite(),
    };
    let internal_revset = context.evaluate(&expression).await?;
    Ok(RevsetImpl::new(internal_revset, index))
}

struct EvaluationContext<'index> {
    store: Arc<Store>,
    index: &'index CompositeIndex,
}

fn to_u32_generation_range(range: &Range<u64>) -> Result<Range<u32>, RevsetEvaluationError> {
    let start = range.start.try_into().map_err(|_| {
        RevsetEvaluationError::Other(
            format!("Lower bound of generation ({}) is too large", range.start).into(),
        )
    })?;
    let end = range.end.try_into().unwrap_or(u32::MAX);
    Ok(start..end)
}

impl EvaluationContext<'_> {
    async fn evaluate(
        &self,
        expression: &ResolvedExpression,
    ) -> Result<Box<dyn InternalRevset>, RevsetEvaluationError> {
        let index = self.index;
        match expression {
            ResolvedExpression::Commits(commit_ids) => {
                Ok(Box::new(self.revset_for_commit_ids(commit_ids)?))
            }
            ResolvedExpression::Ancestors {
                heads,
                generation,
                parents_range,
            } => {
                let head_set = Box::pin(self.evaluate(heads)).await?;
                let head_positions: Vec<_> = head_set.positions_stream(index).try_collect().await?;
                let builder = RevWalkBuilder::new(index)
                    .wanted_heads(head_positions)
                    .wanted_parents_range(parents_range.clone());
                if generation == &GENERATION_RANGE_FULL {
                    let walk = builder.ancestors().detach();
                    Ok(Box::new(RevWalkRevset { walk }))
                } else {
                    let generation = to_u32_generation_range(generation)?;
                    let walk = builder
                        .ancestors_filtered_by_generation(generation)
                        .detach();
                    Ok(Box::new(RevWalkRevset { walk }))
                }
            }
            ResolvedExpression::Range {
                roots,
                heads,
                generation,
                parents_range,
            } => {
                let root_set = Box::pin(self.evaluate(roots)).await?;
                let root_positions: Vec<_> = root_set.positions_stream(index).try_collect().await?;
                // Pre-filter heads so queries like 'immutable_heads()..' can
                // terminate early. immutable_heads() usually includes some
                // visible heads, which can be trivially rejected.
                let head_set = Box::pin(self.evaluate(heads)).await?;
                let head_positions: Vec<_> = head_set.positions_stream(index).try_collect().await?;
                let head_positions = difference_by(
                    EagerRevWalk::new(head_positions.into_iter().map(Ok::<_, RevsetEvaluationError>)),
                    EagerRevWalk::new(root_positions.iter().copied().map(Ok::<_, RevsetEvaluationError>)),
                    |pos1, pos2| pos1.cmp(pos2).reverse(),
                )
                .attach(index);
                let builder = RevWalkBuilder::new(index)
                    .wanted_heads(head_positions.try_collect()?)
                    .wanted_parents_range(parents_range.clone())
                    .unwanted_roots(root_positions);
                if generation == &GENERATION_RANGE_FULL {
                    let walk = builder.ancestors().detach();
                    Ok(Box::new(RevWalkRevset { walk }))
                } else {
                    let generation = to_u32_generation_range(generation)?;
                    let walk = builder
                        .ancestors_filtered_by_generation(generation)
                        .detach();
                    Ok(Box::new(RevWalkRevset { walk }))
                }
            }
            ResolvedExpression::DagRange {
                roots,
                heads,
                generation_from_roots,
            } => {
                let root_set = Box::pin(self.evaluate(roots)).await?;
                let root_positions: Vec<_> = root_set.positions_stream(index).try_collect().await?;
                let head_set = Box::pin(self.evaluate(heads)).await?;
                let head_positions: Vec<_> = head_set.positions_stream(index).try_collect().await?;
                let builder =
                    RevWalkBuilder::new(index).wanted_heads(head_positions);
                if generation_from_roots == &(1..2) {
                    let root_positions_set: HashSet<_> = root_positions.iter().copied().collect();
                    let walk = builder
                        .ancestors_until_roots(root_positions_set.iter().copied())
                        .detach();
                    let candidates = RevWalkRevset { walk };
                    let predicate = PurePredicateFn(move |index: &CompositeIndex, pos| {
                        Ok(index
                            .commits()
                            .entry_by_pos(pos)
                            .parent_positions()
                            .iter()
                            .any(|parent_pos| root_positions_set.contains(parent_pos)))
                    });
                    // TODO: Suppose heads include all visible heads, ToAsyncPredicateFn version can be
                    // optimized to only test the predicate()
                    Ok(Box::new(FilterRevset {
                        candidates,
                        predicate,
                    }))
                } else if generation_from_roots == &GENERATION_RANGE_FULL {
                    let root_positions_set: HashSet<_> = root_positions.into_iter().collect();
                    let mut positions = builder
                        .descendants(root_positions_set)
                        .collect_vec();
                    positions.reverse();
                    Ok(Box::new(EagerRevset { positions }))
                } else {
                    // For small generation range, it might be better to build a reachable map
                    // with generation bit set, which can be calculated incrementally from roots:
                    //   reachable[pos] = (reachable[parent_pos] | ...) << 1
                    let mut positions = builder
                        .descendants_filtered_by_generation(
                            root_positions,
                            to_u32_generation_range(generation_from_roots)?,
                        )
                        .map(|Reverse(pos)| pos)
                        .collect_vec();
                    positions.reverse();
                    Ok(Box::new(EagerRevset { positions }))
                }
            }
            ResolvedExpression::Reachable { sources, domain } => {
                let mut sets = union_find::UnionFind::<GlobalCommitPosition>::new();

                // Compute all reachable subgraphs.
                let domain_revset = Box::pin(self.evaluate(domain)).await?;
                let domain_vec: Vec<_> = domain_revset.positions_stream(index).try_collect().await?;
                let domain_set: HashSet<_> = domain_vec.iter().copied().collect();
                for pos in &domain_set {
                    for parent_pos in index.commits().entry_by_pos(*pos).parent_positions() {
                        if domain_set.contains(&parent_pos) {
                            sets.union(*pos, parent_pos);
                        }
                    }
                }
                // `UnionFind::find` is somewhat slow, so it's faster to only do this once and
                // then cache the result.
                let domain_reps = domain_vec.iter().map(|&pos| sets.find(pos)).collect_vec();

                // Identify disjoint sets reachable from sources. Using a predicate here can be
                // significantly faster for cases like `reachable(filter, X)`, since the filter
                // can be checked for only commits in `X` instead of for all visible commits,
                // and the difference is usually negligible for non-filter revsets.
                let sources_revset = Box::pin(self.evaluate(sources)).await?;
                let mut sources_predicate = sources_revset.to_async_predicate_fn();
                let mut set_reps = HashSet::new();
                for (&pos, &rep) in domain_vec.iter().zip(&domain_reps) {
                    // Skip evaluating predicate if `rep` has already been added.
                    if set_reps.contains(&rep) {
                        continue;
                    }
                    if sources_predicate.matches(index, pos).await? {
                        set_reps.insert(rep);
                    }
                }

                let positions = domain_vec
                    .into_iter()
                    .zip(domain_reps)
                    .filter_map(|(pos, rep)| set_reps.contains(&rep).then_some(pos))
                    .collect_vec();
                Ok(Box::new(EagerRevset { positions }))
            }
            ResolvedExpression::Heads(candidates) => {
                let candidate_set = Box::pin(self.evaluate(candidates)).await?;
                let candidate_positions: Vec<_> = candidate_set.positions_stream(index).try_collect().await?;
                let positions = index
                    .commits()
                    .heads_pos(candidate_positions);
                Ok(Box::new(EagerRevset { positions }))
            }
            ResolvedExpression::HeadsRange {
                roots,
                heads,
                parents_range,
                filter,
            } => {
                let root_set = Box::pin(self.evaluate(roots)).await?;
                let root_positions: Vec<_> = root_set.positions_stream(index).try_collect().await?;
                // Pre-filter heads so queries like 'immutable_heads()..' can
                // terminate early. immutable_heads() usually includes some
                // visible heads, which can be trivially rejected.
                let head_set = Box::pin(self.evaluate(heads)).await?;
                let head_positions: Vec<_> = head_set.positions_stream(index).try_collect().await?;
                let head_positions: Vec<_> = difference_by(
                    EagerRevWalk::new(head_positions.into_iter().map(Ok::<_, RevsetEvaluationError>)),
                    EagerRevWalk::new(root_positions.iter().copied().map(Ok::<_, RevsetEvaluationError>)),
                    |pos1, pos2| pos1.cmp(pos2).reverse(),
                )
                .attach(index)
                .try_collect()?;
                let positions = if let Some(filter) = filter {
                    let filter_fn = Box::pin(self.evaluate_predicate(filter)).await?;
                    let mut filter_pred = filter_fn.to_async_predicate_fn();
                    // Pre-collect which positions pass the filter, then use that for heads computation.
                    // We need to collect because heads_from_range_and_filter expects a sync predicate.
                    // Walk the range to find all positions that pass the filter.
                    let range_walk = RevWalkBuilder::new(index)
                        .wanted_heads(head_positions.clone())
                        .unwanted_roots(root_positions.clone());
                    let range_positions: Vec<_> = range_walk.ancestors().collect();
                    let mut passing_positions = HashSet::new();
                    for pos in &range_positions {
                        if filter_pred.matches(index, *pos).await? {
                            passing_positions.insert(*pos);
                        }
                    }
                    // Now compute heads using the pre-computed filter
                    index.commits().heads_from_range_and_filter::<Infallible>(
                        root_positions,
                        head_positions,
                        parents_range,
                        |pos| Ok(passing_positions.contains(&pos)),
                    ).unwrap()
                } else {
                    let Ok(positions) = index.commits().heads_from_range_and_filter::<Infallible>(
                        root_positions,
                        head_positions,
                        parents_range,
                        |_| Ok(true),
                    );
                    positions
                };
                Ok(Box::new(EagerRevset { positions }))
            }
            ResolvedExpression::Roots(candidates) => {
                let mut positions: Vec<_> = Box::pin(self.evaluate(candidates))
                    .await?
                    .positions_stream(index)
                    .try_collect()
                    .await?;
                let filled = RevWalkBuilder::new(index)
                    .wanted_heads(positions.clone())
                    .descendants(positions.iter().copied().collect())
                    .collect_positions_set();
                positions.retain(|&pos| {
                    !index
                        .commits()
                        .entry_by_pos(pos)
                        .parent_positions()
                        .iter()
                        .any(|parent| filled.contains(parent))
                });
                Ok(Box::new(EagerRevset { positions }))
            }
            ResolvedExpression::ForkPoint(expression) => {
                let expression_set = Box::pin(self.evaluate(expression)).await?;
                let expression_positions: Vec<_> = expression_set.positions_stream(index).try_collect().await?;
                let mut positions_iter = expression_positions.into_iter();
                let Some(position) = positions_iter.next() else {
                    return Ok(Box::new(EagerRevset::empty()));
                };
                let mut positions = vec![position];
                for position in positions_iter {
                    positions = index
                        .commits()
                        .common_ancestors_pos(positions, vec![position]);
                }
                Ok(Box::new(EagerRevset { positions }))
            }
            ResolvedExpression::Bisect(candidates) => {
                let set = Box::pin(self.evaluate(candidates)).await?;
                // TODO: Make this more correct in non-linear history
                let candidate_positions: Vec<_> = set.positions_stream(index).try_collect().await?;
                let positions = if candidate_positions.is_empty() {
                    candidate_positions
                } else {
                    vec![candidate_positions[candidate_positions.len() / 2]]
                };
                Ok(Box::new(EagerRevset { positions }))
            }
            ResolvedExpression::Latest { candidates, count } => {
                let candidate_set = Box::pin(self.evaluate(candidates)).await?;
                Ok(Box::new(self.take_latest_revset(&*candidate_set, *count).await?))
            }
            ResolvedExpression::HasSize { candidates, count } => {
                let set = Box::pin(self.evaluate(candidates)).await?;
                let positions: Vec<_> = set
                    .positions_stream(index)
                    .take(count.saturating_add(1))
                    .try_collect()
                    .await?;
                if positions.len() != *count {
                    // https://github.com/jj-vcs/jj/pull/7252#pullrequestreview-3236259998
                    // in the default engine we have to evaluate the entire
                    // revset (which may be very large) to get an exact count;
                    // we would need to remove .take() above. instead just give
                    // a vaguely approximate error message
                    let determiner = if positions.len() > *count {
                        "more"
                    } else {
                        "fewer"
                    };
                    return Err(RevsetEvaluationError::Other(
                        format!("The revset has {determiner} than the expected {count} revisions")
                            .into(),
                    ));
                }
                Ok(Box::new(EagerRevset { positions }))
            }
            ResolvedExpression::Coalesce(expression1, expression2) => {
                let set1 = Box::pin(self.evaluate(expression1)).await?;
                if set1.positions_stream(index).next().await.is_some() {
                    Ok(set1)
                } else {
                    Box::pin(self.evaluate(expression2)).await
                }
            }
            ResolvedExpression::Union(expression1, expression2) => {
                let set1 = Box::pin(self.evaluate(expression1)).await?;
                let set2 = Box::pin(self.evaluate(expression2)).await?;
                Ok(Box::new(UnionRevset { set1, set2 }))
            }
            ResolvedExpression::FilterWithin {
                candidates,
                predicate,
            } => Ok(Box::new(FilterRevset {
                candidates: Box::pin(self.evaluate(candidates)).await?,
                predicate: Box::pin(self.evaluate_predicate(predicate)).await?,
            })),
            ResolvedExpression::Intersection(expression1, expression2) => {
                let set1 = Box::pin(self.evaluate(expression1)).await?;
                let set2 = Box::pin(self.evaluate(expression2)).await?;
                Ok(Box::new(IntersectionRevset { set1, set2 }))
            }
            ResolvedExpression::Difference(expression1, expression2) => {
                let set1 = Box::pin(self.evaluate(expression1)).await?;
                let set2 = Box::pin(self.evaluate(expression2)).await?;
                Ok(Box::new(DifferenceRevset { set1, set2 }))
            }
        }
    }

    async fn evaluate_predicate(
        &self,
        expression: &ResolvedPredicateExpression,
    ) -> Result<Box<dyn ToAsyncPredicateFn>, RevsetEvaluationError> {
        match expression {
            ResolvedPredicateExpression::Filter(predicate) => {
                Ok(build_predicate_fn(self.store.clone(), predicate).await)
            }
            ResolvedPredicateExpression::Divergent { visible_heads } => {
                let composite = self.index.as_composite().commits();
                let mut reachable_set = AncestorsBitSet::with_capacity(composite.num_commits());
                for id in visible_heads {
                    reachable_set.add_head(composite.commit_id_to_pos(id).unwrap());
                }
                let reachable_set = Rc::new(RefCell::new(reachable_set));
                Ok(box_pure_predicate_fn(
                    move |index: &CompositeIndex, pos: GlobalCommitPosition| {
                        let commits = index.commits();

                        match commits.resolve_change_id_prefix(&HexPrefix::from_id(
                            &commits.entry_by_pos(pos).change_id(),
                        )) {
                            PrefixResolution::NoMatch => {
                                panic!("the commit itself should be reachable")
                            }
                            PrefixResolution::SingleMatch((_change_id, positions)) => {
                                let mut reachable_set = reachable_set.borrow_mut();
                                let targets = commits.resolve_change_targets_for_positions(
                                    &positions,
                                    &mut reachable_set,
                                );
                                Ok(targets.is_divergent())
                            }
                            PrefixResolution::AmbiguousMatch => {
                                panic!("complete change_id should be unambiguous")
                            }
                        }
                    },
                ))
            }
            ResolvedPredicateExpression::Set(expression) => Ok(self.evaluate(expression).await?),
            ResolvedPredicateExpression::NotIn(complement) => {
                let set = Box::pin(self.evaluate_predicate(complement)).await?;
                Ok(Box::new(NotInPredicate(set)))
            }
            ResolvedPredicateExpression::Union(expression1, expression2) => {
                let set1 = Box::pin(self.evaluate_predicate(expression1)).await?;
                let set2 = Box::pin(self.evaluate_predicate(expression2)).await?;
                Ok(Box::new(UnionRevset { set1, set2 }))
            }
            ResolvedPredicateExpression::Intersection(expression1, expression2) => {
                let set1 = Box::pin(self.evaluate_predicate(expression1)).await?;
                let set2 = Box::pin(self.evaluate_predicate(expression2)).await?;
                Ok(Box::new(IntersectionRevset { set1, set2 }))
            }
        }
    }

    fn revset_for_commit_ids(
        &self,
        commit_ids: &[CommitId],
    ) -> Result<EagerRevset, RevsetEvaluationError> {
        let mut positions: Vec<_> = commit_ids
            .iter()
            .map(|id| {
                // Invalid commit IDs should be rejected by the revset frontend,
                // but there are a few edge cases that break the precondition.
                // For example, in jj <= 0.22, the root commit doesn't exist in
                // the root operation.
                self.index.commits().commit_id_to_pos(id).ok_or_else(|| {
                    RevsetEvaluationError::Other(
                        format!(
                            "Commit ID {} not found in index (index or view might be corrupted)",
                            id.hex()
                        )
                        .into(),
                    )
                })
            })
            .try_collect()?;
        positions.sort_unstable_by_key(|&pos| Reverse(pos));
        positions.dedup();
        Ok(EagerRevset { positions })
    }

    async fn take_latest_revset(
        &self,
        candidate_set: &dyn InternalRevset,
        count: usize,
    ) -> Result<EagerRevset, RevsetEvaluationError> {
        if count == 0 {
            return Ok(EagerRevset::empty());
        }

        #[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
        struct Item {
            timestamp: MillisSinceEpoch,
            pos: GlobalCommitPosition, // tie-breaker
        }

        // Maintain min-heap containing the latest (greatest) count items. For small
        // count and large candidate set, this is probably cheaper than building vec
        // and applying selection algorithm.
        let mut latest_items: BinaryHeap<Reverse<Item>> = BinaryHeap::new();
        let mut candidate_stream = candidate_set.positions_stream(self.index);

        while let Some(pos_result) = candidate_stream.next().await {
            let pos = pos_result?;
            let entry = self.index.commits().entry_by_pos(pos);
            let commit = self.store.get_commit_async(&entry.commit_id()).await?;
            let item = Reverse(Item {
                timestamp: commit.committer().timestamp.timestamp,
                pos: entry.position(),
            });

            if latest_items.len() < count {
                latest_items.push(item);
            } else if let Some(mut earliest) = latest_items.peek_mut() {
                if earliest.0 < item.0 {
                    *earliest = item;
                }
            }
        }

        assert!(latest_items.len() <= count);
        let mut positions = latest_items
            .into_iter()
            .map(|item| item.0.pos)
            .collect_vec();
        positions.sort_unstable_by_key(|&pos| Reverse(pos));
        Ok(EagerRevset { positions })
    }
}

/// A wrapper that implements `ToAsyncPredicateFn` for sync predicates that don't need async.
struct PurePredicateFn<F>(F);

impl<F> fmt::Debug for PurePredicateFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PurePredicateFn").finish_non_exhaustive()
    }
}

struct PurePredicateImpl<F>(F);

#[async_trait(?Send)]
impl<F> AsyncPredicate for PurePredicateImpl<F>
where
    F: Fn(&CompositeIndex, GlobalCommitPosition) -> Result<bool, RevsetEvaluationError>,
{
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        (self.0)(index, pos)
    }
}

impl<F> ToAsyncPredicateFn for PurePredicateFn<F>
where
    F: Fn(&CompositeIndex, GlobalCommitPosition) -> Result<bool, RevsetEvaluationError> + Clone,
{
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(PurePredicateImpl(self.0.clone()))
    }
}

fn box_pure_predicate_fn<'a, F>(f: F) -> Box<dyn ToAsyncPredicateFn + 'a>
where
    F: Fn(&CompositeIndex, GlobalCommitPosition) -> Result<bool, RevsetEvaluationError>
        + Clone
        + 'a,
{
    Box::new(PurePredicateFn(f))
}

/// A wrapper that implements `ToAsyncPredicateFn` for async predicates.
#[allow(dead_code)]
struct AsyncPredicateWrapper<F>(F);

impl<F> fmt::Debug for AsyncPredicateWrapper<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncPredicateWrapper").finish_non_exhaustive()
    }
}

#[allow(dead_code)]
struct AsyncPredicateImpl<F>(F);

#[async_trait(?Send)]
impl<F, Fut> AsyncPredicate for AsyncPredicateImpl<F>
where
    F: FnMut(&CompositeIndex, GlobalCommitPosition) -> Fut,
    Fut: std::future::Future<Output = Result<bool, RevsetEvaluationError>>,
{
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        (self.0)(index, pos).await
    }
}

impl<F, Fut> ToAsyncPredicateFn for AsyncPredicateWrapper<F>
where
    F: FnMut(&CompositeIndex, GlobalCommitPosition) -> Fut + Clone,
    Fut: std::future::Future<Output = Result<bool, RevsetEvaluationError>>,
{
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(AsyncPredicateImpl(self.0.clone()))
    }
}

#[allow(dead_code)]
fn box_async_predicate_fn<'a, F, Fut>(f: F) -> Box<dyn ToAsyncPredicateFn + 'a>
where
    F: FnMut(&CompositeIndex, GlobalCommitPosition) -> Fut + Clone + 'a,
    Fut: std::future::Future<Output = Result<bool, RevsetEvaluationError>> + 'a,
{
    Box::new(AsyncPredicateWrapper(f))
}

/// Predicate that checks description matches
struct DescriptionPredicate {
    store: Arc<Store>,
    matcher: Rc<StringMatcher>,
}

impl fmt::Debug for DescriptionPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DescriptionPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for DescriptionPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.matcher.is_match(commit.description()))
    }
}

impl ToAsyncPredicateFn for DescriptionPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            matcher: self.matcher.clone(),
        })
    }
}

/// Predicate that checks subject (first line of description) matches
struct SubjectPredicate {
    store: Arc<Store>,
    matcher: Rc<StringMatcher>,
}

impl fmt::Debug for SubjectPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubjectPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for SubjectPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.matcher.is_match(commit.description().lines().next().unwrap_or_default()))
    }
}

impl ToAsyncPredicateFn for SubjectPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            matcher: self.matcher.clone(),
        })
    }
}

/// Predicate that checks author name matches
struct AuthorNamePredicate {
    store: Arc<Store>,
    matcher: Rc<StringMatcher>,
}

impl fmt::Debug for AuthorNamePredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthorNamePredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for AuthorNamePredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.matcher.is_match(&commit.author().name))
    }
}

impl ToAsyncPredicateFn for AuthorNamePredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            matcher: self.matcher.clone(),
        })
    }
}

/// Predicate that checks author email matches
struct AuthorEmailPredicate {
    store: Arc<Store>,
    matcher: Rc<StringMatcher>,
}

impl fmt::Debug for AuthorEmailPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthorEmailPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for AuthorEmailPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.matcher.is_match(&commit.author().email))
    }
}

impl ToAsyncPredicateFn for AuthorEmailPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            matcher: self.matcher.clone(),
        })
    }
}

/// Predicate that checks author date matches
struct AuthorDatePredicate {
    store: Arc<Store>,
    expression: crate::time_util::DatePattern,
}

impl fmt::Debug for AuthorDatePredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthorDatePredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for AuthorDatePredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.expression.matches(&commit.author().timestamp))
    }
}

impl ToAsyncPredicateFn for AuthorDatePredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            expression: self.expression,
        })
    }
}

/// Predicate that checks committer name matches
struct CommitterNamePredicate {
    store: Arc<Store>,
    matcher: Rc<StringMatcher>,
}

impl fmt::Debug for CommitterNamePredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommitterNamePredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for CommitterNamePredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.matcher.is_match(&commit.committer().name))
    }
}

impl ToAsyncPredicateFn for CommitterNamePredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            matcher: self.matcher.clone(),
        })
    }
}

/// Predicate that checks committer email matches
struct CommitterEmailPredicate {
    store: Arc<Store>,
    matcher: Rc<StringMatcher>,
}

impl fmt::Debug for CommitterEmailPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommitterEmailPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for CommitterEmailPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.matcher.is_match(&commit.committer().email))
    }
}

impl ToAsyncPredicateFn for CommitterEmailPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            matcher: self.matcher.clone(),
        })
    }
}

/// Predicate that checks committer date matches
struct CommitterDatePredicate {
    store: Arc<Store>,
    expression: crate::time_util::DatePattern,
}

impl fmt::Debug for CommitterDatePredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommitterDatePredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for CommitterDatePredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.expression.matches(&commit.committer().timestamp))
    }
}

impl ToAsyncPredicateFn for CommitterDatePredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            expression: self.expression,
        })
    }
}

/// Predicate that checks file paths changed
struct FilePredicate {
    store: Arc<Store>,
    matcher: Rc<dyn Matcher>,
}

impl fmt::Debug for FilePredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilePredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for FilePredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        if let Some(mut paths) = index.changed_paths().changed_paths(pos) {
            return Ok(paths.any(|path| self.matcher.matches(path)));
        }
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(has_diff_from_parent(&self.store, index, &commit, &*self.matcher).await?)
    }
}

impl ToAsyncPredicateFn for FilePredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            matcher: self.matcher.clone(),
        })
    }
}

/// Predicate that checks diff contains text
struct DiffContainsPredicate {
    store: Arc<Store>,
    text_matcher: Rc<StringMatcher>,
    files_matcher: Rc<dyn Matcher>,
}

impl fmt::Debug for DiffContainsPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiffContainsPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for DiffContainsPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let narrowed_files_matcher;
        let files_matcher = if let Some(paths) = index.changed_paths().changed_paths(pos) {
            let matched_paths = paths
                .filter(|path| self.files_matcher.matches(path))
                .collect_vec();
            if matched_paths.is_empty() {
                return Ok(false);
            }
            narrowed_files_matcher = FilesMatcher::new(matched_paths);
            &narrowed_files_matcher as &dyn Matcher
        } else {
            &*self.files_matcher
        };
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(matches_diff_from_parent(&self.store, index, &commit, &self.text_matcher, files_matcher).await?)
    }
}

impl ToAsyncPredicateFn for DiffContainsPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            text_matcher: self.text_matcher.clone(),
            files_matcher: self.files_matcher.clone(),
        })
    }
}

/// Predicate that checks if commit has conflict
struct HasConflictPredicate {
    store: Arc<Store>,
}

impl fmt::Debug for HasConflictPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HasConflictPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for HasConflictPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(commit.has_conflict())
    }
}

impl ToAsyncPredicateFn for HasConflictPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
        })
    }
}

/// Predicate that checks if commit is signed
struct SignedPredicate {
    store: Arc<Store>,
}

impl fmt::Debug for SignedPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SignedPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for SignedPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(commit.is_signed())
    }
}

impl ToAsyncPredicateFn for SignedPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
        })
    }
}

/// Predicate that checks extension predicate
struct ExtensionPredicate {
    store: Arc<Store>,
    ext: Arc<dyn crate::revset::RevsetFilterExtension>,
}

impl fmt::Debug for ExtensionPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionPredicate").finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl AsyncPredicate for ExtensionPredicate {
    async fn matches(
        &mut self,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        let entry = index.commits().entry_by_pos(pos);
        let commit = self.store.get_commit_async(&entry.commit_id()).await?;
        Ok(self.ext.matches_commit(&commit))
    }
}

impl ToAsyncPredicateFn for ExtensionPredicate {
    fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
        Box::new(Self {
            store: self.store.clone(),
            ext: self.ext.clone(),
        })
    }
}

async fn build_predicate_fn(
    store: Arc<Store>,
    predicate: &RevsetFilterPredicate,
) -> Box<dyn ToAsyncPredicateFn> {
    match predicate {
        RevsetFilterPredicate::ParentCount(parent_count_range) => {
            let parent_count_range = parent_count_range.clone();
            box_pure_predicate_fn(move |index, pos| {
                let entry = index.commits().entry_by_pos(pos);
                Ok(parent_count_range.contains(&entry.num_parents()))
            })
        }
        RevsetFilterPredicate::Description(expression) => {
            Box::new(DescriptionPredicate {
                store,
                matcher: Rc::new(expression.to_matcher()),
            })
        }
        RevsetFilterPredicate::Subject(expression) => {
            Box::new(SubjectPredicate {
                store,
                matcher: Rc::new(expression.to_matcher()),
            })
        }
        RevsetFilterPredicate::AuthorName(expression) => {
            Box::new(AuthorNamePredicate {
                store,
                matcher: Rc::new(expression.to_matcher()),
            })
        }
        RevsetFilterPredicate::AuthorEmail(expression) => {
            Box::new(AuthorEmailPredicate {
                store,
                matcher: Rc::new(expression.to_matcher()),
            })
        }
        RevsetFilterPredicate::AuthorDate(expression) => {
            Box::new(AuthorDatePredicate {
                store,
                expression: *expression,
            })
        }
        RevsetFilterPredicate::CommitterName(expression) => {
            Box::new(CommitterNamePredicate {
                store,
                matcher: Rc::new(expression.to_matcher()),
            })
        }
        RevsetFilterPredicate::CommitterEmail(expression) => {
            Box::new(CommitterEmailPredicate {
                store,
                matcher: Rc::new(expression.to_matcher()),
            })
        }
        RevsetFilterPredicate::CommitterDate(expression) => {
            Box::new(CommitterDatePredicate {
                store,
                expression: *expression,
            })
        }
        RevsetFilterPredicate::File(expr) => {
            Box::new(FilePredicate {
                store,
                matcher: expr.to_matcher().into(),
            })
        }
        RevsetFilterPredicate::DiffLines { text, files } => {
            Box::new(DiffContainsPredicate {
                store,
                text_matcher: Rc::new(text.to_matcher()),
                files_matcher: files.to_matcher().into(),
            })
        }
        RevsetFilterPredicate::HasConflict => {
            Box::new(HasConflictPredicate { store })
        }
        RevsetFilterPredicate::Signed => {
            Box::new(SignedPredicate { store })
        }
        RevsetFilterPredicate::Extension(ext) => {
            Box::new(ExtensionPredicate {
                store,
                ext: ext.clone(),
            })
        }
    }
}

async fn has_diff_from_parent(
    store: &Arc<Store>,
    index: &CompositeIndex,
    commit: &Commit,
    matcher: &dyn Matcher,
) -> BackendResult<bool> {
    let parents: Vec<_> = commit.parents_async().await?;
    if let [parent] = parents.as_slice() {
        // Fast path: no need to load the root tree
        let unchanged = commit.tree_ids() == parent.tree_ids();
        if matcher.visit(RepoPath::root()) == Visit::AllRecursively {
            return Ok(!unchanged);
        } else if unchanged {
            return Ok(false);
        }
    }

    // Conflict resolution is expensive, try that only for matched files.
    let from_tree =
        rewrite::merge_commit_trees_no_resolve_without_repo(store, index, &parents).await?;
    let to_tree = commit.tree();
    // TODO: handle copy tracking
    let mut tree_diff = from_tree.diff_stream(&to_tree, matcher);
    // TODO: Resolve values concurrently
    while let Some(entry) = tree_diff.next().await {
        let mut values = entry.values?;
        values.before = resolve_file_values(store, &entry.path, values.before).await?;
        if !values.is_changed() {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

async fn matches_diff_from_parent(
    store: &Arc<Store>,
    index: &CompositeIndex,
    commit: &Commit,
    text_matcher: &StringMatcher,
    files_matcher: &dyn Matcher,
) -> BackendResult<bool> {
    let parents: Vec<_> = commit.parents_async().await?;
    // Conflict resolution is expensive, try that only for matched files.
    let from_tree =
        rewrite::merge_commit_trees_no_resolve_without_repo(store, index, &parents).await?;
    let to_tree = commit.tree();
    // TODO: handle copy tracking
    let mut tree_diff = from_tree.diff_stream(&to_tree, files_matcher);
    // TODO: Resolve values concurrently
    while let Some(entry) = tree_diff.next().await {
        let mut values = entry.values?;
        values.before = resolve_file_values(store, &entry.path, values.before).await?;
        if !values.is_changed() {
            continue;
        }
        let conflict_labels = ConflictLabels::unlabeled();
        let left_future =
            materialize_tree_value(store, &entry.path, values.before, &conflict_labels);
        let right_future =
            materialize_tree_value(store, &entry.path, values.after, &conflict_labels);
        let (left_value, right_value) = futures::try_join!(left_future, right_future)?;
        let left_contents = to_file_content(&entry.path, left_value).await?;
        let right_contents = to_file_content(&entry.path, right_value).await?;
        let merge_options = store.merge_options();
        if diff_match_lines(&left_contents, &right_contents, text_matcher, merge_options)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn diff_match_lines(
    lefts: &Merge<BString>,
    rights: &Merge<BString>,
    matcher: &StringMatcher,
    merge_options: &MergeOptions,
) -> BackendResult<bool> {
    // Filter lines prior to comparison. This might produce inferior hunks due
    // to lack of contexts, but is way faster than full diff.
    if let (Some(left), Some(right)) = (lefts.as_resolved(), rights.as_resolved()) {
        let left_lines = matcher.match_lines(left);
        let right_lines = matcher.match_lines(right);
        Ok(left_lines.ne(right_lines))
    } else {
        let lefts: Merge<BString> = lefts.map(|text| matcher.match_lines(text).collect());
        let rights: Merge<BString> = rights.map(|text| matcher.match_lines(text).collect());
        let lefts = files::merge(&lefts, merge_options);
        let rights = files::merge(&rights, merge_options);
        let diff = ContentDiff::by_line(itertools::chain(&lefts, &rights));
        let different = files::conflict_diff_hunks(diff.hunks(), lefts.as_slice().len())
            .any(|hunk| hunk.kind == DiffHunkKind::Different);
        Ok(different)
    }
}

async fn to_file_content(
    path: &RepoPath,
    value: MaterializedTreeValue,
) -> BackendResult<Merge<BString>> {
    let empty = || Merge::resolved(BString::default());
    match value {
        MaterializedTreeValue::Absent => Ok(empty()),
        MaterializedTreeValue::AccessDenied(_) => Ok(empty()),
        MaterializedTreeValue::File(mut file) => {
            Ok(Merge::resolved(file.read_all(path).await?.into()))
        }
        MaterializedTreeValue::Symlink { id: _, target } => Ok(Merge::resolved(target.into())),
        MaterializedTreeValue::GitSubmodule(_) => Ok(empty()),
        MaterializedTreeValue::FileConflict(file) => Ok(file.contents),
        MaterializedTreeValue::OtherConflict { .. } => Ok(empty()),
        MaterializedTreeValue::Tree(id) => {
            panic!("Unexpected tree with id {id:?} in diff at path {path:?}");
        }
    }
}

#[cfg(test)]
#[rustversion::attr(
    since(1.89),
    expect(clippy::cloned_ref_to_slice_refs, reason = "makes tests more readable")
)]
mod tests {
    use futures::FutureExt as _;
    use indoc::indoc;

    use super::*;
    use crate::default_index::DefaultMutableIndex;
    use crate::default_index::readonly::FieldLengths;
    use crate::files::FileMergeHunkLevel;
    use crate::merge::SameChange;
    use crate::str_util::StringPattern;

    const TEST_FIELD_LENGTHS: FieldLengths = FieldLengths {
        commit_id: 3,
        change_id: 16,
    };

    /// Generator of unique 16-byte ChangeId excluding root id
    fn change_id_generator() -> impl FnMut() -> ChangeId {
        let mut iter = (1_u128..).map(|n| ChangeId::new(n.to_le_bytes().into()));
        move || iter.next().unwrap()
    }

    async fn try_collect_stream_vec<T, E>(
        stream: impl Stream<Item = Result<T, E>>,
    ) -> Result<Vec<T>, E> {
        use futures::TryStreamExt;
        stream.try_collect().await
    }

    fn collect_stream_sync<T, E>(stream: impl Stream<Item = Result<T, E>>) -> Result<Vec<T>, E> {
        stream
            .collect::<Vec<_>>()
            .now_or_never()
            .expect("stream should not block")
            .into_iter()
            .collect()
    }

    async fn test_predicate<P: ToAsyncPredicateFn + ?Sized>(
        predicate: &P,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> bool {
        predicate.to_async_predicate_fn().matches(index, pos).await.unwrap()
    }

    async fn test_predicate_result<P: ToAsyncPredicateFn + ?Sized>(
        predicate: &P,
        index: &CompositeIndex,
        pos: GlobalCommitPosition,
    ) -> Result<bool, RevsetEvaluationError> {
        predicate.to_async_predicate_fn().matches(index, pos).await
    }

    #[test]
    fn test_revset_combinator() {
        pollster::block_on(async {
            let mut new_change_id = change_id_generator();
            let mut index = DefaultMutableIndex::full(TEST_FIELD_LENGTHS);
            let id_0 = CommitId::from_hex("000000");
            let id_1 = CommitId::from_hex("111111");
            let id_2 = CommitId::from_hex("222222");
            let id_3 = CommitId::from_hex("333333");
            let id_4 = CommitId::from_hex("444444");
            index.add_commit_data(id_0.clone(), new_change_id(), &[]);
            index.add_commit_data(id_1.clone(), new_change_id(), &[id_0.clone()]);
            index.add_commit_data(id_2.clone(), new_change_id(), &[id_1.clone()]);
            index.add_commit_data(id_3.clone(), new_change_id(), &[id_2.clone()]);
            index.add_commit_data(id_4.clone(), new_change_id(), &[id_3.clone()]);

            let index = index.as_composite();
            let get_pos = |id: &CommitId| index.commits().commit_id_to_pos(id).unwrap();
            let make_positions = |ids: &[&CommitId]| ids.iter().copied().map(get_pos).collect_vec();
            let make_set = |ids: &[&CommitId]| -> Box<dyn InternalRevset> {
                let positions = make_positions(ids);
                Box::new(EagerRevset { positions })
            };

            let set = make_set(&[&id_4, &id_3, &id_2, &id_0]);
            assert!(test_predicate(&*set, index, get_pos(&id_4)).await);
            assert!(test_predicate(&*set, index, get_pos(&id_3)).await);
            assert!(test_predicate(&*set, index, get_pos(&id_2)).await);
            assert!(!test_predicate(&*set, index, get_pos(&id_1)).await);
            assert!(test_predicate(&*set, index, get_pos(&id_0)).await);
            // Uninteresting entries can be skipped
            assert!(test_predicate(&*set, index, get_pos(&id_3)).await);
            assert!(!test_predicate(&*set, index, get_pos(&id_1)).await);
            assert!(test_predicate(&*set, index, get_pos(&id_0)).await);

            let id_4_clone = id_4.clone();
            let set = FilterRevset {
                candidates: make_set(&[&id_4, &id_2, &id_0]),
                predicate: PurePredicateFn(move |index: &CompositeIndex, pos| {
                    Ok(index.commits().entry_by_pos(pos).commit_id() != id_4_clone)
                }),
            };
            assert_eq!(
                collect_stream_sync(set.positions_stream(index)).unwrap(),
                make_positions(&[&id_2, &id_0])
            );
            assert!(!test_predicate(&set, index, get_pos(&id_4)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_3)).await);
            assert!(test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(test_predicate(&set, index, get_pos(&id_0)).await);

            // Intersection by FilterRevset
            let set = FilterRevset {
                candidates: make_set(&[&id_4, &id_2, &id_0]),
                predicate: make_set(&[&id_3, &id_2, &id_1]),
            };
            assert_eq!(
                collect_stream_sync(set.positions_stream(index)).unwrap(),
                make_positions(&[&id_2])
            );
            assert!(!test_predicate(&set, index, get_pos(&id_4)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_3)).await);
            assert!(test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_0)).await);

            let set = UnionRevset {
                set1: make_set(&[&id_4, &id_2]),
                set2: make_set(&[&id_3, &id_2, &id_1]),
            };
            assert_eq!(
                collect_stream_sync(set.positions_stream(index)).unwrap(),
                make_positions(&[&id_4, &id_3, &id_2, &id_1])
            );
            assert!(test_predicate(&set, index, get_pos(&id_4)).await);
            assert!(test_predicate(&set, index, get_pos(&id_3)).await);
            assert!(test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_0)).await);

            let set = IntersectionRevset {
                set1: make_set(&[&id_4, &id_2, &id_0]),
                set2: make_set(&[&id_3, &id_2, &id_1]),
            };
            assert_eq!(
                collect_stream_sync(set.positions_stream(index)).unwrap(),
                make_positions(&[&id_2])
            );
            assert!(!test_predicate(&set, index, get_pos(&id_4)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_3)).await);
            assert!(test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_0)).await);

            let set = DifferenceRevset {
                set1: make_set(&[&id_4, &id_2, &id_0]),
                set2: make_set(&[&id_3, &id_2, &id_1]),
            };
            assert_eq!(
                collect_stream_sync(set.positions_stream(index)).unwrap(),
                make_positions(&[&id_4, &id_0])
            );
            assert!(test_predicate(&set, index, get_pos(&id_4)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_3)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(test_predicate(&set, index, get_pos(&id_0)).await);
        });
    }

    #[test]
    fn test_revset_combinator_error_propagation() {
        pollster::block_on(async {
            let mut new_change_id = change_id_generator();
            let mut index = DefaultMutableIndex::full(TEST_FIELD_LENGTHS);
            let id_0 = CommitId::from_hex("000000");
            let id_1 = CommitId::from_hex("111111");
            let id_2 = CommitId::from_hex("222222");
            index.add_commit_data(id_0.clone(), new_change_id(), &[]);
            index.add_commit_data(id_1.clone(), new_change_id(), &[id_0.clone()]);
            index.add_commit_data(id_2.clone(), new_change_id(), &[id_1.clone()]);

            let index = index.as_composite();
            let get_pos = |id: &CommitId| index.commits().commit_id_to_pos(id).unwrap();
            let make_positions = |ids: &[&CommitId]| ids.iter().copied().map(get_pos).collect_vec();

            // Helper struct for a "bad" predicate that errors on a specific commit
            #[derive(Debug)]
            struct BadPredicate {
                bad_id: CommitId,
            }
            #[async_trait(?Send)]
            impl AsyncPredicate for BadPredicate {
                async fn matches(
                    &mut self,
                    index: &CompositeIndex,
                    pos: GlobalCommitPosition,
                ) -> Result<bool, RevsetEvaluationError> {
                    if index.commits().entry_by_pos(pos).commit_id() == self.bad_id {
                        Err(RevsetEvaluationError::Other("bad".into()))
                    } else {
                        Ok(true)
                    }
                }
            }
            #[derive(Debug, Clone)]
            struct BadPredicateFn {
                bad_id: CommitId,
            }
            impl ToAsyncPredicateFn for BadPredicateFn {
                fn to_async_predicate_fn(&self) -> Box<dyn AsyncPredicate + '_> {
                    Box::new(BadPredicate {
                        bad_id: self.bad_id.clone(),
                    })
                }
            }

            let make_good_set = |ids: &[&CommitId]| -> Box<dyn InternalRevset> {
                let positions = make_positions(ids);
                Box::new(EagerRevset { positions })
            };
            let make_bad_set = |ids: &[&CommitId], bad_id: &CommitId| -> Box<dyn InternalRevset> {
                let positions = make_positions(ids);
                Box::new(FilterRevset {
                    candidates: EagerRevset { positions },
                    predicate: BadPredicateFn {
                        bad_id: bad_id.clone(),
                    },
                })
            };

            // Helper to collect stream with a limit
            async fn collect_stream_take(
                mut stream: PositionsStream<'_>,
                n: usize,
            ) -> Result<Vec<GlobalCommitPosition>, RevsetEvaluationError> {
                let mut results = Vec::new();
                for _ in 0..n {
                    match stream.next().await {
                        Some(Ok(pos)) => results.push(pos),
                        Some(Err(e)) => return Err(e),
                        None => break,
                    }
                }
                Ok(results)
            }

            // Error from filter predicate
            let set = make_bad_set(&[&id_2, &id_1, &id_0], &id_1);
            assert_eq!(
                collect_stream_take(set.positions_stream(index), 1)
                    .await
                    .unwrap(),
                make_positions(&[&id_2])
            );
            assert!(collect_stream_take(set.positions_stream(index), 2).await.is_err());
            assert!(test_predicate(&*set, index, get_pos(&id_2)).await);
            assert!(!test_predicate_result(&*set, index, get_pos(&id_1)).await.is_ok());

            // Error from filter candidates
            let set = FilterRevset {
                candidates: make_bad_set(&[&id_2, &id_1, &id_0], &id_1),
                predicate: PurePredicateFn(|_: &CompositeIndex, _: GlobalCommitPosition| Ok(true)),
            };
            assert_eq!(
                collect_stream_take(set.positions_stream(index), 1)
                    .await
                    .unwrap(),
                make_positions(&[&id_2])
            );
            assert!(collect_stream_take(set.positions_stream(index), 2).await.is_err());
            assert!(test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate_result(&set, index, get_pos(&id_1)).await.is_ok());

            // Error from left side of union, immediately
            let set = UnionRevset {
                set1: make_bad_set(&[&id_1], &id_1),
                set2: make_good_set(&[&id_2, &id_1]),
            };
            assert!(collect_stream_take(set.positions_stream(index), 1).await.is_err());
            assert!(test_predicate(&set, index, get_pos(&id_2)).await); // works because bad id isn't visited
            assert!(!test_predicate_result(&set, index, get_pos(&id_1)).await.is_ok());

            // Error from right side of union, lazily
            let set = UnionRevset {
                set1: make_good_set(&[&id_2, &id_1]),
                set2: make_bad_set(&[&id_1, &id_0], &id_0),
            };
            assert_eq!(
                collect_stream_take(set.positions_stream(index), 2)
                    .await
                    .unwrap(),
                make_positions(&[&id_2, &id_1])
            );
            assert!(collect_stream_take(set.positions_stream(index), 3).await.is_err());
            assert!(test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(!test_predicate_result(&set, index, get_pos(&id_0)).await.is_ok());

            // Error from left side of intersection, immediately
            let set = IntersectionRevset {
                set1: make_bad_set(&[&id_1], &id_1),
                set2: make_good_set(&[&id_2, &id_1]),
            };
            assert!(collect_stream_take(set.positions_stream(index), 1).await.is_err());
            assert!(!test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate_result(&set, index, get_pos(&id_1)).await.is_ok());

            // Error from right side of intersection, lazily
            let set = IntersectionRevset {
                set1: make_good_set(&[&id_2, &id_1, &id_0]),
                set2: make_bad_set(&[&id_1, &id_0], &id_0),
            };
            assert_eq!(
                collect_stream_take(set.positions_stream(index), 1)
                    .await
                    .unwrap(),
                make_positions(&[&id_1])
            );
            assert!(collect_stream_take(set.positions_stream(index), 2).await.is_err());
            assert!(!test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(!test_predicate_result(&set, index, get_pos(&id_0)).await.is_ok());

            // Error from left side of difference, immediately
            let set = DifferenceRevset {
                set1: make_bad_set(&[&id_1], &id_1),
                set2: make_good_set(&[&id_2, &id_1]),
            };
            assert!(collect_stream_take(set.positions_stream(index), 1).await.is_err());
            assert!(!test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate_result(&set, index, get_pos(&id_1)).await.is_ok());

            // Error from right side of difference, lazily
            let set = DifferenceRevset {
                set1: make_good_set(&[&id_2, &id_1, &id_0]),
                set2: make_bad_set(&[&id_1, &id_0], &id_0),
            };
            assert_eq!(
                collect_stream_take(set.positions_stream(index), 1)
                    .await
                    .unwrap(),
                make_positions(&[&id_2])
            );
            assert!(collect_stream_take(set.positions_stream(index), 2).await.is_err());
            assert!(test_predicate(&set, index, get_pos(&id_2)).await);
            assert!(!test_predicate(&set, index, get_pos(&id_1)).await);
            assert!(!test_predicate_result(&set, index, get_pos(&id_0)).await.is_ok());
        });
    }

    #[test]
    fn test_positions_accumulator() {
        let mut new_change_id = change_id_generator();
        let mut index = DefaultMutableIndex::full(TEST_FIELD_LENGTHS);
        let id_0 = CommitId::from_hex("000000");
        let id_1 = CommitId::from_hex("111111");
        let id_2 = CommitId::from_hex("222222");
        let id_3 = CommitId::from_hex("333333");
        let id_4 = CommitId::from_hex("444444");
        index.add_commit_data(id_0.clone(), new_change_id(), &[]);
        index.add_commit_data(id_1.clone(), new_change_id(), &[id_0.clone()]);
        index.add_commit_data(id_2.clone(), new_change_id(), &[id_1.clone()]);
        index.add_commit_data(id_3.clone(), new_change_id(), &[id_2.clone()]);
        index.add_commit_data(id_4.clone(), new_change_id(), &[id_3.clone()]);

        let index = index.as_composite();
        let get_pos = |id: &CommitId| index.commits().commit_id_to_pos(id).unwrap();
        let make_positions = |ids: &[&CommitId]| ids.iter().copied().map(get_pos).collect_vec();

        // Helper to create a BoxedRevWalk from positions
        fn make_walk(positions: Vec<GlobalCommitPosition>) -> BoxedRevWalk<'static> {
            Box::new(EagerRevWalk::new(positions.into_iter().map(Ok)))
        }

        let full_positions = make_positions(&[&id_4, &id_3, &id_2, &id_1, &id_0]);

        // Consumes entries incrementally
        let positions_accum = PositionsAccumulator::new(index, make_walk(full_positions.clone()));

        assert!(positions_accum.contains(&id_3).unwrap());
        assert_eq!(positions_accum.consumed_len(), 2);

        assert!(positions_accum.contains(&id_0).unwrap());
        assert_eq!(positions_accum.consumed_len(), 5);

        assert!(positions_accum.contains(&id_3).unwrap());
        assert_eq!(positions_accum.consumed_len(), 5);

        // Does not consume positions for unknown commits
        let positions_accum = PositionsAccumulator::new(index, make_walk(full_positions.clone()));

        assert!(
            !positions_accum
                .contains(&CommitId::from_hex("999999"))
                .unwrap()
        );
        assert_eq!(positions_accum.consumed_len(), 0);

        // Does not consume without necessity
        let set_positions = make_positions(&[&id_3, &id_2, &id_1]);
        let positions_accum = PositionsAccumulator::new(index, make_walk(set_positions));

        assert!(!positions_accum.contains(&id_4).unwrap());
        assert_eq!(positions_accum.consumed_len(), 1);

        assert!(positions_accum.contains(&id_3).unwrap());
        assert_eq!(positions_accum.consumed_len(), 1);

        assert!(!positions_accum.contains(&id_0).unwrap());
        assert_eq!(positions_accum.consumed_len(), 3);

        assert!(positions_accum.contains(&id_1).unwrap());
    }

    fn diff_match_lines_samples() -> (Merge<BString>, Merge<BString>) {
        // left2      left1      base       right1      right2
        // ---------- ---------- ---------- ----------- -----------
        // "left 1.1" "line 1"   "line 1"   "line 1"    "line 1"
        // "line 2"   "line 2"   "line 2"   "line 2"    "line 2"
        // "left 3.1" "left 3.1" "line 3"   "right 3.1" "right 3.1"
        // "left 3.2" "left 3.2"
        // "left 3.3"
        // "line 4"   "line 4"   "line 4"   "line 4"    "line 4"
        // "line 5"   "line 5"              "line 5"
        let base = indoc! {"
            line 1
            line 2
            line 3
            line 4
        "};
        let left1 = indoc! {"
            line 1
            line 2
            left 3.1
            left 3.2
            line 4
            line 5
        "};
        let left2 = indoc! {"
            left 1.1
            line 2
            left 3.1
            left 3.2
            left 3.3
            line 4
            line 5
        "};
        let right1 = indoc! {"
            line 1
            line 2
            right 3.1
            line 4
            line 5
        "};
        let right2 = indoc! {"
            line 1
            line 2
            right 3.1
            line 4
        "};

        let conflict1 = Merge::from_vec([left1, base, right1].map(BString::from).to_vec());
        let conflict2 = Merge::from_vec([left2, base, right2].map(BString::from).to_vec());
        (conflict1, conflict2)
    }

    #[test]
    fn test_diff_match_lines_between_resolved() {
        let (conflict1, conflict2) = diff_match_lines_samples();
        let left1 = Merge::resolved(conflict1.first().clone());
        let left2 = Merge::resolved(conflict2.first().clone());
        let diff = |needle: &str| {
            let matcher = StringPattern::substring(needle).to_matcher();
            let options = MergeOptions {
                hunk_level: FileMergeHunkLevel::Line,
                same_change: SameChange::Accept,
            };
            diff_match_lines(&left1, &left2, &matcher, &options).unwrap()
        };

        assert!(diff(""));
        assert!(!diff("no match"));
        assert!(diff("line "));
        assert!(diff(" 1"));
        assert!(!diff(" 2"));
        assert!(diff(" 3"));
        assert!(!diff(" 3.1"));
        assert!(!diff(" 3.2"));
        assert!(diff(" 3.3"));
        assert!(!diff(" 4"));
        assert!(!diff(" 5"));
    }

    #[test]
    fn test_diff_match_lines_between_conflicts() {
        let (conflict1, conflict2) = diff_match_lines_samples();
        let diff = |needle: &str| {
            let matcher = StringPattern::substring(needle).to_matcher();
            let options = MergeOptions {
                hunk_level: FileMergeHunkLevel::Line,
                same_change: SameChange::Accept,
            };
            diff_match_lines(&conflict1, &conflict2, &matcher, &options).unwrap()
        };

        assert!(diff(""));
        assert!(!diff("no match"));
        assert!(diff("line "));
        assert!(diff(" 1"));
        assert!(!diff(" 2"));
        assert!(diff(" 3"));
        // " 3.1" and " 3.2" could be considered different because the hunk
        // includes a changed line " 3.3". However, we filters out unmatched
        // lines first, therefore the changed line is omitted from the hunk.
        assert!(!diff(" 3.1"));
        assert!(!diff(" 3.2"));
        assert!(diff(" 3.3"));
        assert!(!diff(" 4"));
        assert!(!diff(" 5")); // per A-B+A=A rule
    }

    #[test]
    fn test_diff_match_lines_between_resolved_and_conflict() {
        let (_conflict1, conflict2) = diff_match_lines_samples();
        let base = Merge::resolved(conflict2.get_remove(0).unwrap().clone());
        let diff = |needle: &str| {
            let matcher = StringPattern::substring(needle).to_matcher();
            let options = MergeOptions {
                hunk_level: FileMergeHunkLevel::Line,
                same_change: SameChange::Accept,
            };
            diff_match_lines(&base, &conflict2, &matcher, &options).unwrap()
        };

        assert!(diff(""));
        assert!(!diff("no match"));
        assert!(diff("line "));
        assert!(diff(" 1"));
        assert!(!diff(" 2"));
        assert!(diff(" 3"));
        assert!(diff(" 3.1"));
        assert!(diff(" 3.2"));
        assert!(!diff(" 4"));
        assert!(diff(" 5"));
    }
}
