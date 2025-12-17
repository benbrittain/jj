// Copyright 2020 The Jujutsu Authors
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

#![expect(missing_docs)]

use std::fmt::Debug;
use std::fmt::Error;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::pin::Pin;
use std::sync::Arc;
use std::task;

use futures::Stream;
use itertools::Itertools as _;

use crate::backend;
use crate::backend::BackendResult;
use crate::backend::TreeEntriesNonRecursiveIterator;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::matchers::Matcher;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::repo_path::RepoPathComponent;
use crate::store::Store;

#[derive(Clone)]
pub struct Tree {
    store: Arc<Store>,
    dir: RepoPathBuf,
    id: TreeId,
    data: Arc<backend::Tree>,
}

impl Debug for Tree {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("Tree")
            .field("dir", &self.dir)
            .field("id", &self.id)
            .finish()
    }
}

impl PartialEq for Tree {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dir == other.dir
    }
}

impl Eq for Tree {}

impl Hash for Tree {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dir.hash(state);
        self.id.hash(state);
    }
}

impl Tree {
    pub fn new(store: Arc<Store>, dir: RepoPathBuf, id: TreeId, data: Arc<backend::Tree>) -> Self {
        Self {
            store,
            dir,
            id,
            data,
        }
    }

    pub fn empty(store: Arc<Store>, dir: RepoPathBuf) -> Self {
        let id = store.empty_tree_id().clone();
        Self {
            store,
            dir,
            id,
            data: Arc::new(backend::Tree::default()),
        }
    }

    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    pub fn dir(&self) -> &RepoPath {
        &self.dir
    }

    pub fn id(&self) -> &TreeId {
        &self.id
    }

    pub fn data(&self) -> &backend::Tree {
        &self.data
    }

    pub fn entries_non_recursive(&self) -> TreeEntriesNonRecursiveIterator<'_> {
        self.data.entries()
    }

    pub fn entries_matching<'matcher>(
        &self,
        matcher: &'matcher dyn Matcher,
    ) -> TreeEntriesIterator<'matcher> {
        TreeEntriesIterator::new(self.clone(), matcher)
    }

    pub fn value(&self, basename: &RepoPathComponent) -> Option<&TreeValue> {
        self.data.value(basename)
    }

    pub async fn path_value(&self, path: &RepoPath) -> BackendResult<Option<TreeValue>> {
        assert_eq!(self.dir(), RepoPath::root());
        match path.split() {
            Some((dir, basename)) => {
                let tree = self.sub_tree_recursive(dir).await?;
                Ok(tree.and_then(|tree| tree.data.value(basename).cloned()))
            }
            None => Ok(Some(TreeValue::Tree(self.id.clone()))),
        }
    }

    pub async fn sub_tree(&self, name: &RepoPathComponent) -> BackendResult<Option<Self>> {
        if let Some(sub_tree) = self.data.value(name) {
            match sub_tree {
                TreeValue::Tree(sub_tree_id) => {
                    let subdir = self.dir.join(name);
                    let sub_tree = self.store.get_tree_async(subdir, sub_tree_id).await?;
                    Ok(Some(sub_tree))
                }
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    async fn known_sub_tree(&self, subdir: RepoPathBuf, id: &TreeId) -> Self {
        self.store.get_tree_async(subdir, id).await.unwrap()
    }

    /// Look up the tree at the given path.
    pub async fn sub_tree_recursive(&self, path: &RepoPath) -> BackendResult<Option<Self>> {
        let mut current_tree = self.clone();
        for name in path.components() {
            match current_tree.sub_tree(name).await? {
                None => {
                    return Ok(None);
                }
                Some(sub_tree) => {
                    current_tree = sub_tree;
                }
            }
        }
        // TODO: It would be nice to be able to return a reference here, but
        // then we would have to figure out how to share Tree instances
        // across threads.
        Ok(Some(current_tree))
    }
}

pub struct TreeEntriesIterator<'matcher> {
    stack: Vec<TreeEntriesDirItem>,
    matcher: &'matcher dyn Matcher,
    pending_subtree: Option<Pin<Box<dyn Future<Output = Tree> + Send>>>,
}

struct TreeEntriesDirItem {
    tree: Tree,
    entries: Vec<(RepoPathBuf, TreeValue)>,
}

impl From<Tree> for TreeEntriesDirItem {
    fn from(tree: Tree) -> Self {
        let mut entries = tree
            .entries_non_recursive()
            .map(|entry| (tree.dir().join(entry.name()), entry.value().clone()))
            .collect_vec();
        entries.reverse();
        Self { tree, entries }
    }
}

impl<'matcher> TreeEntriesIterator<'matcher> {
    fn new(tree: Tree, matcher: &'matcher dyn Matcher) -> Self {
        // TODO: Restrict walk according to Matcher::visit()
        Self {
            stack: vec![TreeEntriesDirItem::from(tree)],
            matcher,
            pending_subtree: None,
        }
    }
}

impl Stream for TreeEntriesIterator<'_> {
    type Item = (RepoPathBuf, TreeValue);

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut task::Context<'_>,
    ) -> task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(fut) = &mut this.pending_subtree {
                match fut.as_mut().poll(cx) {
                    task::Poll::Ready(subtree) => {
                        this.pending_subtree = None;
                        this.stack.push(TreeEntriesDirItem::from(subtree));
                    }
                    task::Poll::Pending => return task::Poll::Pending,
                }
            }

            let Some(top) = this.stack.last_mut() else {
                return task::Poll::Ready(None);
            };

            let Some((path, value)) = top.entries.pop() else {
                this.stack.pop();
                continue;
            };

            match value {
                TreeValue::Tree(id) => {
                    if this.matcher.visit(&path).is_nothing() {
                        continue;
                    }
                    let store = top.tree.store().clone();
                    this.pending_subtree = Some(Box::pin(async move {
                        store.get_tree_async(path, &id).await.unwrap()
                    }));
                }
                value => {
                    if this.matcher.matches(&path) {
                        return task::Poll::Ready(Some((path, value)));
                    }
                }
            }
        }
    }
}
