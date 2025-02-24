// Copyright 2024 The Jujutsu Authors
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

use jj_lib::secret_backend::SecretBackend;

use crate::common::TestEnvironment;

#[test]
fn test_diff() {
    let test_env = TestEnvironment::default();
    test_env
        .run_jj_in(test_env.env_root(), ["git", "init", "repo"])
        .success();
    let repo_path = test_env.env_root().join("repo");

    std::fs::create_dir(repo_path.join("dir")).unwrap();
    std::fs::write(repo_path.join("a-first"), "foo\n").unwrap();
    std::fs::write(repo_path.join("deleted-secret"), "foo\n").unwrap();
    std::fs::write(repo_path.join("dir").join("secret"), "foo\n").unwrap();
    std::fs::write(repo_path.join("modified-secret"), "foo\n").unwrap();
    std::fs::write(repo_path.join("z-last"), "foo\n").unwrap();
    test_env.run_jj_in(&repo_path, ["new"]).success();
    std::fs::write(repo_path.join("a-first"), "bar\n").unwrap();
    std::fs::remove_file(repo_path.join("deleted-secret")).unwrap();
    std::fs::write(repo_path.join("added-secret"), "bar\n").unwrap();
    std::fs::write(repo_path.join("dir").join("secret"), "bar\n").unwrap();
    std::fs::write(repo_path.join("modified-secret"), "bar\n").unwrap();
    std::fs::write(repo_path.join("z-last"), "bar\n").unwrap();

    SecretBackend::adopt_git_repo(&repo_path);

    let output = test_env.run_jj_in(&repo_path, ["diff", "--color-words"]);
    insta::assert_snapshot!(output.normalize_backslash(), @r"
    Modified regular file a-first:
       1    1: foobar
    Access denied to added-secret: No access
    Access denied to deleted-secret: No access
    Access denied to dir/secret: No access
    Access denied to modified-secret: No access
    Modified regular file z-last:
       1    1: foobar
    [EOF]
    ");
    let output = test_env.run_jj_in(&repo_path, ["diff", "--summary"]);
    insta::assert_snapshot!(output.normalize_backslash(), @r"
    M a-first
    C {a-first => added-secret}
    D deleted-secret
    M dir/secret
    M modified-secret
    M z-last
    [EOF]
    ");
    let output = test_env.run_jj_in(&repo_path, ["diff", "--types"]);
    insta::assert_snapshot!(output.normalize_backslash(), @r"
    FF a-first
    FF {a-first => added-secret}
    F- deleted-secret
    FF dir/secret
    FF modified-secret
    FF z-last
    [EOF]
    ");
    let output = test_env.run_jj_in(&repo_path, ["diff", "--stat"]);
    insta::assert_snapshot!(output.normalize_backslash(), @r"
    a-first                   | 2 +-
    {a-first => added-secret} | 2 +-
    deleted-secret            | 1 -
    dir/secret                | 0
    modified-secret           | 0
    z-last                    | 2 +-
    6 files changed, 3 insertions(+), 4 deletions(-)
    [EOF]
    ");
    let output = test_env.run_jj_in(&repo_path, ["diff", "--git"]);
    insta::assert_snapshot!(output.normalize_backslash(), @r"
    diff --git a/a-first b/a-first
    index 257cc5642c..5716ca5987 100644
    --- a/a-first
    +++ b/a-first
    @@ -1,1 +1,1 @@
    -foo
    +bar
    [EOF]
    ------- stderr -------
    Error: Access denied to added-secret
    Caused by: No access
    [EOF]
    [exit status: 1]
    ");

    // TODO: Test external tool
}

#[test]
fn test_file_list_show() {
    let test_env = TestEnvironment::default();
    test_env
        .run_jj_in(test_env.env_root(), ["git", "init", "repo"])
        .success();
    let repo_path = test_env.env_root().join("repo");

    std::fs::write(repo_path.join("a-first"), "foo\n").unwrap();
    std::fs::write(repo_path.join("secret"), "bar\n").unwrap();
    std::fs::write(repo_path.join("z-last"), "baz\n").unwrap();

    SecretBackend::adopt_git_repo(&repo_path);

    // "file list" should just work since it doesn't access file content
    let output = test_env.run_jj_in(&repo_path, ["file", "list"]);
    insta::assert_snapshot!(output, @r"
    a-first
    secret
    z-last
    [EOF]
    ");

    let output = test_env.run_jj_in(&repo_path, ["file", "show", "."]);
    insta::assert_snapshot!(output.normalize_backslash(), @r"
    foo
    baz
    [EOF]
    ------- stderr -------
    Warning: Path 'secret' exists but access is denied: No access
    [EOF]
    ");
}
