#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: materialize-paper-assets.sh --paper PAPER --paperbench-root PATH

Materialize the Git LFS files for one PaperBench paper when paper.md is still
an LFS pointer. Existing materialized files are left untouched.
EOF
}

paper=""
paperbench_root=""
while (($#)); do
  case "$1" in
    --paper)
      (($# >= 2)) || {
        printf '%s\n' '--paper requires a value.' >&2
        exit 2
      }
      paper="$2"
      shift 2
      ;;
    --paperbench-root)
      (($# >= 2)) || {
        printf '%s\n' '--paperbench-root requires a path.' >&2
        exit 2
      }
      paperbench_root="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$paper" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'Invalid PaperBench paper ID: %s\n' "$paper" >&2
  exit 2
fi
if [[ -z "$paperbench_root" || ! -d "$paperbench_root" ]]; then
  printf 'PaperBench root is unavailable: %s\n' "$paperbench_root" >&2
  exit 1
fi

paper_md="$paperbench_root/data/papers/$paper/paper.md"
if [[ ! -f "$paper_md" ]]; then
  printf 'PaperBench paper is unavailable: %s\n' "$paper_md" >&2
  exit 1
fi
if ! grep -Fqx 'version https://git-lfs.github.com/spec/v1' \
  <(head -n 1 "$paper_md"); then
  exit 0
fi

git_lfs_bin="$(command -v git-lfs 2>/dev/null || true)"
if [[ -z "$git_lfs_bin" && -x "$HOME/.local/bin/git-lfs" ]]; then
  git_lfs_bin="$HOME/.local/bin/git-lfs"
fi
if [[ -z "$git_lfs_bin" || ! -x "$git_lfs_bin" ]]; then
  printf '%s\n' \
    'git-lfs is required to materialize the selected PaperBench paper.' >&2
  exit 1
fi

repo_root="$(git -C "$paperbench_root" rev-parse --show-toplevel)"
paperbench_prefix="$(git -C "$paperbench_root" rev-parse --show-prefix)"
include_path="${paperbench_prefix}data/papers/$paper/**"
printf 'Materializing Git LFS assets for %s (%s).\n' "$paper" "$include_path"
PATH="$(dirname -- "$git_lfs_bin"):$PATH" \
  git -C "$repo_root" -c lfs.fetchexclude= lfs pull \
    --include="$include_path"

if grep -Fqx 'version https://git-lfs.github.com/spec/v1' \
  <(head -n 1 "$paper_md"); then
  printf 'Git LFS materialization did not replace %s.\n' "$paper_md" >&2
  exit 1
fi
if (($(wc -l < "$paper_md") <= 5)); then
  printf 'Materialized paper is unexpectedly short: %s\n' "$paper_md" >&2
  exit 1
fi
