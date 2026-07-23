#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-bbox-roemia.sh [--dry-run] [--requirements-csv PATH] [-- CHZ_ARG...]

Run the full BBox-Adapter PaperBench task on roemia. Docker images remain in
Docker's data root; run artifacts, package/model caches, and container temporary
files are stored below /data/$USER/paperbench by default.

Environment overrides:
  PAPERBENCH_DATA_ROOT   Host data directory (default: /data/$USER/paperbench)
  CODEX_AUTH_FILE       Host Codex auth file (default: $HOME/.codex/auth.json)
  PAPERBENCH_AGENT_ENV  PaperBench agent.env used by the checkout
EOF
}

dry_run=false
requirements_csv=""
extra_args=()
while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    --requirements-csv)
      if (($# < 2)); then
        printf '%s\n' '--requirements-csv requires a path.' >&2
        exit 2
      fi
      requirements_csv="$2"
      shift 2
      ;;
    --)
      shift
      extra_args+=("$@")
      break
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paperbench_root="$(cd -- "$script_dir/../.." && pwd)"
data_root="${PAPERBENCH_DATA_ROOT:-/data/$USER/paperbench}"
auth_file="${CODEX_AUTH_FILE:-$HOME/.codex/auth.json}"
agent_env="${PAPERBENCH_AGENT_ENV:-$paperbench_root/paperbench/solvers/agent.env}"
cache_dir="$data_root/cache"
runs_dir="$data_root/runs"
agent_tmp_dir="$data_root/tmp/agent"
reproduction_tmp_dir="$data_root/tmp/reproduction"

if [[ ! -f "$auth_file" ]]; then
  printf 'Codex auth file does not exist: %s\n' "$auth_file" >&2
  exit 1
fi
if [[ ! -f "$agent_env" ]]; then
  printf 'PaperBench agent.env does not exist: %s\n' "$agent_env" >&2
  exit 1
fi
if [[ -n "$requirements_csv" ]]; then
  if [[ ! -f "$requirements_csv" ]]; then
    printf 'Requirements CSV does not exist: %s\n' "$requirements_csv" >&2
    exit 1
  fi
  requirements_csv="$(cd -- "$(dirname -- "$requirements_csv")" && pwd)/$(basename -- "$requirements_csv")"
fi

command=(
  uv run python -m paperbench.nano.entrypoint
  paperbench.paper_split=bbox-poc
  paperbench.runs_dir="$runs_dir"
  paperbench.docker_image=pb-codex-env:latest
  paperbench.solver=paperbench.solvers.codex.solver:CodexSolver
  paperbench.solver.codex_auth_file="$auth_file"
  paperbench.solver.model=gpt-5.6-sol
  paperbench.solver.reasoning_effort=high
  paperbench.solver.reasoning_summary=detailed
  paperbench.solver.time_limit=86400
  paperbench.solver.upload_interval_seconds=1800
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig
  paperbench.solver.computer_runtime.env.pull_from_registry=false
  paperbench.solver.computer_runtime.env.is_nvidia_gpu_env=true
  paperbench.solver.computer_runtime.env.volumes_config.paperbench_cache.bind_source="$cache_dir"
  paperbench.solver.computer_runtime.env.volumes_config.paperbench_cache.bind_dest=/root/.cache
  paperbench.solver.computer_runtime.env.volumes_config.paperbench_cache.mode=rw
  paperbench.solver.computer_runtime.env.volumes_config.paperbench_tmp.bind_source="$agent_tmp_dir"
  paperbench.solver.computer_runtime.env.volumes_config.paperbench_tmp.bind_dest=/tmp
  paperbench.solver.computer_runtime.env.volumes_config.paperbench_tmp.mode=rw
  paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false
  paperbench.reproduction.computer_runtime.env.is_nvidia_gpu_env=true
  paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_cache.bind_source="$cache_dir"
  paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_cache.bind_dest=/root/.cache
  paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_cache.mode=rw
  paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_tmp.bind_source="$reproduction_tmp_dir"
  paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_tmp.bind_dest=/tmp
  paperbench.reproduction.computer_runtime.env.volumes_config.paperbench_tmp.mode=rw
  paperbench.reproduction.skip_reproduction=False
  paperbench.reproduction.timeout=86400
  paperbench.judge.code_only=False
  paperbench.judge.grade=True
  paperbench.judge.grade_locally=True
  paperbench.judge.completer_config=paperbench.judge.codex_cli_turn_completer:CodexCliTurnCompleter.Config
  paperbench.judge.completer_config.model=gpt-5.4
  paperbench.judge.completer_config.codex_home="$HOME/.codex"
  paperbench.judge.completer_config.reasoning_effort=medium
  paperbench.judge.completer_config.max_concurrency=4
  runner.max_retries=0
  runner.concurrency=1
  runner.recorder=nanoeval.json_recorder:json_recorder
)

if [[ -n "$requirements_csv" ]]; then
  command+=(paperbench.requirements_csv="$requirements_csv")
fi
if ((${#extra_args[@]})); then
  command+=("${extra_args[@]}")
fi

if $dry_run; then
  printf 'cd %q\n' "$paperbench_root"
  printf 'UV_CACHE_DIR=%q ' "$cache_dir/host-uv"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ "$data_root" != /data/* ]]; then
  printf 'Refusing to run: PAPERBENCH_DATA_ROOT must be under /data on roemia: %s\n' \
    "$data_root" >&2
  exit 1
fi
if [[ ! -d "$(dirname -- "$data_root")" || ! -w "$(dirname -- "$data_root")" ]]; then
  printf 'Data parent is unavailable or not writable: %s\n' "$(dirname -- "$data_root")" >&2
  exit 1
fi

mkdir -p "$cache_dir/host-uv" "$runs_dir" "$agent_tmp_dir" "$reproduction_tmp_dir"
chmod 700 "$data_root" "$cache_dir" "$runs_dir"
chmod 1777 "$agent_tmp_dir" "$reproduction_tmp_dir"

export UV_CACHE_DIR="$cache_dir/host-uv"
cd "$paperbench_root"
exec "${command[@]}"
