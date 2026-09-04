#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-paperbench-roemia.sh --paper PAPER [--dry-run] [--standalone]
                                 [--gpu ID_OR_UUID] [--requirements-csv PATH]
                                 [-- CHZ_ARG...]

Low-level launcher for one PaperBench PoC task on a GPU server. Normal end-to-end
runs must use ReproGapBench's run_paperbench_with_qwen_judge.sh. Pass
--standalone explicitly only when intentionally skipping that Qwen chain.

PAPER must be an alias listed in experiments/paper-aliases.tsv. The launcher
resolves the alias to a PaperBench paper ID and selects that paper directly;
per-paper split files are not required.

Docker images remain in Docker's data root; run artifacts, package/model caches,
and container temporary files are stored below the configured data mount.

Environment overrides:
  PAPERBENCH_DATA_BASE   Host data mount (default: /data)
  PAPERBENCH_DATA_ROOT   Host data directory below PAPERBENCH_DATA_BASE
  PAPERBENCH_GPU_DEVICE  GPU index or UUID (default: select one idle GPU)
  PAPERBENCH_AGENT_GPU_NAME_PATTERN
                         Restrict automatic selection to matching GPU names
  PAPERBENCH_AGENT_PREFERRED_GPU_NAME_PATTERN
                         Prefer matching GPU names, then fall back to other
                         eligible GPUs
  PAPERBENCH_AGENT_MIN_GPU_MEMORY_MIB
                         Minimum total GPU memory for automatic selection
                         (default: 0)
  PAPERBENCH_WORKLOAD_EXCLUDED_GPU_INDICES
                         Comma-separated GPU indices unavailable to the Agent
                         and Reproduction workloads
  PAPERBENCH_AGENT_GPU_WAIT_TIMEOUT_SECONDS
                         GPU wait timeout (default: 21600 / 6 hours)
  PAPERBENCH_AGENT_GPU_POLL_INTERVAL_SECONDS
                         GPU recheck interval (default: 60 / 1 minute)
  REPROGAP_GPU_ACQUIRE_MODE
                         wait (default) or fail-fast for queue dispatch
  PAPERBENCH_NVIDIA_USE_VISIBLE_DEVICES
                         auto (default), 1, or 0. Auto uses the legacy
                         NVIDIA_VISIBLE_DEVICES path when the host persistence
                         socket required by its CDI spec is absent.
  CODEX_AUTH_FILE       Host Codex auth source copied into an isolated per-run
                        directory (default: $HOME/.codex/auth.json)
  PAPERBENCH_AGENT_ENV  PaperBench agent.env used by the checkout
EOF
}

dry_run=false
launch_mode=""
requirements_csv=""
requested_gpu="${PAPERBENCH_GPU_DEVICE:-}"
gpu_name_pattern="${PAPERBENCH_AGENT_GPU_NAME_PATTERN:-}"
preferred_gpu_name_pattern="${PAPERBENCH_AGENT_PREFERRED_GPU_NAME_PATTERN:-}"
minimum_gpu_memory_mib="${PAPERBENCH_AGENT_MIN_GPU_MEMORY_MIB:-0}"
excluded_gpu_indices="${PAPERBENCH_WORKLOAD_EXCLUDED_GPU_INDICES:-}"
gpu_wait_timeout="${PAPERBENCH_AGENT_GPU_WAIT_TIMEOUT_SECONDS:-21600}"
gpu_poll_interval="${PAPERBENCH_AGENT_GPU_POLL_INTERVAL_SECONDS:-60}"
gpu_acquire_mode="${REPROGAP_GPU_ACQUIRE_MODE:-wait}"
nvidia_visible_devices_mode="${PAPERBENCH_NVIDIA_USE_VISIBLE_DEVICES:-auto}"
paper=""
extra_args=()
while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    --qwen-chain)
      launch_mode="qwen-chain"
      shift
      ;;
    --standalone)
      launch_mode="standalone"
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
    --paper)
      if (($# < 2)); then
        printf '%s\n' '--paper requires a PaperBench paper ID.' >&2
        exit 2
      fi
      paper="$2"
      shift 2
      ;;
    --gpu)
      if (($# < 2)); then
        printf '%s\n' '--gpu requires a GPU index or UUID.' >&2
        exit 2
      fi
      requested_gpu="$2"
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

if [[ -z "$paper" ]]; then
  printf '%s\n' '--paper is required.' >&2
  usage >&2
  exit 2
fi

solver_override=""
if ((${#extra_args[@]})); then
  for extra_arg in "${extra_args[@]}"; do
    case "$extra_arg" in
      paperbench.solver=*)
        solver_override="${extra_arg#paperbench.solver=}"
        ;;
    esac
  done
fi
use_codex_solver_defaults=false
if [[ -z "$solver_override" || \
  "$solver_override" == "paperbench.solvers.codex.solver:CodexSolver" ]]; then
  use_codex_solver_defaults=true
fi
if [[ ! "$gpu_wait_timeout" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid PAPERBENCH_AGENT_GPU_WAIT_TIMEOUT_SECONDS: %s\n' \
    "$gpu_wait_timeout" >&2
  exit 2
fi
if [[ ! "$gpu_poll_interval" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid PAPERBENCH_AGENT_GPU_POLL_INTERVAL_SECONDS: %s\n' \
    "$gpu_poll_interval" >&2
  exit 2
fi
case "$gpu_acquire_mode" in
  wait|fail-fast) ;;
  *)
    printf 'Invalid REPROGAP_GPU_ACQUIRE_MODE: %s\n' "$gpu_acquire_mode" >&2
    exit 2
    ;;
esac
case "$nvidia_visible_devices_mode" in
  auto|0|1|false|true) ;;
  *)
    printf 'Invalid PAPERBENCH_NVIDIA_USE_VISIBLE_DEVICES: %s\n' \
      "$nvidia_visible_devices_mode" >&2
    exit 2
    ;;
esac
if [[ ! "$minimum_gpu_memory_mib" =~ ^[0-9]+$ ]]; then
  printf 'Invalid PAPERBENCH_AGENT_MIN_GPU_MEMORY_MIB: %s\n' \
    "$minimum_gpu_memory_mib" >&2
  exit 2
fi
if [[ -n "$excluded_gpu_indices" && \
  ! "$excluded_gpu_indices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'Invalid PAPERBENCH_WORKLOAD_EXCLUDED_GPU_INDICES: %s\n' \
    "$excluded_gpu_indices" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paperbench_root="$(cd -- "$script_dir/../.." && pwd)"
paper_registry="$paperbench_root/experiments/paper-aliases.tsv"
if [[ ! -f "$paper_registry" ]]; then
  printf 'Paper alias registry does not exist: %s\n' "$paper_registry" >&2
  exit 1
fi
paper_id=""
while IFS=$'\t' read -r registered_alias registered_paper_id extra; do
  [[ -n "$registered_alias" && "${registered_alias:0:1}" != "#" ]] || continue
  if [[ -z "$registered_paper_id" || -n "${extra:-}" ]]; then
    printf 'Invalid paper alias registry row for alias %s\n' "$registered_alias" >&2
    exit 1
  fi
  [[ "$registered_alias" == "$paper" ]] || continue
  if [[ -n "$paper_id" ]]; then
    printf 'Duplicate paper alias in registry: %s\n' "$paper" >&2
    exit 1
  fi
  paper_id="$registered_paper_id"
done < "$paper_registry"
if [[ -z "$paper_id" ]]; then
  printf 'Unsupported paper alias: %s\n' "$paper" >&2
  usage >&2
  exit 2
fi

if ! $dry_run && [[ -z "$launch_mode" ]]; then
  printf '%s\n' \
    'Use run_paperbench_with_qwen_judge.sh by default. Pass --standalone only for an intentionally unchained PaperBench run.' \
    >&2
  exit 2
fi

lfs_materializer="$script_dir/materialize-paper-assets.sh"
data_base="${PAPERBENCH_DATA_BASE:-/data}"
data_root="${PAPERBENCH_DATA_ROOT:-$data_base/$USER/paperbench}"
auth_file="${CODEX_AUTH_FILE:-$HOME/.codex/auth.json}"
agent_env="${PAPERBENCH_AGENT_ENV:-$paperbench_root/paperbench/solvers/agent.env}"
cache_dir="$data_root/cache"
runs_dir="$data_root/runs"
agent_tmp_dir="$data_root/tmp/agent"
reproduction_tmp_dir="$data_root/tmp/reproduction"
auth_runtime_root="$data_root/tmp/codex-auth"
runtime_codex_home="$auth_runtime_root/dry-run"
runtime_auth_file="$runtime_codex_home/auth.json"
gpu_lock_dir="$data_root/gpu-locks"

if [[ "$data_base" != /* || "$data_base" == / ]]; then
  printf 'PAPERBENCH_DATA_BASE must be an absolute non-root path: %s\n' \
    "$data_base" >&2
  exit 1
fi

if ! $dry_run; then
  if [[ ! -f "$auth_file" ]]; then
    printf 'Codex auth file does not exist: %s\n' "$auth_file" >&2
    exit 1
  fi
  if [[ ! -f "$agent_env" ]]; then
    printf 'PaperBench agent.env does not exist: %s\n' "$agent_env" >&2
    exit 1
  fi
fi
if [[ -n "$requirements_csv" ]]; then
  if [[ ! -f "$requirements_csv" ]]; then
    printf 'Requirements CSV does not exist: %s\n' "$requirements_csv" >&2
    exit 1
  fi
  requirements_csv="$(cd -- "$(dirname -- "$requirements_csv")" && pwd)/$(basename -- "$requirements_csv")"
fi
if [[ ! -x "$lfs_materializer" ]]; then
  printf 'Paper asset materializer is unavailable: %s\n' "$lfs_materializer" >&2
  exit 1
fi
if ! $dry_run; then
  "$lfs_materializer" \
    --paper "$paper" \
    --paperbench-root "$paperbench_root"
fi

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

gpu_device_id=""
gpu_index=""
gpu_free_memory=""
gpu_lock_fd=""

gpu_index_is_excluded() {
  local candidate_index="$1"
  [[ -n "$excluded_gpu_indices" && ",${excluded_gpu_indices}," == *",${candidate_index},"* ]]
}

select_gpu() {
  local acquire_lock="$1"
  local inventory
  local active_gpu_uuids
  local index
  local uuid
  local name
  local total_memory
  local free_memory
  local preference_rank
  local requested_gpu_found=false

  gpu_device_id=""
  gpu_index=""
  gpu_free_memory=""

  inventory="$(
    nvidia-smi \
      --query-gpu=index,uuid,name,memory.total,memory.free \
      --format=csv,noheader,nounits
  )"
  active_gpu_uuids="$(
    nvidia-smi \
      --query-compute-apps=gpu_uuid \
      --format=csv,noheader 2>/dev/null || true
  )"

  while IFS=',' read -r preference_rank free_memory index uuid; do
    preference_rank="$(trim_whitespace "$preference_rank")"
    free_memory="$(trim_whitespace "$free_memory")"
    index="$(trim_whitespace "$index")"
    uuid="$(trim_whitespace "$uuid")"

    if [[ -n "$requested_gpu" && "$requested_gpu" != "$index" && "$requested_gpu" != "$uuid" ]]; then
      continue
    fi
    requested_gpu_found=true
    if grep -Fxq "$uuid" <<<"$active_gpu_uuids"; then
      continue
    fi

    if [[ "$acquire_lock" == true ]]; then
      exec 9>"$gpu_lock_dir/$uuid.lock"
      if ! flock -n 9; then
        exec 9>&-
        continue
      fi
      gpu_lock_fd=9
    fi

    gpu_device_id="$uuid"
    gpu_index="$index"
    gpu_free_memory="$free_memory"
    break
  done < <(
    while IFS=',' read -r index uuid name total_memory free_memory; do
      index="$(trim_whitespace "$index")"
      uuid="$(trim_whitespace "$uuid")"
      name="$(trim_whitespace "$name")"
      total_memory="$(trim_whitespace "$total_memory")"
      free_memory="$(trim_whitespace "$free_memory")"
      if gpu_index_is_excluded "$index"; then
        continue
      fi
      if [[ -n "$gpu_name_pattern" && "$name" != *"$gpu_name_pattern"* ]]; then
        continue
      fi
      if ((total_memory < minimum_gpu_memory_mib)); then
        continue
      fi
      preference_rank=1
      if [[ -z "$preferred_gpu_name_pattern" || \
        "$name" == *"$preferred_gpu_name_pattern"* ]]; then
        preference_rank=0
      fi
      printf '%s,%s,%s,%s\n' \
        "$preference_rank" "$free_memory" "$index" "$uuid"
    done <<<"$inventory" | sort -t, -k1,1n -k2,2nr
  )

  if [[ -z "$gpu_device_id" ]]; then
    if [[ -n "$requested_gpu" && "$requested_gpu_found" == false ]]; then
      return 2
    fi
    return 1
  fi

  printf 'Selected GPU %s (%s, %s MiB free).\n' \
    "$gpu_index" "$gpu_device_id" "$gpu_free_memory" >&2
}

wait_for_gpu() {
  local wait_started="$SECONDS"
  local wait_elapsed
  local wait_remaining
  local next_interval
  local selection_status
  local wait_description

  if [[ -n "$requested_gpu" ]]; then
    wait_description="requested GPU $requested_gpu"
  elif [[ -n "$gpu_name_pattern" && "$minimum_gpu_memory_mib" != 0 ]]; then
    wait_description="an idle, unlocked $gpu_name_pattern GPU with at least $minimum_gpu_memory_mib MiB total memory"
  elif [[ -n "$gpu_name_pattern" ]]; then
    wait_description="an idle, unlocked $gpu_name_pattern GPU"
  elif [[ "$minimum_gpu_memory_mib" != 0 ]]; then
    wait_description="an idle, unlocked NVIDIA GPU with at least $minimum_gpu_memory_mib MiB total memory"
  else
    wait_description="an idle, unlocked NVIDIA GPU"
  fi

  while true; do
    if select_gpu true; then
      return 0
    else
      selection_status=$?
    fi
    if ((selection_status == 2)); then
      printf 'Requested GPU was not found: %s\n' "$requested_gpu" >&2
      return 1
    fi
    if [[ "$gpu_acquire_mode" == fail-fast ]]; then
      printf '%s\n' 'No eligible GPU is immediately available.' >&2
      return 75
    fi

    wait_elapsed="$((SECONDS - wait_started))"
    if ((wait_elapsed >= gpu_wait_timeout)); then
      printf 'Timed out after %s second(s) waiting for %s.\n' \
        "$gpu_wait_timeout" "$wait_description" >&2
      return 124
    fi

    wait_remaining="$((gpu_wait_timeout - wait_elapsed))"
    next_interval="$gpu_poll_interval"
    if ((next_interval > wait_remaining)); then
      next_interval="$wait_remaining"
    fi
    printf '[%s] Waiting for %s; elapsed %s second(s), checking again in %s second(s).\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$wait_description" \
      "$wait_elapsed" \
      "$next_interval" >&2
    sleep "$next_interval"
  done
}

if $dry_run; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! select_gpu false; then
      if [[ -n "$requested_gpu" ]]; then
        printf 'Requested GPU was not found or is unavailable: %s\n' \
          "$requested_gpu" >&2
      else
        printf 'No idle NVIDIA GPU is available.\n' >&2
      fi
      exit 1
    fi
  else
    gpu_device_id="${requested_gpu:-AUTO_FREE_GPU}"
  fi
else
  if [[ "$data_root" != "$data_base"/* ]]; then
    printf 'Refusing to run: PAPERBENCH_DATA_ROOT must be below %s: %s\n' \
      "$data_base" "$data_root" >&2
    exit 1
  fi
  if [[ ! -d "$(dirname -- "$data_root")" || ! -w "$(dirname -- "$data_root")" ]]; then
    printf 'Data parent is unavailable or not writable: %s\n' "$(dirname -- "$data_root")" >&2
    exit 1
  fi
  for command_name in nvidia-smi flock; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      printf 'Required command is unavailable: %s\n' "$command_name" >&2
      exit 1
    fi
  done

  mkdir -p \
    "$cache_dir/host-uv" \
    "$runs_dir" \
    "$agent_tmp_dir" \
    "$reproduction_tmp_dir" \
    "$auth_runtime_root" \
    "$gpu_lock_dir"
  chmod 700 "$data_root" "$cache_dir" "$runs_dir" "$auth_runtime_root" "$gpu_lock_dir"
  chmod 1777 "$agent_tmp_dir" "$reproduction_tmp_dir"
  runtime_codex_home="$(mktemp -d "$auth_runtime_root/run.XXXXXX")"
  runtime_auth_file="$runtime_codex_home/auth.json"
  chmod 700 "$runtime_codex_home"
  install -m 600 "$auth_file" "$runtime_auth_file"
  cleanup_runtime_auth() {
    rm -rf -- "$runtime_codex_home"
  }
  trap cleanup_runtime_auth EXIT
  wait_for_gpu
fi

use_nvidia_visible_devices=false
case "$nvidia_visible_devices_mode" in
  1|true)
    use_nvidia_visible_devices=true
    ;;
  auto)
    if ! $dry_run && [[ ! -S /run/nvidia-persistenced/socket ]]; then
      # A stale system CDI spec may still require this deleted socket. The
      # legacy runtime selects the same locked GPU without that host mount.
      use_nvidia_visible_devices=true
      printf '%s\n' \
        'NVIDIA persistence socket is absent; using NVIDIA_VISIBLE_DEVICES runtime fallback.' >&2
    fi
    ;;
esac

command=(
  uv run python -m paperbench.nano.entrypoint
  paperbench.paper_id="$paper_id"
  paperbench.runs_dir="$runs_dir"
  paperbench.docker_image=pb-codex-env:latest
  paperbench.solver.time_limit=86400
  paperbench.solver.upload_interval_seconds=1800
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig
  paperbench.solver.computer_runtime.env.pull_from_registry=false
  paperbench.solver.computer_runtime.env.is_nvidia_gpu_env=true
  paperbench.solver.computer_runtime.env.gpu_device_id="$gpu_device_id"
  paperbench.solver.computer_runtime.env.use_nvidia_visible_devices="$use_nvidia_visible_devices"
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
  paperbench.reproduction.computer_runtime.env.gpu_device_id="$gpu_device_id"
  paperbench.reproduction.computer_runtime.env.use_nvidia_visible_devices="$use_nvidia_visible_devices"
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
  paperbench.judge.completer_config.codex_home="$runtime_codex_home"
  paperbench.judge.completer_config.reasoning_effort=medium
  paperbench.judge.completer_config.max_concurrency=4
  runner.max_retries=0
  runner.concurrency=1
  runner.recorder=nanoeval.json_recorder:json_recorder
)

if $use_codex_solver_defaults; then
  command+=(
    paperbench.solver=paperbench.solvers.codex.solver:CodexSolver
    paperbench.solver.codex_auth_file="$runtime_auth_file"
    paperbench.solver.persist_refreshed_auth=true
    paperbench.solver.model=gpt-5.6-sol
    paperbench.solver.reasoning_effort=high
    paperbench.solver.reasoning_summary=detailed
  )
fi

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

export UV_CACHE_DIR="$cache_dir/host-uv"
cd "$paperbench_root"
"${command[@]}"
