# PaperBench

This repo contains the dataset and code for the paper "PaperBench: Evaluating AI's Ability to Replicate AI Research".

## Leaderboard

### PaperBench Results

| Agent                              | Score (%)  | # runs | Date       |
| ---------------------------------- | ---------- | ------ | ---------- |
| IterativeAgent o1-high (36h limit) | 26.0 ± 0.3 | 3      | 2025-04-02 |
| IterativeAgent o1-high (24h limit) | 24.4 ± 0.7 | 3      | 2025-04-02 |
| BasicAgent claude-3.5-sonnet       | 21.0 ± 0.8 | 3      | 2025-04-02 |
| IterativeAgent claude-3.5-sonnet   | 16.1 ± 0.1 | 3      | 2025-04-02 |
| BasicAgent o1-high                 | 13.2 ± 0.3 | 3      | 2025-04-02 |
| IterativeAgent o3-mini-high        | 8.5 ± 0.8  | 3      | 2025-04-02 |
| BasicAgent deepseek-r1             | 6.0 ± 0.3  | 3      | 2025-04-02 |
| BasicAgent gpt-4o                  | 4.1 ± 0.1  | 3      | 2025-04-02 |
| BasicAgent gemini-2.0-flash        | 3.2 ± 0.2  | 3      | 2025-04-02 |
| BasicAgent o3-mini-high            | 2.6 ± 0.2  | 3      | 2025-04-02 |

### PaperBench Code-Dev Results

| Agent                  | Score (%)  | # runs | Date       |
| ---------------------- | ---------- | ------ | ---------- |
| IterativeAgent o1-high | 43.4 ± 0.8 | 3      | 2025-04-02 |

## Introduction

PaperBench evaluates AI agents on replicating 20 Spotlight and Oral papers from ICML 2024 from scratch.

Each sample of PaperBench includes a research paper and a rubric that defines the requirements for a successful replication.

PaperBench runs in 3 stages:

1. **Agent Rollout**: The agent is executed in an ubuntu container where it must create its submission: a codebase that replicates the paper.
2. **Reproduction**: The agent's submitted codebase is executed in a fresh second container with GPU access so that results from executing the codebase are obtained. This creates the _executed submission_.
3. **Grading**: The _executed submission_ is graded using paper's rubric. A third container is created where the judge is run.

## PaperBench Setup

All commands in this README should be run from the [root](./) of the PaperBench project.

### Installation

Install PaperBench with [uv](https://docs.astral.sh/uv/)

```console
uv sync
```

### Get the data

The dataset is stored using [Git-LFS](https://git-lfs.com/) and is intentionally not fetched during the install
step above. Hydrate it manually and point PaperBench at the hydrated directory:

```console
git clone https://github.com/openai/frontier-evals.git --filter=blob:none
cd frontier-evals
git lfs fetch --include "project/paperbench/data/**" --exclude ""
git lfs checkout project/paperbench/data
export PAPERBENCH_DATA_DIR="$(pwd)/project/paperbench/data"  # add to your shell profile
```

If you are already working from a full `frontier-evals` clone, you can run the `git lfs fetch` / `git lfs checkout`
commands from the repository root and skip setting `PAPERBENCH_DATA_DIR` (the default path resolves to
`<repo>/project/paperbench/data`). When the data is stored elsewhere, set the environment variable to the location
you hydrated.

The JudgeEval tarballs that cannot be redistributed automatically can still be created with:

```console
PAPERBENCH_DATA_DIR=/path/to/data uv run python -m paperbench.judge.judge_eval.download_data
```

### Environment variables

PaperBench requires API keys for running the agents and judge. To set up your environment variables, copy the top-level `.env.example` file to a new file named `.env` and fill in the required values.

**Note:** The API key used by the Judge, `GRADER_OPENAI_API_KEY`, defaults to `OPENAI_API_KEY` if it's not set.

Once you have filled in the `.env` file, source it:

```bash
source .env
```

### Agent resources

Some papers require access to the OpenAI API and HuggingFace in order to be fully replicated. To make these API keys available to the agent, first copy `paperbench/solvers/agent.env.example` to `paperbench/solvers/agent.env` and populate the `OPENAI_API_KEY` and `HF_TOKEN` fields. This is a local secrets file and is ignored by Git. Values must be written literally in `KEY=value` format; the current loader does not expand shell references such as `OPENAI_API_KEY=${OPENAI_API_KEY}`. The `agent.env` is provided in the agent's workspace and the agent is informed of this file in our [default instructions for BasicAgent](paperbench/solvers/aisi-basic-agent/templates.py), but `agent.env` is not sourced automatically on behalf of the agent. During reproduction, PaperBench reads this same local file and sets its entries as environment variables in the separate reproduction container.

The HuggingFace token can be obtained through the website after making an account, and should have authorization to access [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b) and [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

For convenience, the table below lists which papers require a HuggingFace token or OpenAI API key for it to be fully replicated:

| **Paper ID**                            | **HF Token Needed?**                                                 | **OpenAI API Needed?**                |
| --------------------------------------- | -------------------------------------------------------------------- | ------------------------------------- |
| **lca‑on‑the‑line**                     | Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k)) | No                                    |
| **stochastic‑interpolants**             | Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k)) | No                                    |
| **test‑time‑model‑adaptation**          | Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k)) | No                                    |
| **robust‑clip**                         | Yes ([ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k)) | No                                    |
| **bbox**                                | No                                                                   | Yes (`gpt‑3.5‑turbo` / `gpt-4`)       |
| **semantic‑self‑consistency** (dev set) | No                                                                   | Yes (`gpt‑3.5‑turbo` / `gpt‑4o‑mini`) |

### Build Docker images

Running agents with nanoeval and alcatraz requires Docker, which you can install by following the guide [here](https://docs.docker.com/engine/install/).

**(Optional):** To enable GPU support for the agents, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

PaperBench requires the following Docker images to be built before it can run:

- [paperbench/Dockerfile.base](paperbench/Dockerfile.base): The base environment for agents which contains basic packages like `curl`, `git`, `python`, etc. and a Conda environment that can be used to run the agents. We re-use this docker image if users request grading to occur on a computer with `paperbench.judge.grade_locally=false`
- [paperbench/solvers/codex/Dockerfile.agent](paperbench/solvers/codex/Dockerfile.agent): The `pb-env` image extended with a pinned Codex CLI for CodexSolver rollouts.
- [paperbench/reproducer.Dockerfile](paperbench/reproducer.Dockerfile): The environment in which the submission is reproduced.

We provide the following agents out-of-the-box:

- [paperbench/solvers/dummy/solver.py](paperbench/solvers/dummy/solver.py): A agent for that creates a dummy submission, useful for testing the eval end-to-end.
- [paperbench/solvers/basicagent/solver.py](paperbench/solvers/basicagent/solver.py): Simple ReAct style agents with tools available to them.
- [paperbench/solvers/codex/solver.py](paperbench/solvers/codex/solver.py): A solver that runs Codex CLI non-interactively inside the PaperBench agent container.

For convenience, we've provided a [script](paperbench/scripts/build-docker-images.sh) that builds all the above images:

```bash
bash paperbench/scripts/build-docker-images.sh
```

## Set agent configurations

Configurations for each agent are stored in the [paperbench/solvers/](paperbench/solvers/) directory. Each agent has a `config.yaml` file which sets the agent's possible configurations, e.g. which model to use, how long to run the agent for, etc.

## Quickstart

To see all configurable options for PaperBench, run:

```bash
uv run python -m paperbench.nano.entrypoint --help
```

To use our default solver (`BasicAgentSolver`), you must set `paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver`, specifying to use the `BasicAgentSolver` solver.

To test that things are set up correctly, run the dummy agent with the dummy judge (`paperbench.judge.scaffold=dummy`) on the debug split. Note this dummy agent is different to the [default nanoeval "dummy" agent](../common/nanoeval/nanoeval/solvers/computer_tasks/solver.py); our dummy agent tests several aspects of the infrastructure and makes a toy submission.

```bash
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.solvers.dummy.solver:PaperBenchDummySolver \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.judge.scaffold=dummy \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

To run BasicAgent, use `paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver`. By default, this will run for 5 minutes and use `gpt-4.1-mini`.

Below, we present a command that you can use for development, which runs the
above configuration on on the dev split of PaperBench, with short timeouts and
cheap models.

```bash
uv run python -m paperbench.nano.entrypoint \
    paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.paper_split=dev \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model='gpt-4.1-mini' \
    paperbench.reproduction.timeout=60 \
    runner.max_retries=0 \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

- **Note** that we set the default concurrency to 5, you may want to adjust this based on your setup. To set the concurrency, simply specify `runner.concurrency=<int>` in the command.
- **Note** By default, our solvers use the `AlcatrazComputerRuntime` for the `ComputerRuntime`. You may choose to use your own custom computer runtime. Simply point to it via `paperbench.solver.computer_runtime`, and then configure it appropriately. We recommend looking at the `BasePBSolver` implementation for guidance.
- **Note** You may use a custom solver: set `paperbench.solver` to point to your custom solver class. This solver should be a `PythonCodingSolver` class. We recommend inheriting from `BasePBSolver` for convenience.

### Codex CLI on semantic-self-consistency

Set `paperbench.paper_id=semantic-self-consistency` to select only that development paper; a
custom single-paper split is not required. The Codex agent image pins the CLI version at build
time, while the solver records the actual version,
model, reasoning effort, reasoning-summary setting, wall-clock runtime, exit status, and raw
JSONL events for each run. The solver requests `detailed` public reasoning summaries by default;
when the selected model emits them, they appear as reasoning items in `agent.log`. These are
model-provided summaries, not the model's private chain-of-thought.
The JSONL event stream is saved as PaperBench's `agent.log`, so the standard blacklist monitor
runs before reproduction and grading. While Codex is running, the solver saves the current
submission and logs every 30 minutes; the solver also saves a final checkpoint as soon as Codex
exits, before PaperBench validates and grades the submission.
At each checkpoint, the solver also derives `url-observations.jsonl` from the raw event stream.
Each record identifies a redacted URL exposed by a Codex `web_search` event or by the command
text of a `command_execution` event. Repeated started/completed events for the same item are
consolidated. Embedded credentials and query parameters whose names indicate authentication,
keys, passwords, secrets, signatures, or tokens are replaced with `REDACTED`. A
`command_text` record means that a URL appeared in a command; it does not by itself prove that
the command successfully connected to that URL. Network requests made internally by package
managers or libraries remain outside this file unless Codex exposes their URLs in an event.
Only the Codex CLI is added to the agent image. The Python solver and paper selection remain in
the host checkout, where PaperBench controls the container lifecycle. Because Codex uses
shell commands directly, `CodexSolver` defaults to PaperBench's shell-only Alcatraz runtime.
Before `codex exec` starts, the solver passes the unchanged non-iterative BasicAgent system
message selected for the current `code_only` setting through Codex's `developer_instructions`
configuration. The user input contains only the normal PaperBench task instructions and the
exact time budget, detected GPU availability, `agent.env` location, and root access formatted
with BasicAgent's existing runtime templates. It does not add Codex-specific completeness,
baseline-prioritization, or coverage-audit instructions. The solver runs the same Docker sanity
checks as BasicAgent and records the detected GPU and instruction mode in
`codex-rollout.json`.

Codex authentication is deliberately separate from the paper's API access. Run `codex login`
with ChatGPT on the host first. `CodexSolver` mounts the resulting
`~/.codex/auth.json` read-only at `/home/.codex/auth.json` in the agent container, verifies that
`codex login status` reports ChatGPT authentication, and removes `OPENAI_API_KEY` from the
`codex exec` process environment. The auth file is outside `/home/submission` and `/home/logs`,
so PaperBench does not include it in submission snapshots. If Codex stores credentials only in
the OS keychain, configure its credential store as `file` and sign in again before this run.

The target paper still requires OpenAI API access when the generated reproduction code calls
`gpt-3.5-turbo` and `gpt-4o-mini`. Put the literal key only in the Git-ignored local file
`paperbench/solvers/agent.env`:

```dotenv
OPENAI_API_KEY=your-project-scoped-api-key
```

The agent may explicitly source `/home/agent.env` when testing paper code; the separate
reproduction container receives the same value as an environment variable automatically.
PaperBench excludes its data directory from Git LFS downloads by default, so fetch this task
explicitly. Then build all three local images and run:

```bash
git -c lfs.fetchexclude= lfs pull \
    --include='project/paperbench/data/papers/semantic-self-consistency/**'

bash paperbench/scripts/build-docker-images.sh

uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_id=semantic-self-consistency \
    paperbench.docker_image=pb-codex-env:latest \
    paperbench.solver=paperbench.solvers.codex.solver:CodexSolver \
    paperbench.solver.codex_auth_file='~/.codex/auth.json' \
    paperbench.solver.model='gpt-5.6-sol' \
    paperbench.solver.reasoning_effort=high \
    paperbench.solver.reasoning_summary=detailed \
    paperbench.solver.time_limit=86400 \
    paperbench.solver.upload_interval_seconds=1800 \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
    paperbench.reproduction.timeout=86400 \
    runner.max_retries=0 \
    runner.concurrency=1 \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

This Semantic Self-Consistency task excludes the local Llama and Mistral experiments. Its
required language-model generations use the OpenAI API, while SciBERT and RoBERTa-base
featurization can run on CPU. The command above therefore uses CPU reproduction on a Mac;
measure a bounded preflight before committing to the full API generation workload.

For an initial code-only smoke test, add `paperbench.judge.code_only=True`. The solver still
runs inside PaperBench, so rollout time, logs, blacklist monitoring, and submission capture
retain their normal benchmark meaning.

For a completion-persistence intervention, both `CodexSolver` and the iterative
`BasicAgentSolver` accept `completion_review=True`. At the first normal completion, the solver
preserves `initial-submission.tar.gz` and, when at least 1,800 seconds of the original rollout
budget remain, continues the same session once using only that remaining budget. The concise
review instruction reports the remaining minutes and asks the Agent to check for omitted
in-scope experiments and incomplete end-to-end paths; if the paper is already fully reproduced,
it may finish without changes. This is opt-in and does not add time beyond the original limit.

PaperBench also supports `paperbench.requirements_mode=rubric-criteria`. This uploads
`/home/paper/rubric_criteria.json`, a flat JSON array of the rubric leaves' `requirements` text
copied verbatim. Weights, hierarchy, IDs, `judge.addendum.md`, and grader-only fields are not
included. The mode is distinct from `rubric-visible`, which exposes the raw evaluation files.

The criteria files are pre-generated for every registered paper under
`data/rubric-criteria/`; task setup verifies the source-rubric and artifact hashes recorded in
`manifest.json` before uploading one file. Regenerate the complete set after changing any
rubric, then verify that no generated file is missing or stale:

```bash
python -m paperbench.scripts.generate_rubric_criteria
python -m paperbench.scripts.generate_rubric_criteria --check
```

Generation first validates every source rubric and only then replaces the generated set. A
source error therefore cannot partially update otherwise valid artifacts; after correcting the
source, rerunning the command rebuilds all papers. The registry currently contains 23 papers:
the benchmark's `all` evaluation split contains 20, while the other 3 are development papers in
the `dev` split.

#### Supplying reproduction Requirements

PaperBench can optionally provide a generated Requirements CSV as an additional agent input.
This is an execution condition, not a judge setting. The paper, addendum, rubric, solver,
network access, time limit, and judge remain unchanged.

For the baseline condition, omit `paperbench.requirements_csv`. To run the same task with
Requirements, add the host path to the existing command:

```bash
paperbench.requirements_csv='/absolute/path/to/requirements.csv'
```

PaperBench validates that the CSV contains `requirement_id`, `requirement_statement`, and
`procedure_category`, then copies the exact file to
`/home/requirements/requirements.csv` in the agent container and marks it read-only. The task
instructions tell the agent to read each `requirement_statement`. When the option is omitted,
PaperBench does not create the Requirements directory and does not change the instructions.
The Docker image does not need to be rebuilt when the CSV changes.

Each run's `metadata.json` records whether Requirements were provided, their row count, their
SHA-256, and the container path. The run-group summary records the same condition without
embedding the machine-specific host path. Use a new run group when changing this condition;
resuming a rollout created under another input condition would not be a valid comparison.

#### Running a PoC paper on roemia with experiment data under `/data`

Keep the `frontier-evals` checkout under `/data/$USER` on roemia. The
[`run-paperbench-roemia.sh`](paperbench/scripts/run-paperbench-roemia.sh) launcher runs one full
PoC task selected by `--paper`. The launcher accepts aliases listed in
[`experiments/paper-aliases.tsv`](experiments/paper-aliases.tsv) and passes the resulting paper
ID directly to PaperBench. The registry contains every PaperBench paper, so adding a per-paper
split file is unnecessary. Docker images remain in Docker's configured data root,
while host run artifacts are written to `/data/$USER/paperbench/runs`. The launcher bind-mounts
`/data/$USER/paperbench/cache` at `/root/.cache` in both the Codex agent and reproduction
containers, and uses separate `/data`-backed `/tmp` directories for the two stages. This keeps
Hugging Face, Torch, pip, dataset, and temporary downloads out of Docker's writable layers.
At startup, the launcher excludes GPUs with active compute processes, chooses the remaining GPU
with the most free memory, and passes its UUID to both stages. Alcatraz exposes only that GPU to
each Docker container. A per-GPU lock under `/data/$USER/paperbench/gpu-locks` prevents concurrent
instances of this launcher from choosing the same device.

After building `pb-env`, `pb-codex-env`, and `pb-reproducer`, inspect the command without
starting a run:

```bash
paperbench/scripts/run-paperbench-roemia.sh --paper bam --dry-run
```

Run the BaM baseline condition without a Requirements CSV:

```bash
paperbench/scripts/run-paperbench-roemia.sh --paper bam
```

To request a particular idle GPU by host index or UUID instead of selecting one automatically:

```bash
paperbench/scripts/run-paperbench-roemia.sh --paper bam --gpu 0
```

The launcher stops rather than falling back to another device if the requested GPU is busy or
locked. `PAPERBENCH_GPU_DEVICE` provides the equivalent environment override.

Run the Requirements condition by naming the exact CSV explicitly:

```bash
paperbench/scripts/run-paperbench-roemia.sh \
    --paper bam \
    --requirements-csv /absolute/path/to/requirements.csv
```

Override `PAPERBENCH_DATA_ROOT` only when another `/data` location is required. The launcher
rejects non-`/data` roots during a real run. `--dry-run` allows a temporary root so the storage
configuration can be tested without a GPU server.

#### Switching the judge backend

Keep the rollout, submission, rubric, and `SimpleJudge` unchanged, and select only the
`TurnCompleter` backend.

For development grading against the local Qwen vLLM server, add:

```bash
paperbench.judge.completer_config=paperbench.judge.local_judge_completer:LocalJudgeCompleter.Config \
paperbench.judge.completer_config.model='Qwen/Qwen3.6-27B' \
paperbench.judge.completer_config.base_url='http://127.0.0.1:58137/v1' \
paperbench.judge.completer_config.tokenizer_path='/absolute/path/to/Qwen--Qwen3.6-27B/tokenizer.json' \
paperbench.judge.completer_config.max_concurrency=4
```

For two vLLM data-parallel ranks behind the same endpoint, use eight concurrent
Judge calls and enable explicit rank routing:

```bash
paperbench.judge.completer_config.max_concurrency=8 \
paperbench.judge.completer_config.data_parallel_size=2
```

`LocalJudgeCompleter` sends the existing role-tagged Judge messages directly to vLLM's
OpenAI-compatible `/chat/completions` endpoint. It does not add or rewrite a prompt. The same
backend is automatically used for the main leaf decision and the structured score-parsing call.
When `data_parallel_size` is greater than one, these completers share a least-in-flight router
and send vLLM's `X-data-parallel-rank` header. A rank is released after either a successful
response or an exception. This prevents completed and waiting requests from accumulating on one
rank while another rank is idle.
Structured calls send a strict JSON Schema and validate the returned JSON before PaperBench uses
it. The defaults target Qwen3.6-27B with `temperature=1.0`, `top_p=0.95`,
`presence_penalty=1.5`, and `seed=0`. Like PaperBench's API Judge configuration, the local
completer uses a 200,000-token context and does not send `max_tokens` unless it is explicitly
overridden. When `tokenizer_path` is set, `SimpleJudge` truncates its paper, logs, and selected
submission files using that model's Hugging Face `tokenizer.json`; API and Codex CLI completers
continue to use their original `tiktoken` encodings. The context value is a maximum rather than a
fixed allocation for every request.
Generation settings and
the model and tokenizer-file content are part of the leaf-checkpoint identity; the server address,
timeout, retry count, and concurrency are operational settings and do not invalidate completed
leaves. The local server
needs no OpenAI credential; when its endpoint requires a bearer token, store it only in the
environment variable named by `api_key_env` (default: `QWEN_JUDGE_API_KEY`).

The local score is a development proxy, not a PaperBench-comparable final score, because the
model and serving stack differ from the canonical Judge. Start vLLM after GPU reproduction has
finished, or reserve a separate GPU for it. To grade an already saved and executed submission
without rerunning the Agent or reproduction, use the same completer with the standalone Judge.
On roemia, wrap this command with
`research/project/ReproGapBench/infra/roemia/with_qwen_judge.sh --gpu <GPU> --` to start the
server only when absent and safely stop only a server owned by that wrapper:

```bash
uv run python paperbench/scripts/run_judge.py \
  submission_path=/absolute/path/to/submission_executed \
  paper_id=bam \
  judge=simple \
  code_only=True \
  out_dir=/data/$USER/paperbench/local-judge/bam \
  leaf_checkpoint_path=/data/$USER/paperbench/local-judge/bam/leaf-checkpoints.jsonl \
  completer_config=paperbench.judge.local_judge_completer:LocalJudgeCompleter.Config \
  completer_config.base_url='http://127.0.0.1:58137/v1' \
  completer_config.tokenizer_path='/absolute/path/to/Qwen--Qwen3.6-27B/tokenizer.json'
```

Restart the same command with the same `leaf_checkpoint_path` to reuse every
compatible, successfully graded leaf. The standalone command fingerprints the
submission directory before grading, so the checkpoint is not reused after a
submission file changes.

For development grading with the ChatGPT-authenticated Codex CLI, add:

```bash
paperbench.judge.completer_config=paperbench.judge.codex_cli_turn_completer:CodexCliTurnCompleter.Config \
paperbench.judge.completer_config.model='gpt-5.4' \
paperbench.judge.completer_config.codex_home='~/.codex' \
paperbench.judge.completer_config.reasoning_effort=high \
paperbench.judge.completer_config.timeout_seconds=900 \
paperbench.judge.completer_config.timeout_retries=1 \
paperbench.judge.completer_config.max_concurrency=4
```

`gpt-5.4` is the oldest non-Spark general-purpose reasoning model currently exposed by the
ChatGPT-authenticated Codex CLI. It is still substantially newer than the canonical
`o3-mini-2025-01-31` judge, so it is only a development proxy. The `high` reasoning setting
matches PaperBench's canonical main judge setting. The CLI backend
serializes the existing Judge conversation and adds only the transport instruction
`Respond to the following conversation as the assistant.` because `codex exec` accepts one
input prompt rather than a native role-tagged conversation. It otherwise leaves the Judge
messages unchanged. Shell, web-search, subagent, and app tools are disabled through Codex CLI
configuration rather than through an extra prompt instruction. The backend
copies only `auth.json` into a private temporary `CODEX_HOME`, verifies that
`codex login status` reports ChatGPT authentication, removes `OPENAI_API_KEY` and
`GRADER_OPENAI_API_KEY` from every Codex process, and deletes the temporary home after each
call. It applies the same backend to both the main judge call and the structured score-parsing
calls. It does not require `GRADER_OPENAI_API_KEY`. It uses the ChatGPT plan's Codex allowance
and may stop when that allowance is exhausted.

For the final PaperBench-comparable grade, replace the CLI fields above with the canonical API
completer and provide `GRADER_OPENAI_API_KEY`:

```bash
paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
paperbench.judge.completer_config.model='o3-mini-2025-01-31' \
paperbench.judge.completer_config.reasoning_effort=high
```

`grader_output.json` records the exact completer class in `completer_backend` and its settings
in `completer_config`. Treat a CLI grade as a development measurement: model snapshots and the
Codex execution layer differ from the canonical API judge. Report the API grade as the final
PaperBench score, and compare CLI/API leaf decisions separately when evaluating the cheaper
development judge.

#### Reproduction salvage attempt selection

PaperBench retains its existing reproduction salvage conditions: an attempt that runs for less
than `retry_threshold` (600 seconds by default) is retried with the next Python/virtualenv
configuration, while an attempt reaching the threshold is selected immediately. If every
attempt is shorter than the threshold, the longest attempt is selected instead of the last
attempt. This fallback follows PaperBench's existing assumption that a longer run is less likely
to be an early exit, while preventing a short failure in the final configuration from
arbitrarily replacing a more substantive earlier attempt.

A timed-out reproduction is terminal and is never retried with another Python/virtualenv
configuration. A timeout has already consumed the configured reproduction budget (24 hours in
the full launcher), so retrying it could multiply one run into four full-duration attempts
without evidence that an interpreter change would help.

When salvage is enabled, each attempt is retained as
`submission_executed_attempt_N.tar.gz` with matching metadata. The selected archive is copied to
the canonical `submission_executed.tar.gz` path consumed by the Judge. Consequently, future
analysis can select another retained attempt without rerunning the Agent rollout.

#### Judge timeout and leaf-level resume

Each Codex CLI call has a 900-second limit by default. Codex is started in a separate POSIX
process group; when the limit is reached, PaperBench sends `SIGTERM` to the entire group,
waits five seconds, and then sends `SIGKILL` if anything remains. The timed-out call is retried
once by default. These are per-call settings: one leaf can include file selection, grading, and
structured score-parsing calls, and only the call that timed out is repeated. Set
`timeout_retries=0` to disable that retry.

For the local vLLM completer, an `APITimeoutError` is recorded immediately as
an invalid leaf instead of entering the Judge's general invalid-leaf retry
loop. Configure `completer_config.max_retries=0` when one HTTP attempt per leaf
is required. Other transient exceptions and invalid model responses retain the
general leaf retry behavior.

Every successfully graded leaf is appended immediately to
`*_leaf_checkpoints.jsonl` beside the grader output. If grading is interrupted before
`*_grader_output_*.json` is written, running the Judge again with the same checkpoint file
reuses completed leaves and calls the Judge only for unfinished or invalid leaves. A saved
leaf is reused only when all of the following match:

- the extracted submission paths and file contents;
- the target paper, agent addendum, and Judge addendum;
- the filtered rubric, including `code_only` and resource settings;
- the Judge type, model, reasoning setting, and response format;
- the leaf ID and full leaf requirement.

Authentication paths, concurrency, timeout, and retry settings are operational details and do
not invalidate otherwise compatible checkpoints. When retrying in the same run path, the
sidecar is selected automatically. When launching a new run around the same saved submission,
pass the previous file explicitly:

An individual leaf that raises an exception or returns `valid_score=False` is retried up to
three times after its initial attempt. Only a valid result is written to the checkpoint. If all
four attempts are invalid, the final invalid result is retained in the grader output so that the
run remains auditable and the leaf can be resumed later from the same checkpoint.

```bash
paperbench.judge.leaf_checkpoint_path='/absolute/path/to/submission_executed_grader_output_0_leaf_checkpoints.jsonl'
```

For `paperbench/scripts/run_judge.py`, use the standalone field instead:

```bash
leaf_checkpoint_path='/absolute/path/to/leaf-checkpoints.jsonl'
```

The leaf checkpoint mechanism is shared by API and CLI `SimpleJudge`; process-group termination
and per-call timeout retry apply only to the Codex CLI backend. A completed grader output still
takes precedence unless `paperbench.judge.overwrite_existing_output=True` is set.

### Canonical command

To run PaperBench with BasicAgent gpt-5 for 24 hours on the full set of papers, use the
following command:

```bash
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=all \
    paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver \
    paperbench.solver.completer_config=paperbench.solvers.basicagent.completer:OpenAIResponsesTurnCompleterConfig \
    paperbench.solver.completer_config.model='gpt-5-2025-08-07' \
    paperbench.solver.completer_config.tools="[{'type':'web_search'}]" \
    paperbench.solver.completer_config.reasoning=preparedness_turn_completer.oai_responses_turn_completer.completer:ReasoningConfig \
    paperbench.solver.completer_config.reasoning.effort='high' \
    paperbench.solver.completer_config.reasoning.summary='auto' \
    paperbench.solver.time_limit=86400 \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.solver.computer_runtime.env.is_nvidia_gpu_env=true \
    paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
    paperbench.reproduction.computer_runtime.env.is_nvidia_gpu_env=true \
    paperbench.reproduction.timeout=86400 \
    runner.max_retries=0 \
    runner.concurrency=20 \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

If you want GPUs when running locally, set `is_nvidia_gpu_env=true` on each component's runtime as shown above. This assumes your Docker host already has the NVIDIA driver and container toolkit installed so the `nvidia` runtime is available. This of course does not apply if you are using a different `ComputerRuntime`. By default, we are using the `AlcatrazComputerRuntime`, for which GPUs are configured as described above.

### I have submissions and just want to run grading

If you have already run your agent or you are a human and have submissions
ready to be graded, you can skip the agent rollout by using our provided
`PBDirectSubmissionSolver`. Place your submissions as folders in a directory (say
in `path/to/foo/`) in the following format:

```plaintext
foo/
    <paper_id_1>/
        submission/
        other_submission/
        ...
        (i.e. multiple submissions for the same paper (intended for when paperbench.n_tries > 1).
        Single submission is also fine. Missing submissions for a try will just
        be graded as scoring 0 for that try)
    ...
    (does not need a folder for each paper, missing papers will be treated as
    missing submissions, i.e. graded as scoring 0)
```

Then run the paperbench like above but with appropriately configured `PBDirectSubmissionSolver`, e.g.:

```bash
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=path/to/foo/ \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.judge.scaffold=dummy \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

## Retrieving results

A run group is created for each launch of PaperBench. A run group contains multiple individual runs, one for each paper attempt. Runs are stored in the `runs` directory.

In each run group directory, there is a `group.log` file that contains the logs for that run group.

In each run directory there is:
- `agent.log`: (optional) Rollout log from the solver. If present, the monitoring step will run and the run's grade will be marked with `"monitor_ran": true`.
- `url-observations.jsonl`: (CodexSolver only) Redacted URLs exposed by Codex web-search events or written in shell command text, with the source event and item ID.
- `grade.json`: The grading result for that run.
- `metadata.json`: Metadata for that run, including the Requirements input condition and
  content hash when applicable.
- `run.log`: The log for that run.
- `status.json`: The status of that run.
- A submissions directory, containing multiple timestamped submission directories (e.g., `2025-03-28T10-34-35-UTC`), each with:
  - `log.json`: Logs from this submission attempt
  - `submission.tar.gz`: The archived submission files
  - If the submission was executed/graded, the directory may also contain:
    - `submission_executed_grader_output_0.json`: Output from the grader.
    - `submission_executed_grader_output_0_leaf_checkpoints.jsonl`: Successfully graded leaves
      saved incrementally for interrupted-Judge resume.
    - `submission_executed_metadata.json`: Metadata about the execution.
    - `submission_executed.tar.gz`: The archived files after execution.

Snapshots from the agent rollout are also stored in the run directory. An initial snapshot is created when the agent starts, and a final snapshot is created when the agent finishes. Intermediate snapshots are created throughout the agent rollout and can be set via `paperbench.solver.upload_interval_messages` or `paperbench.solver.upload_interval_seconds`.

`runs` directory structure:

```
runs/
├── <run_group_id>/
│   ├── group.log
│   ├── <run_id>/
│   │   ├── agent.log
│   │   ├── grade.json
│   │   ├── metadata.json
│   │   └── run.log
│   │   ├── status.json
│   │   ├── submissions/
│   │   │   ├── <timestamp-1>/
│   │   │   │   ├── log.json
│   │   │   │   └── submission.tar.gz
│   │   │   └── <timestamp-2>/
│   │   │       ├── log.json
│   │   │       ├── submission.tar.gz
│   │   │       ├── submission_executed_grader_output_0.json  # if graded
│   │   │       ├── submission_executed_grader_output_0_leaf_checkpoints.jsonl
│   │   │       ├── submission_executed_metadata.json  # if executed
│   │   │       └── submission_executed.tar.gz  # if executed
│   └── <other_run_ids>/...
└── <other_run_group_ids>/...
```

## PaperBench Code-Dev

**PaperBench Code-Dev** is a lighter-weight variant of PaperBench. Unlike the full PaperBench pipeline -- which involves executing the agent's submission in a separate reproduction step -- PaperBench Code-Dev skips the reproduction step and only grades the agent's submission on the **Code Development** requirements. This means:

- The Judge only checks **Code Development** requirements (e.g., "Is there an implementation of method X?"). It skips checking Execution requirements that check that the code runs correctly, and skips checking Result Match requirements that check that the paper's empirical results have been replicated.
- You **don't need a GPU to run the reproduction step** where the agent's submission is executed. This often reduces cost and runtime significantly.
- There is **less of a need to make a GPU available to the agent** when it is creating its submission. Although having access to a GPU is helpful for the agent to run intensive experiments that verify that its code is correct, the agent can get away with less end-to-end testing of its code since it is only graded on **Code Development** requirements.

We think PaperBench Code-Dev offers a convenient, lower-cost, but less rigorous way of assessing paper replication. It doesn't require GPUs and typically cuts grading costs (we've seen around an 85% reduction in o3-mini SimpleJudge costs for the average submission), making it a accessible alternative for assessing models' abilities to replicate papers.

To run the Code-Dev variant, simply include the following flag:

```bash
paperbench.judge.code_only=True
```

## Dataset

The PaperBench dataset is stored in the [papers](data/papers) directory. Each paper has its own directory, which contains:

- The paper in PDF and Markdown format.
- `addendum.md` containing information from the author that is helpful for replication.
- `assets` directory, containing necessary resources for replication. For example, images from the paper are includes in this directory.
- `rubric.json` file that is used to grade submissions
- `blacklist.txt` containing websites that the agent is disallowed from using (e.g. the paper's original codebase).
- `config.yaml` file that defines the paper's id.
- `judge.addendum.md` containing information from the author that is helpful for the judge to know whilst grading submissions. Not every paper has a judge addendum

### Viewing rubrics

We include a web app to view and edit rubrics. To view a paper's rubric that is in `PAPER_DIR` with filename `RUBRIC_FILE_NAME`:

```bash
uv run python paperbench/gui/app.py --path-to-paper <PAPER_DIR> --rubric-file-name <RUBRIC_FILE_NAME>
```

To view a rubric that has been graded (see below), pass the `--graded` flag:

```bash
uv run python paperbench/gui/app.py --path-to-paper <PAPER_DIR> --rubric-file-name <RUBRIC_FILE_NAME> --graded
```

## JudgeEval

We've created an auxiliary evaluation, [JudgeEval](paperbench/judge/judge_eval), to evaluate the accuracy of judges. See the JudgeEval [README](paperbench/judge/judge_eval/README.md) for more information.

## Authors

Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Chan Jun Shern, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, Tejal Patwardhan

## Citation

Please cite using the following BibTeX entry:

```
@misc{starace2025paperbenchevaluatingaisability,
      title={PaperBench: Evaluating AI's Ability to Replicate AI Research},
      author={Giulio Starace and Oliver Jaffe and Dane Sherburn and James Aung and Jun Shern Chan and Leon Maksin and Rachel Dias and Evan Mays and Benjamin Kinsella and Wyatt Thompson and Johannes Heidecke and Amelia Glaese and Tejal Patwardhan},
      year={2025},
      eprint={2504.01848},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.01848},
}
```
