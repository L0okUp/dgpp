# Azampatti Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound — master-port plan

**Status:** port implementation on the current master lineage; qualification is recorded separately.
**Branch:** `agent/azampatti-qwen38-a5b-int4-autoround-port` (this worktree)
**WIP lineage (this machine):** `~/dgpp-a5b` → `~/dgpp-a5b-mtp` → `~/dgpp-a5b-retune`
(tip `agent/a5b-retune`, fork base `84028a4`, 2026-09-28)

## Target checkpoint

The target is [azampatti's Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
checkpoint: the standard architecture, quantized to int4 (GPTQ layout, group-128,
healed weights), with **backbone routing at top-k 5 instead of 10**. Its BF16 MTP
draft experts remain trained and routed at **top-k 10** — a checkpoint property,
not a knob. Published serving results on one Spark via the vLLM/b12x stack land
around 70–75 tok/s; our own measured baseline on the old-base branch was
~50–53 tok/s C1 (`~/benchmarks/a5b-retune-20260929-r2`, which was interrupted
before completing qualification). The selected retune config lives in
`~/launch-configs/dgpp-a5b-mtp-retuned-selected.json` (C4, MTP depth 3,
draft-logit-scale 2.5, FP8 dense, mmap n-gram, decode graph).

## Current state of the WIP (fork base `84028a4`)

~2,017 insertions / ~139 deletions across 47 files:

- **Source-profile gate** (`src/models/qwen/config.cpp`): a strict
  `QwenSourceProfile::A5bAutoGptq` profile that validates the exact AutoRound
  dynamic-module policy, geometry (48 layers, 512 experts, top-k 5, mtp 1), and
  sets profile knowledge including `mtp_num_experts_per_tok = 10`.
- **Packed int4 experts**: `experts_packed` / `GlmPackedMatrix` plumbing, signed
  AutoRound scale handling, safetensors index-ownership fixes, PackQ K-sizes
  640 and 2560, n-gram table staged from the FP8 release via a composite
  checkpoint (`tools/a5b_composite_checkpoint.py`).
- **Serve/templates**: three one-node templates (eager, graph,
  example), retune harness (`scripts/a5b_retune.py`,
  `run_a5b_retune_then_overnight.sh`).
- **Tests**: ~600 lines (safetensors, AutoGPTQ quant, qwen config/binding/
  forward/loader), mostly not yet run against master's fixtures.

## Why this is a port, not a rebase

Current master is **~89 commits / +35k lines** ahead of the fork point, and
in that window landed **the Saren AutoRound int4 packed serving path**:

- `add_gptq` / `experts_gptq_int4` / `GlmPackedMatrix` bindings in
  `src/models/qwen/binding.cpp`, `loader.cpp` (2026-09-28 record:
  `benchmarks/results/2026-09-28-qwen-autoround-int4/`);
- `engine.ngram_table_model` for the FP8-shard n-gram table;
- `num_experts_per_tok` now read from the checkpoint config.

29 of the 47 branch files were rewritten underneath. A mechanical `git rebase`
would produce heavy conflicts in exactly the files master replaced — and then
*discard* code master already provides. So: **branch from `origin/master` and
carry the surviving deltas feature by feature.**

Surviving value of the branch: the profile gate, per-stage top-k, templates,
harness/scripts, tests, and the profile knowledge (MTP=10) that master does not
yet encode anywhere.

## Port steps

Each step ends green on unit tests; steps 1, 2 and 6 are gated as noted.

1. **Profile + bindings** (~0.5d). Port `QwenSourceProfile::A5bAutoGptq` and the
   geometry/policy assertions onto master's config validation. Bind A5B tensors
   through master's existing `experts_gptq_int4` bindings; do not port the
   branch's `experts_packed` loader path.
   *Check first:* `lm_head` quantization of the A5B checkpoint (the profile
   expects an int4 `lm_head` override; master binds `lm_head_gptq_int8` for
   Saren). If it is int4, decide between an int4 head binding and mapping onto
   master's bit-plane argmax head. **Gate:** `qwen_loader_test` + config tests.
2. **Per-stage router top-k** (~1–2d, the real work). Master routes backbone and
   MTP drafts with one global `num_experts_per_tok`; A5B needs 5 (backbone) and
   10 (MTP draft). Make K per-stage in `forward.cpp` / graph decode, defaulting
   to the global value so every existing model is bitwise unchanged.
   **Gate:** bitwise parity on all existing Qwen templates, greedy and sampled.
3. **N-gram table** (~0.5d). Replace the composite-checkpoint staging with
   `engine.ngram_table_model`; delete `tools/a5b_composite_checkpoint.py` and
   the index-ownership work that master subsumes.
   **Gate:** `qwen_load_check` on a single-node A5B run reads the table per plan.
4. **PackQ kernels** (~0.5d). Re-add compiled K = 640 / 2560 against master's
   *rewritten* `packq_gemv.cuh`; re-validate rather than trust the old diff.
   **Gate:** `packq_gemv_test` + micro-bench parity on Saren's config.
5. **Tests + templates** (~0.5d). Move safetensors/AutoGPTQ/config/binding tests
   onto master's fixtures; regenerate all three templates against the current
   `cluster_config` schema; drop harness keys master no longer reads.
6. **Finish the interrupted retune** (~1d). Complete the qualification campaign
   on master (baseline + C1/C4 candidates), record it as a proper
   `benchmarks/results/` entry, and run the standard gates: greedy bitwise match
   against the reference chain, C1/C4 decode against the old ~50–53 baseline and
   the 70–75 tok/s b12x reference.

Total ≈ 3–5 focused days; steps 2 and 6 dominate.

## Risks

- `lm_head` int4 vs int8 binding (step 1) — unresolved, may add small work.
- Per-stage top-k touches graph capture — the riskiest correctness surface.
- Retune harness keys drifted with master's `cluster_config` churn.
- The checkpoint is a pruned/healed model: ship it as its own model card and
  state the measured capability trade (its own eval: ~50 vs 51.8 capability,
  93 vs 89 tool use) alongside any performance claim.

## Definition of done

A single-node A5B template on master that loads the checkpoint unmodified,
routes 5/10 backbone-vs-draft, passes bitwise gates against Saren's int4
reference chain on non-A5B configs, and has a recorded benchmark campaign
with the retune completed.
