# CLAUDE.md

Instructions for the coding agent working with Naresh on this project.

## Project goal

Measure whether a frozen quantized small language model produces identical
outputs across different phones, different execution backends (NPU, GPU,
CPU), different bit-widths and different thermal states. Produce a
measurement paper, not a model. The deliverable is evidence, a metric
vocabulary and a released trace dataset.

The paper draft lives alongside this repo. Section names referenced below
(Methodology, Main Results, Ablations) map to that draft.

## Non-negotiable invariants

These are the things that, if broken, silently invalidate every number the
project produces. Treat a change that touches them as requiring explicit
justification in the experiment log.

1. **The artifact is frozen.** Weights are quantized once, hashed with
   SHA-256, and the hash is verified at load time on every backend. If you
   find yourself regenerating a quantized file per device, stop: you have
   just destroyed the experiment's control.
2. **Greedy decoding only.** Temperature zero, no sampling, no seeds in the
   decode path. Any disagreement we observe must be attributable to the
   stack, not to randomness. If a result changes when you change a seed, the
   seed is leaking into the measurement and that is a bug.
3. **Logits, not text.** Always log per-step top-k logits. A text-only trace
   cannot distinguish "the logits differed by 1e-7 and nothing happened"
   from "the logits differed by 1e-7 and the answer flipped". That
   distinction is the paper.
4. **Within-device repeats before between-device claims.** Every cell runs
   `config.repeats` times. No cross-device difference is reported unless
   within-device variance is smaller than between-device variance.
5. **Record the software bill of materials.** Runtime version, driver
   version, OS build, delegate, thread count and tile size go into every
   run's `manifest.json`. Driver versions move fast and an unlabelled result
   is unreproducible six months later.

## Layout and conventions

- Source lives under `src/coda/`. The package is installed editable, so
  import as `from coda import metrics`, never with relative path hacks.
- `backends.py` is the central abstraction. Every execution target,
  simulated or real, implements the `Backend` protocol: `load`, `prefill`,
  `decode_step`, `describe`. Adding a device means adding one class there and
  one entry in the config. Nothing else in the codebase should know what an
  NPU is.
- `kernels.py` holds the numerics. This is where reduction order, scale
  granularity and accumulator width are made explicit and switchable. It is
  deliberately readable rather than fast.
- `metrics.py` is pure functions over traces. No file IO, no config, no
  globals. This keeps the metrics testable and keeps the definitions honest.
- Configs are YAML, loaded into the dataclasses in `config.py`. Do not read
  raw dicts in application code; add a typed field instead.
- Runs write to `runs/<timestamp>/` and the newest is symlinked as
  `runs/latest`. Never overwrite a run directory.
- Type hints everywhere. Keep functions short enough to read in one screen.
- Standard library plus the pinned dependencies. Do not add a framework to
  solve a fifty-line problem.

## Working style with Naresh

- He is building research skill alongside the code. When you implement
  something non-obvious, especially in `kernels.py` or `metrics.py`, explain
  the numerical reasoning in a short comment, not just the mechanics.
- Prefer one small verified change over a large speculative refactor. The
  experiment log is the source of truth for what has been tried.
- When a result looks exciting, your first job is to try to break it. Check
  the within-device repeats, check the hash, check that both backends
  actually loaded the same file. Most surprising divergence results on the
  first pass are bugs.
- Do not invent numbers in documentation or in the paper. Placeholder cells
  stay as `XX.X` until a real measurement replaces them.

## Next steps, in order

1. **Make the smoke test meaningful.** `scripts/run_smoke.sh` should show a
   greedy agreement rate under 100 percent between `ref` and `reorder`. If
   it shows exactly 100 percent, the simulated backends are not actually
   differing and the harness is not yet testing anything.
2. **Wire a real SLM.** Load Llama-3.2-1B-Instruct through transformers for
   the FP32 reference trace, and a GGUF Q4_K_M build through
   `llama-cpp-python` for the quantized trace. Confirm the two disagree, and
   measure where.
3. **Build `LlamaCppAdbBackend`.** Push a `llama-cli` binary and the GGUF to
   a phone over adb, run it with a fixed prompt, and parse the logit dump.
   This is the real unlock. Budget a week and expect the logit dump format
   to be the hard part.
4. **Add the thermal protocol.** `thermal.py` already defines the three
   states and the sensor read. Make `run_audit.py` refuse to record a cell
   unless the measured temperature matches the requested state.
5. **Add a second device from a different SoC vendor.** Until then every
   claim is about one piece of silicon.
6. **Implement the mitigations** in this order, cheapest first: pinned
   delegate, margin-guarded FP32 recompute, power-of-two scales.

## What not to do

- Do not add a web dashboard, an experiment tracking service, or a plugin
  system. The project has one user and a fixed pipeline.
- Do not optimize `kernels.py` for speed. It exists to be read and to be
  obviously correct.
- Do not broaden scope to training, fine-tuning or new quantization methods.
  This is a measurement paper. Depth on one question beats breadth.
