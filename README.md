# coda: Cross-Device Output Divergence Audit for Quantized SLMs

Starter repository for Naresh Kumar's Vizuara research project:
**do quantized small language models give different answers on different
phones?**

This repo gives you a working measurement harness on day one. It can run
end to end on a laptop with zero phones attached, by simulating the thing
that actually causes divergence (different reduction orders and different
quantization scale policies in a matrix multiply). Once you have a phone in
hand, you swap the simulated backend for a real device backend and the rest
of the pipeline, the metrics, the plots and the results table do not change.

That ordering is deliberate. Get the metrics and the plumbing right against
a reference you fully control, then point the same code at hardware.

## The research question

A quantized SLM shipped in a mobile app is assumed to be a pure function:
same weights plus same prompt plus greedy decoding equals same tokens. On a
phone that assumption is fragile, because the same artifact can execute on
an NPU, a GPU or a CPU, each with a different accumulation order and a
different activation quantization policy, and because the thermal governor
can migrate work between those units mid-session.

We measure three things, kept strictly separate:

1. **Numerical disagreement.** Do the logits differ at all?
2. **Behavioral disagreement.** Did a differing logit change the chosen token?
3. **Task disagreement.** Did a changed token change the answer?

Most of the value of this project is in not confusing those three.

## Repository layout

```
coda/
  README.md                  this file
  CLAUDE.md                  instructions for your coding agent
  requirements.txt           pinned dependencies
  configs/
    default.yaml             the one config you edit most
  src/coda/
    __init__.py
    config.py                dataclass config, loaded from YAML
    backends.py              pluggable execution backends (the core abstraction)
    kernels.py               quantized matmul variants with different numerics
    datasets.py              prompt suite loading (GSM8K, ARC-Easy, ...)
    tracing.py               per-step top-k logit trace capture and storage
    metrics.py               GAR, FDP, top-1 flip rate, margin at divergence
    thermal.py               thermal state control and sensor reading
    run_audit.py             entry point: run the device x backend matrix
    evaluate.py              entry point: turn traces into the results table
    plots.py                 entry point: regenerate the paper figures
  scripts/
    setup.sh                 create the venv and install
    run_smoke.sh             60 second end to end check, no phone needed
    run_audit.sh             the full matrix
    run_eval.sh              traces to table and figures
  experiments/
    log_template.md          one row per experiment, maps to the paper table
```

## Setup

Requires Python 3.10 or newer.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

If you prefer to do it by hand:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## The first three things to run

### 1. Smoke test (about one minute, no phone, no model download)

```bash
bash scripts/run_smoke.sh
```

This runs the whole pipeline on a tiny synthetic model across four simulated
backends, and prints Table 1 and Table 2. Identical weights, identical
prompts, and the agreement rates are not all 100 percent. That is the thesis
of the paper in miniature.

Read the output carefully, because the pattern is the first real lesson of
the project:

- `cpu_neon`, which differs from the reference **only in reduction order**,
  will likely agree with it completely. Reordering a float sum perturbs the
  logits by around 1e-6, and the smallest nonzero gap between two int8
  logits is around 1e-3, so the perturbation almost never reaches far enough
  to flip an argmax.
- `cpu_i8mm` and `npu_pertensor`, which additionally **quantize activations**
  or **collapse per-channel scales to per-tensor**, disagree substantially.

In other words: on an integer model, reduction order alone is nearly
harmless, and the damage comes from differences in quantization policy. That
is a prediction you should carry onto real hardware and try to falsify, not
a conclusion. Real NPUs also use narrow accumulators, which is a third
mechanism again; compare `npu_fp16_accum` to see it.

### 2. Real model, simulated backends (about ten minutes, downloads a model)

```bash
python -m coda.run_audit --config configs/default.yaml --backends ref,reorder,pertensor
python -m coda.evaluate --run-dir runs/latest
```

Now you have real logit traces from a real SLM, and a results table with the
same columns as Table 1 of the paper draft. Read the table. Ask yourself
whether the agreement numbers surprise you, and whether the first-divergence
positions cluster early or late.

### 3. First real device

Register a phone in `configs/default.yaml` under `devices`, then:

```bash
adb devices                                      # confirm the phone is visible
python -m coda.run_audit --config configs/default.yaml --backends llamacpp_cpu
```

`backends.py` has a documented `LlamaCppAdbBackend` stub showing exactly
which four methods a real device backend must implement. Filling those in is
your week 3 task, and it is the single highest-value piece of engineering in
the project.

## How this maps to the paper

| Repo output | Paper element |
| --- | --- |
| `runs/*/table_main.csv` | Table 1, cross-device agreement |
| `runs/*/table_ablation.csv` | Table 2, bit-width, thermal and mitigation ablations |
| `plots.py --panel margin` | Figure 2, panel B |
| `plots.py --panel thermal` | Figure 2, panel C |
| `experiments/log_template.md` | your lab notebook, one row per table row |

## Datasets

All loaders pull public datasets through `datasets` at run time. Nothing
large is committed to this repo. The default prompt suite is a stratified
subsample so that a full audit pass finishes in minutes rather than hours;
raise `dataset.n_per_task` when you are ready for the real numbers.

## A note on scope

You will be tempted to add devices. Resist that until the metrics are
trustworthy. One phone measured carefully, with within-device repeats proving
that your between-device differences are real, is a publishable result. Ten
phones measured carelessly is not.
