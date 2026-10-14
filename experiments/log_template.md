# Experiment log

One row per cell of the paper's results tables. Fill a row the day you run
it, not the week you write the paper. The columns are deliberately the same
as `table_main.csv` so that copying is mechanical.

Rule: a run that is not in this file did not happen. If you cannot name the
run directory, the backend and the thermal state, you cannot put the number
in a paper.

---

## Table 1 rows: cross-device agreement

Maps to **Table 1** of the draft (`main.tex`, `tab:main`).

| date | run dir | device | SoC | backend | bit-width | thermal | GAR vs ref (%) | worst pairwise GAR (%) | median FDP | top-1 flip rate (%) | intra-device flakiness (%) | task acc (%) | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | runs/ | ref | host-fp64 | ref | int8 | cold | 100.0 | | 0 | 0.0 | | | reference trace |
| | runs/ | | | cpu_neon | | cold | | | | | | | |
| | runs/ | | | cpu_i8mm | | cold | | | | | | | |
| | runs/ | | | gpu_opencl | | cold | | | | | | | |
| | runs/ | | | npu_pertensor | | cold | | | | | | | |
| | runs/ | | | | | | | | | | | | |

**Before you trust a row:** intra-device flakiness must be at or near zero.
If a configuration disagrees with itself across repeats, every cross-device
number in that row is noise wearing a result's clothes.

---

## Table 2 rows: ablations

Maps to **Table 2** of the draft (`main.tex`, `tab:ablation`).

### Bit-width sweep

| date | run dir | recipe | bits | GAR (%) | median FDP | share of flips with margin < 0.2 | throughput retained (%) | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | runs/ | FP16 reference | 16 | | | | | |
| | runs/ | Q8_0 | 8 | | | | | |
| | runs/ | Q4_K_M | 4 | | | | | |
| | runs/ | Q4_0 | 4 | | | | | |

### Thermal sweep

The two throttled rows are the test of H4. Record the delegate the runtime
actually used, read from the runtime's own log, not from what you requested.

| date | run dir | thermal state | measured max zone (C) | delegate used | GAR (%) | median FDP | tok/s | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | runs/ | cold | | | | | | |
| | runs/ | warm | | | | | | |
| | runs/ | throttled, same delegate | | | | | | |
| | runs/ | throttled, delegate fallback | | | | | | |

### Mitigations

| date | run dir | mitigation | tau | GAR (%) | recompute cost (% of steps) | throughput retained (%) | accuracy cost (points) | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | runs/ | none | | | | 100.0 | 0.0 | baseline |
| | runs/ | power-of-two scales | | | | | | |
| | runs/ | pinned delegate | | | | | | |
| | runs/ | margin-guarded FP32 recompute | 0.5 | | | | | |
| | runs/ | margin-guarded FP32 recompute | 1.0 | | | | | |

---

## Run narrative

One short entry per session. Hypothesis first, result second, decision third.
The decision line is the one that saves you time later.

### YYYY-MM-DD

- **Question:** what did I want to know before I started?
- **Setup:** config, devices, backends, thermal states, number of prompts.
- **Artifact hash:** the SHA-256 that every backend verified.
- **Result:** the numbers, including the ones that did not support the idea.
- **Surprise:** anything I did not expect. These are usually bugs. Check them
  before you get attached to them.
- **Decision:** what I am doing next, and what I am deliberately not doing.

---

## Standing checklist

Run through this before adding any row to a table.

- [ ] Every backend verified the same artifact hash.
- [ ] Decoding was greedy, temperature zero, no sampling.
- [ ] Repeats ran and intra-device flakiness is at or near zero.
- [ ] Thermal state was read from sensors, not assumed from elapsed time.
- [ ] Runtime version, driver version and OS build are in `manifest.json`.
- [ ] The delegate actually used is recorded, not the one requested.
- [ ] Prompt suite composition (shape counts) is unchanged from the last run,
      or the change is noted here.
