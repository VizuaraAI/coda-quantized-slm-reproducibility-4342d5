"""Entry point: turn a run's traces into the paper's results tables.

    python -m coda.evaluate --run-dir runs/latest

Writes table_main.csv and table_ablation.csv next to the traces, with the
same columns as Table 1 and Table 2 of the paper draft. Those CSVs are what
you paste over the XX.X cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import metrics
from .tracing import Trace, index_by_config, load_run


def pick_reference(config_ids: list[str], requested: str | None) -> str:
    """Choose the configuration every other one is compared against.

    Config ids are ``device@thermal_state``. A bare device name is accepted
    when it resolves to exactly one configuration, because that is what you
    will type.
    """
    if requested:
        if requested in config_ids:
            return requested
        matches = [c for c in config_ids if c.split("@")[0] == requested]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise SystemExit(
                f"reference {requested!r} not in this run. Available: {config_ids}"
            )
        raise SystemExit(
            f"reference {requested!r} is ambiguous across thermal states: "
            f"{matches}. Name the state too, for example {matches[0]!r}."
        )
    for cid in config_ids:
        if cid.split("@")[0].startswith("ref") or "reference" in cid:
            return cid
    return config_ids[0]


def pairs_between(
    index: dict[str, dict[str, list[Trace]]], a: str, b: str
) -> list[metrics.PairDivergence]:
    """Compare repeat 0 of two configurations across every shared prompt."""
    shared = sorted(set(index[a]) & set(index[b]))
    return [metrics.compare_pair(index[a][p][0], index[b][p][0]) for p in shared]


def task_accuracy(traces: list[Trace]) -> float:
    """Exact-match accuracy over the prompts that carry a reference answer."""
    graded = [t for t in traces if t.meta.get("answer")]
    if not graded:
        return float("nan")
    hits = sum(metrics.exact_match(t.text, str(t.meta["answer"])) for t in graded)
    return 100.0 * hits / len(graded)


def build_main_table(
    index: dict[str, dict[str, list[Trace]]], manifest: dict, reference: str
) -> pd.DataFrame:
    """Table 1: agreement of every configuration against the reference."""
    configs = sorted(index)
    acc_by_config = {
        cid: task_accuracy([r[0] for r in index[cid].values()]) for cid in configs
    }

    rows = []
    for cid in configs:
        info = manifest.get("configurations", {}).get(cid, {})
        vs_ref = pairs_between(index, reference, cid) if cid != reference else []

        # Worst pairwise agreement against any other configuration in the run.
        worst = float("nan")
        for other in configs:
            if other == cid:
                continue
            gar = metrics.greedy_agreement(pairs_between(index, cid, other))
            worst = gar if pd.isna(worst) else min(worst, gar)

        flake = max(
            metrics.intra_device_flakiness(repeats) for repeats in index[cid].values()
        )
        rows.append(
            {
                "config_id": cid,
                "device": info.get("device", cid.split("@")[0]),
                "soc": info.get("soc", ""),
                "backend": info.get("backend", ""),
                "thermal_state": info.get("thermal_state", ""),
                "gar_vs_ref_pct": (
                    100.0 if cid == reference else metrics.greedy_agreement(vs_ref)
                ),
                "worst_pairwise_gar_pct": worst,
                "median_first_divergence": (
                    0.0 if cid == reference else metrics.median_first_divergence(vs_ref)
                ),
                "top1_flip_rate_pct": (
                    0.0 if cid == reference else metrics.top1_flip_rate(vs_ref)
                ),
                "amplification_factor": (
                    float("nan")
                    if cid == reference
                    else metrics.amplification_factor(vs_ref)
                ),
                "max_logit_error": (
                    0.0
                    if cid == reference
                    else max((p.max_logit_error for p in vs_ref), default=float("nan"))
                ),
                "intra_device_flakiness_pct": flake,
                "task_accuracy_pct": acc_by_config[cid],
            }
        )

    df = pd.DataFrame(rows)
    spread = metrics.accuracy_spread(
        {k: v for k, v in acc_by_config.items() if pd.notna(v)}
    )
    summary = {
        "config_id": "FLEET SPREAD",
        "device": "",
        "soc": "",
        "backend": "",
        "thermal_state": "",
        "gar_vs_ref_pct": df["gar_vs_ref_pct"].max() - df["gar_vs_ref_pct"].min(),
        "worst_pairwise_gar_pct": df["worst_pairwise_gar_pct"].min(),
        "median_first_divergence": df["median_first_divergence"].min(),
        "top1_flip_rate_pct": df["top1_flip_rate_pct"].max(),
        "amplification_factor": df["amplification_factor"].max(),
        "max_logit_error": df["max_logit_error"].max(),
        "intra_device_flakiness_pct": df["intra_device_flakiness_pct"].max(),
        "task_accuracy_pct": spread,
    }
    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def build_ablation_table(
    index: dict[str, dict[str, list[Trace]]], manifest: dict, reference: str
) -> pd.DataFrame:
    """Table 2: the same metrics grouped by thermal state and by backend."""
    rows = []
    for cid in sorted(index):
        if cid == reference:
            continue
        info = manifest.get("configurations", {}).get(cid, {})
        pairs = pairs_between(index, reference, cid)
        traces = [r[0] for r in index[cid].values()]
        edges, share = metrics.margin_histogram(pairs)
        rows.append(
            {
                "config_id": cid,
                "backend": info.get("backend", ""),
                "thermal_state": info.get("thermal_state", ""),
                "gar_pct": metrics.greedy_agreement(pairs),
                "median_first_divergence": metrics.median_first_divergence(pairs),
                "share_flips_margin_below_0.2": float(share[0]) if len(share) else
                float("nan"),
                # Expected cost of the margin-guarded mitigation at two
                # thresholds, which is the reproducibility versus speed curve.
                "recompute_cost_tau_0.5_pct": metrics.guarded_recompute_cost(
                    traces, 0.5
                ),
                "recompute_cost_tau_1.0_pct": metrics.guarded_recompute_cost(
                    traces, 1.0
                ),
                "task_accuracy_pct": task_accuracy(traces),
            }
        )
    return pd.DataFrame(rows)


def evaluate(run_dir: Path, reference: str | None = None) -> None:
    traces = load_run(run_dir)
    index = index_by_config(traces)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    ref = pick_reference(sorted(index), reference)
    main = build_main_table(index, manifest, ref)
    ablation = build_ablation_table(index, manifest, ref)

    main.to_csv(run_dir / "table_main.csv", index=False)
    ablation.to_csv(run_dir / "table_ablation.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(f"\nreference configuration: {ref}\n")
        print("Table 1: cross-device agreement")
        print(
            main[
                [
                    "config_id",
                    "backend",
                    "thermal_state",
                    "gar_vs_ref_pct",
                    "worst_pairwise_gar_pct",
                    "median_first_divergence",
                    "top1_flip_rate_pct",
                    "intra_device_flakiness_pct",
                    "task_accuracy_pct",
                ]
            ].round(2).to_string(index=False)
        )
        print("\nTable 2: ablation")
        print(ablation.round(2).to_string(index=False))

    flaky = main["intra_device_flakiness_pct"].max()
    if pd.notna(flaky) and flaky > 0:
        print(
            f"\nWARNING: intra-device flakiness is {flaky:.1f}%, not zero. "
            "A configuration that disagrees with itself cannot support a "
            "cross-device claim. Investigate that before reading anything else."
        )
    print(f"\nwrote {run_dir / 'table_main.csv'}")
    print(f"wrote {run_dir / 'table_ablation.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/latest")
    parser.add_argument(
        "--reference",
        default=None,
        help="config_id every other configuration is compared against",
    )
    args = parser.parse_args(argv)
    evaluate(Path(args.run_dir), args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
