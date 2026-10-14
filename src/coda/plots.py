"""Entry point: regenerate the paper figures from measured traces.

    python -m coda.plots --run-dir runs/latest

Produces the panels of Figure 2 from real data. The styling matches
out/paper/figures/make_figures.py in the kit, so a measured panel can be
dropped straight into the draft in place of the illustrative one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import metrics
from .evaluate import pairs_between, pick_reference
from .tracing import index_by_config, load_run

INK = "#1F2933"
MUTED = "#5A6472"
LINE = "#B6BDC8"
GRID = "#E6E9EE"
BLUE = "#2E5E9E"
GREY = "#8A93A0"
ACCENT = "#C2410C"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": LINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def panel_agreement(ax, index, reference) -> None:
    """Measured greedy agreement rate of every configuration, Figure 2 style."""
    configs = [c for c in sorted(index) if c != reference]
    values = [
        metrics.greedy_agreement(pairs_between(index, reference, c)) for c in configs
    ]
    colors = [GREY if "npu" in c.lower() else BLUE for c in configs]
    ax.bar(range(len(configs)), values, 0.62, color=colors)
    ax.set_xticks(range(len(configs)), configs, rotation=28, ha="right", fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Greedy Agreement Rate (%)")
    ax.set_title(f"A    Agreement vs. {reference}", fontsize=9.4, loc="left",
                 pad=9, fontweight="bold")
    tidy(ax)


def panel_margin(ax, index, reference) -> None:
    """Where flips happen, as a function of the top-1 to top-2 margin."""
    pairs: list[metrics.PairDivergence] = []
    for cid in sorted(index):
        if cid != reference:
            pairs.extend(pairs_between(index, reference, cid))
    edges, share = metrics.margin_histogram(pairs, bins=10, hi=2.0)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [ACCENT] * 3 + [GREY] * (len(centers) - 3)
    ax.bar(centers, share, width=(edges[1] - edges[0]) * 0.92, color=colors)
    ax.set_xlabel("top-1 minus top-2 logit margin at the diverging step")
    ax.set_ylabel("share of observed token flips (%)")
    ax.set_title("B    Divergence is a near-tie phenomenon", fontsize=9.4,
                 loc="left", pad=9, fontweight="bold")
    tidy(ax)


def panel_thermal(ax, index, reference) -> None:
    """Agreement grouped by thermal state, for each backend in the run."""
    states = ["cold", "warm", "throttled"]
    devices = sorted({c.split("@")[0] for c in index})
    width = 0.8 / max(len(devices), 1)

    for i, device in enumerate(devices):
        values = []
        for state in states:
            cid = f"{device}@{state}"
            if cid not in index or cid == reference:
                values.append(np.nan)
                continue
            values.append(metrics.greedy_agreement(pairs_between(index, reference, cid)))
        offsets = np.arange(len(states)) + (i - (len(devices) - 1) / 2) * width
        ax.bar(offsets, values, width * 0.9, label=device,
               color=GREY if "npu" in device.lower() else BLUE, alpha=0.85)

    ax.set_xticks(range(len(states)), states)
    ax.set_ylim(0, 105)
    ax.set_xlabel("thermal state")
    ax.set_ylabel("Greedy Agreement Rate (%)")
    ax.set_title("C    Agreement by thermal state", fontsize=9.4, loc="left",
                 pad=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    tidy(ax)
    ax.text(
        0.99, 0.03,
        "a flat profile with a step at the fallback point supports H4",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, style="italic",
        ha="right",
    )


PANELS = {
    "agreement": panel_agreement,
    "margin": panel_margin,
    "thermal": panel_thermal,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/latest")
    parser.add_argument("--reference", default=None)
    parser.add_argument(
        "--panel",
        default="all",
        choices=[*PANELS, "all"],
        help="render one panel, or all three side by side",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    index = index_by_config(load_run(run_dir))
    reference = pick_reference(sorted(index), args.reference)

    names = list(PANELS) if args.panel == "all" else [args.panel]
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 4.4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        PANELS[name](ax, index, reference)

    fig.suptitle(
        f"Measured cross-device divergence, run {run_dir.name}",
        fontsize=11, fontweight="bold", x=0.02, ha="left", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out) if args.out else run_dir / f"fig_{args.panel}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
