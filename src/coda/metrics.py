"""Divergence metrics. Pure functions over traces, no IO and no config.

The metrics are deliberately layered, because conflating the layers is the
most common way to get this kind of study wrong:

* **Numerical**  : ``logit_error`` asks whether the arithmetic differed.
* **Behavioral** : ``top1_flip_rate``, ``greedy_agreement``,
  ``first_divergence_position`` ask whether the differing arithmetic changed
  a decision.
* **Task**       : ``accuracy_spread`` asks whether the changed decision
  changed the answer a user sees.

A result is only interesting when it survives all three layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracing import Trace


@dataclass
class PairDivergence:
    """Divergence between two traces of the same prompt."""

    prompt_id: str
    config_a: str
    config_b: str
    identical: bool
    first_divergence: int | None  # token index, None when identical
    compared_steps: int
    flipped_steps: int
    max_logit_error: float
    rel_logit_error: float
    margin_at_divergence: float | None

    @property
    def flip_rate(self) -> float:
        return self.flipped_steps / self.compared_steps if self.compared_steps else 0.0


def first_divergence_position(a: Trace, b: Trace) -> int | None:
    """Index of the first differing token, or None if the sequences agree.

    Early divergence is far more damaging than late divergence, because
    everything downstream is then conditioned on a different prefix. Reporting
    only a mean agreement rate hides this entirely, so we keep the position.
    """
    for i, (ta, tb) in enumerate(zip(a.token_ids, b.token_ids)):
        if ta != tb:
            return i
    if len(a.token_ids) != len(b.token_ids):
        return min(len(a.token_ids), len(b.token_ids))
    return None


def logit_error(a: Trace, b: Trace) -> tuple[float, float]:
    """Absolute and relative L-infinity error on the shared top-1 logits.

    Only steps before the first divergence are comparable: after a flip the
    two models are conditioned on different prefixes, so a large logit
    difference says nothing about the kernels.
    """
    cut = first_divergence_position(a, b)
    n = min(a.n_steps, b.n_steps) if cut is None else cut
    if n == 0:
        return 0.0, 0.0
    la = a.topk_logits[:n, 0].astype(np.float64)
    lb = b.topk_logits[:n, 0].astype(np.float64)
    abs_err = float(np.max(np.abs(la - lb)))
    denom = float(np.max(np.abs(la))) or 1.0
    return abs_err, abs_err / denom


def compare_pair(a: Trace, b: Trace) -> PairDivergence:
    """Full divergence record for one prompt across two configurations."""
    if a.prompt_id != b.prompt_id:
        raise ValueError(f"prompt mismatch: {a.prompt_id} vs {b.prompt_id}")
    cut = first_divergence_position(a, b)
    n = min(a.n_steps, b.n_steps)
    flipped = sum(1 for i in range(n) if a.token_ids[i] != b.token_ids[i])
    abs_err, rel_err = logit_error(a, b)
    margin = float(a.margins[cut]) if cut is not None and cut < a.n_steps else None
    return PairDivergence(
        prompt_id=a.prompt_id,
        config_a=a.config_id,
        config_b=b.config_id,
        identical=cut is None,
        first_divergence=cut,
        compared_steps=n,
        flipped_steps=flipped,
        max_logit_error=abs_err,
        rel_logit_error=rel_err,
        margin_at_divergence=margin,
    )


def greedy_agreement(pairs: list[PairDivergence]) -> float:
    """Greedy agreement rate (GAR): share of prompts that agree exactly, in percent.

    This is the headline number. It is what a product team means when it asks
    whether two phones give the same answer.
    """
    if not pairs:
        return float("nan")
    return 100.0 * sum(p.identical for p in pairs) / len(pairs)


def median_first_divergence(pairs: list[PairDivergence]) -> float:
    """Median first-divergence position over diverging prompts only.

    Averaging in the agreeing prompts as "infinity" would make the number
    meaningless, so they are excluded and reported separately via GAR.
    """
    positions = [p.first_divergence for p in pairs if p.first_divergence is not None]
    return float(np.median(positions)) if positions else float("nan")


def top1_flip_rate(pairs: list[PairDivergence]) -> float:
    """Per-step argmax disagreement rate, in percent.

    Compare this against ``greedy_agreement``: a tiny flip rate combined with
    a low agreement rate is the signature of autoregressive amplification,
    where one flipped token derails the rest of the generation.
    """
    steps = sum(p.compared_steps for p in pairs)
    if steps == 0:
        return float("nan")
    return 100.0 * sum(p.flipped_steps for p in pairs) / steps


def amplification_factor(pairs: list[PairDivergence]) -> float:
    """How much worse end-to-end disagreement is than per-step error predicts.

    Under independence, a per-step flip rate p over L steps would give a
    sequence disagreement of 1 - (1-p)^L. The ratio of the observed
    disagreement to that prediction is the amplification factor. A value above
    one means flips are not independent: they cascade.
    """
    flip = top1_flip_rate(pairs) / 100.0
    if not pairs or not np.isfinite(flip):
        return float("nan")
    mean_len = float(np.mean([p.compared_steps for p in pairs]))
    predicted = 1.0 - (1.0 - flip) ** mean_len
    observed = 1.0 - greedy_agreement(pairs) / 100.0
    if predicted <= 0:
        return float("nan")
    return observed / predicted


def margin_histogram(
    pairs: list[PairDivergence], bins: int = 10, hi: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of the top-1 to top-2 margin at the diverging step.

    Feeds Figure 2 panel B. If the mass sits in the lowest bins, divergence is
    a near-tie phenomenon and margin-guarded recomputation will be cheap.
    """
    margins = [
        p.margin_at_divergence
        for p in pairs
        if p.margin_at_divergence is not None and np.isfinite(p.margin_at_divergence)
    ]
    edges = np.linspace(0.0, hi, bins + 1)
    counts, _ = np.histogram(np.clip(margins, 0.0, hi), bins=edges)
    total = counts.sum()
    share = 100.0 * counts / total if total else counts.astype(float)
    return edges, share


def intra_device_flakiness(repeats: list[Trace]) -> float:
    """Share of repeat pairs on the SAME configuration that disagree, in percent.

    This is the control. If this is not near zero, the device itself is
    nondeterministic and no cross-device claim can be made until that is
    understood and either fixed or reported as its own finding.
    """
    if len(repeats) < 2:
        return 0.0
    disagreeing = 0
    total = 0
    for i in range(len(repeats)):
        for j in range(i + 1, len(repeats)):
            total += 1
            if first_divergence_position(repeats[i], repeats[j]) is not None:
                disagreeing += 1
    return 100.0 * disagreeing / total


def accuracy_spread(accuracy_by_config: dict[str, float]) -> float:
    """Maximum minus minimum task accuracy across the matrix, in points.

    The number that turns an arithmetic curiosity into a reproducibility
    problem: it is how much a user's measured correctness depends on which
    phone they happen to own.
    """
    if len(accuracy_by_config) < 2:
        return 0.0
    values = list(accuracy_by_config.values())
    return float(max(values) - min(values))


def exact_match(prediction: str, answer: str) -> bool:
    """Light normalization then exact match. Replace with per-task graders."""
    norm = lambda s: " ".join(s.strip().lower().split()).rstrip(".")  # noqa: E731
    return norm(prediction).endswith(norm(answer))


def guarded_recompute_cost(traces: list[Trace], tau: float) -> float:
    """Share of decoding steps whose margin falls below tau, in percent.

    This is the expected overhead of margin-guarded FP32 recomputation: the
    fraction of steps that have to be redone on the CPU. Sweep tau to get the
    reproducibility versus throughput curve in the mitigation ablation.
    """
    margins = np.concatenate([t.margins for t in traces]) if traces else np.array([])
    if margins.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(margins < tau))
