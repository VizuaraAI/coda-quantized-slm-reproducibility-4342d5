"""The prompt suite.

Two modes. ``offline=True`` uses the built-in prompt list, so the smoke test
and the metrics development loop need no network and no download. Flip it to
False and the same tasks are drawn from the public datasets through the
``datasets`` library.

Stratification matters here. Divergence depends on how many near-tie
decoding steps a prompt contains, and that varies enormously between a
single-token multiple-choice answer and a multi-step arithmetic chain. A
prompt suite that is all one shape will give you a confidently wrong headline
number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Task name to the HF dataset spec used when offline is False.
HF_SPECS: dict[str, dict[str, str]] = {
    "gsm8k": {"path": "gsm8k", "name": "main", "split": "test"},
    "arc_easy": {"path": "allenai/ai2_arc", "name": "ARC-Easy", "split": "test"},
    "mmlu": {"path": "cais/mmlu", "name": "all", "split": "test"},
    "triviaqa": {"path": "mandarjoshi/trivia_qa", "name": "rc.nocontext", "split": "validation"},
    "ifeval": {"path": "google/IFEval", "name": "default", "split": "train"},
}


@dataclass(frozen=True)
class Prompt:
    """One item of the suite."""

    prompt_id: str
    task: str
    text: str
    answer: str = ""
    # "short" answers have few decoding steps and so few chances to diverge;
    # "chain" answers have many. Keep the label, it explains your results.
    shape: str = "short"


_BUILTIN: dict[str, list[tuple[str, str, str]]] = {
    "gsm8k": [
        (
            "A phone draws 3 watts while generating and runs for 7 minutes. "
            "How many watt-minutes does it use? Answer step by step.",
            "21",
            "chain",
        ),
        (
            "A model file is 1.8 GB at 4 bits per weight. Roughly how many "
            "billion weights does it hold? Answer step by step.",
            "3.6",
            "chain",
        ),
        (
            "Two phones each run 250 prompts. If 12 prompts disagree, what "
            "percentage agree? Answer step by step.",
            "95.2",
            "chain",
        ),
        (
            "A run takes 4 seconds per prompt across 2000 prompts and 5 "
            "repeats. How many hours is that? Answer step by step.",
            "11.1",
            "chain",
        ),
    ],
    "arc_easy": [
        ("Which unit measures electric current? A) volt B) ampere C) ohm", "B", "short"),
        ("Heat causes a metal bar to do what? A) shrink B) expand C) nothing", "B", "short"),
        ("Which is a conductor? A) rubber B) glass C) copper", "C", "short"),
        ("What does an NPU mainly accelerate? A) audio B) matrix math C) storage", "B", "short"),
    ],
    "mmlu": [
        (
            "In floating point arithmetic, addition is not: A) commutative "
            "B) associative C) defined",
            "B",
            "short",
        ),
        (
            "Symmetric int8 quantization of a weight matrix stores: A) codes "
            "and scales B) only codes C) only scales",
            "A",
            "short",
        ),
        (
            "Dynamic voltage and frequency scaling mainly responds to: "
            "A) network load B) junction temperature C) battery chemistry",
            "B",
            "short",
        ),
        (
            "Greedy decoding selects the token with the: A) highest logit "
            "B) lowest logit C) median logit",
            "A",
            "short",
        ),
    ],
    "triviaqa": [
        ("Which company designs the Hexagon digital signal processor?", "Qualcomm", "short"),
        ("What does the acronym NPU stand for?", "neural processing unit", "short"),
        ("Which instruction set extension adds int8 matrix multiply to Arm?", "i8mm", "short"),
        ("What file format does llama.cpp use for quantized weights?", "GGUF", "short"),
    ],
    "ifeval": [
        (
            "List exactly three sources of numerical divergence in quantized "
            "inference. Use a numbered list and no other text.",
            "",
            "format",
        ),
        (
            "Describe thermal throttling in exactly two sentences, with no "
            "bullet points.",
            "",
            "format",
        ),
        (
            "Name one mitigation for cross-kernel divergence. Reply with a "
            "single lowercase word.",
            "",
            "format",
        ),
        (
            "Summarize what a logit margin is, in under twenty words, as one "
            "sentence ending in a period.",
            "",
            "format",
        ),
    ],
}


def _builtin_prompts(task: str, n: int, rng: np.random.Generator) -> list[Prompt]:
    pool = _BUILTIN.get(task)
    if pool is None:
        raise KeyError(
            f"no built-in prompts for task {task!r}; have {sorted(_BUILTIN)}. "
            "Either add some, or set dataset.offline=False."
        )
    # Cycle rather than truncate, so n_per_task larger than the pool still
    # works for timing experiments. Repeated prompts are tagged so they are
    # never mistaken for independent samples.
    out: list[Prompt] = []
    for i in range(n):
        text, answer, shape = pool[i % len(pool)]
        suffix = "" if i < len(pool) else f" (pass {i // len(pool)})"
        out.append(
            Prompt(
                prompt_id=f"{task}-{i:03d}",
                task=task,
                text=text + suffix,
                answer=answer,
                shape=shape,
            )
        )
    rng.shuffle(out)  # type: ignore[arg-type]
    return out


def _hf_prompts(task: str, n: int, seed: int) -> list[Prompt]:
    from datasets import load_dataset

    spec = HF_SPECS.get(task)
    if spec is None:
        raise KeyError(f"no HF spec for task {task!r}; add one to HF_SPECS")
    ds = load_dataset(spec["path"], spec["name"], split=spec["split"])
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    out: list[Prompt] = []
    for i, row in enumerate(ds):
        if task == "gsm8k":
            text = row["question"] + "\nAnswer step by step."
            answer = row["answer"].split("####")[-1].strip()
            shape = "chain"
        elif task == "arc_easy":
            choices = row["choices"]
            opts = " ".join(
                f"{label}) {t}"
                for label, t in zip(choices["label"], choices["text"])
            )
            text = f"{row['question']} {opts}"
            answer = row["answerKey"]
            shape = "short"
        elif task == "mmlu":
            opts = " ".join(f"{c}) {t}" for c, t in zip("ABCD", row["choices"]))
            text = f"{row['question']} {opts}"
            answer = "ABCD"[int(row["answer"])]
            shape = "short"
        elif task == "triviaqa":
            text = row["question"]
            answer = row["answer"]["value"]
            shape = "short"
        else:  # ifeval and anything else with a bare prompt field
            text = row.get("prompt", "")
            answer = ""
            shape = "format"
        out.append(
            Prompt(
                prompt_id=f"{task}-{i:03d}",
                task=task,
                text=text,
                answer=answer,
                shape=shape,
            )
        )
    return out


def load_prompts(cfg: Any) -> list[Prompt]:
    """Build the stratified prompt suite described by a DatasetConfig."""
    rng = np.random.default_rng(cfg.seed)
    prompts: list[Prompt] = []
    for task in cfg.tasks:
        if cfg.offline:
            prompts.extend(_builtin_prompts(task, cfg.n_per_task, rng))
        else:
            prompts.extend(_hf_prompts(task, cfg.n_per_task, cfg.seed))
    return prompts


def shape_counts(prompts: list[Prompt]) -> dict[str, int]:
    """How the suite is stratified. Print this; an unbalanced suite misleads."""
    counts: dict[str, int] = {}
    for p in prompts:
        counts[p.shape] = counts.get(p.shape, 0) + 1
    return counts


# Imported late so the module's public surface stays readable.
from typing import Any  # noqa: E402
