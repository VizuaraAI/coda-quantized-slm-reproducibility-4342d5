"""Per-step logit traces: the unit of evidence in this project.

A trace records, for one prompt on one configuration, the token chosen at
each decoding step and the top-k logits that justified it. Text-only logs
cannot distinguish "the logits differed and nothing happened" from "the
logits differed and the answer flipped", so we never log text alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trace:
    """One prompt, one configuration, one run."""

    prompt_id: str
    config_id: str
    repeat: int
    token_ids: list[int]
    # (steps, k) logit values and the ids they belong to, in descending order.
    topk_logits: np.ndarray
    topk_ids: np.ndarray
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.topk_logits = np.asarray(self.topk_logits, dtype=np.float32)
        self.topk_ids = np.asarray(self.topk_ids, dtype=np.int64)
        if self.topk_logits.shape != self.topk_ids.shape:
            raise ValueError("topk_logits and topk_ids must have the same shape")
        if len(self.token_ids) != self.topk_logits.shape[0]:
            raise ValueError(
                f"{len(self.token_ids)} tokens but "
                f"{self.topk_logits.shape[0]} logit rows"
            )

    @property
    def n_steps(self) -> int:
        return len(self.token_ids)

    @property
    def margins(self) -> np.ndarray:
        """Top-1 minus top-2 logit at each step.

        The central quantity of the analysis. A step with a large margin is
        robust to numerical perturbation; a step with a near-zero margin is a
        coin flip that the hardware decides.
        """
        if self.topk_logits.shape[1] < 2:
            return np.full(self.n_steps, np.inf, dtype=np.float32)
        return self.topk_logits[:, 0] - self.topk_logits[:, 1]

    def key(self) -> str:
        return f"{self.config_id}::{self.prompt_id}::r{self.repeat}"


def trace_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "traces"


def save_trace(run_dir: str | Path, trace: Trace) -> Path:
    """Write one trace as a single compressed npz, metadata included."""
    out = trace_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = trace.key().replace("::", "__").replace("/", "_")
    path = out / f"{safe}.npz"
    np.savez_compressed(
        path,
        topk_logits=trace.topk_logits,
        topk_ids=trace.topk_ids,
        token_ids=np.asarray(trace.token_ids, dtype=np.int64),
        header=json.dumps(
            {
                "prompt_id": trace.prompt_id,
                "config_id": trace.config_id,
                "repeat": trace.repeat,
                "text": trace.text,
                "meta": trace.meta,
            }
        ),
    )
    return path


def load_trace(path: str | Path) -> Trace:
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        return Trace(
            prompt_id=header["prompt_id"],
            config_id=header["config_id"],
            repeat=int(header["repeat"]),
            token_ids=[int(t) for t in data["token_ids"]],
            topk_logits=data["topk_logits"],
            topk_ids=data["topk_ids"],
            text=header.get("text", ""),
            meta=header.get("meta", {}),
        )


def load_run(run_dir: str | Path) -> list[Trace]:
    """Load every trace in a run directory."""
    paths = sorted(trace_dir(run_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no traces under {trace_dir(run_dir)}")
    return [load_trace(p) for p in paths]


def index_by_config(traces: list[Trace]) -> dict[str, dict[str, list[Trace]]]:
    """Group traces as ``config_id -> prompt_id -> [repeats]``."""
    index: dict[str, dict[str, list[Trace]]] = {}
    for tr in traces:
        index.setdefault(tr.config_id, {}).setdefault(tr.prompt_id, []).append(tr)
    for prompts in index.values():
        for repeats in prompts.values():
            repeats.sort(key=lambda t: t.repeat)
    return index


def file_sha256(path: str | Path) -> str:
    """Hash an artifact. Called on every backend load, not just once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    """Hash in-memory weights, for the synthetic model that has no file."""
    digest = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()
