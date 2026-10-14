"""Quantized matrix-vector kernels with explicit, switchable numerics.

This module exists so that the three things that actually differ between a
Hexagon NPU, an Adreno GPU and an Arm CPU are visible in Python and can be
toggled from a config:

1. **Reduction order.** Floating-point addition is not associative, so a
   tiled accumulation with 32-wide tiles and a sequential accumulation over
   the full K dimension give different sums for the same inputs. Vendors
   pick tile shapes to suit their memory hierarchy, not to match each other.

2. **Scale granularity and rounding.** Per-channel and per-tensor scales are
   both legal readings of the same quantized checkpoint, and the point at
   which the scale is applied (before or after accumulation) changes the
   rounding. Chen (arXiv:2608.13756) localized cross-kernel INT8 divergence
   to exactly this stage rather than to accumulation.

3. **Activation quantization.** Weight-only kernels keep activations in
   float; NPU kernels usually quantize activations too, which introduces a
   rounding step that the CPU path never performs.

The code here is written to be read, not to be fast. Decode is a
matrix-vector product, so the loops are over K with vectorized inner work,
which is fast enough for the model sizes this harness traces.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_ACCUM_DTYPES = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
}


@dataclass
class QuantizedMatrix:
    """Integer codes plus dequantization scales. Produced once, never per device."""

    codes: np.ndarray  # shape (K, N), integer codes stored as int32
    scales: np.ndarray  # shape (N,) for per-channel, (1,) for per-tensor
    bits: int
    granularity: str  # "per_channel" or "per_tensor"

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape  # type: ignore[return-value]


@dataclass
class KernelSpec:
    """A named point in the space of legal implementations of one GEMV.

    Two specs that differ only in ``reduction`` compute the same mathematical
    function and are both correct. That is the whole problem.
    """

    name: str = "ref"
    reduction: str = "pairwise"  # pairwise | sequential | blocked | blocked_reverse
    accum_dtype: str = "float32"
    block: int = 32
    granularity: str = "per_channel"  # how this backend reads the scales
    act_bits: int | None = None  # None keeps activations in float
    pow2_scales: bool = False  # mitigation: constrain scales to powers of two
    notes: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reduction": self.reduction,
            "accum_dtype": self.accum_dtype,
            "block": self.block,
            "granularity": self.granularity,
            "act_bits": self.act_bits,
            "pow2_scales": self.pow2_scales,
            "notes": self.notes,
        }


def round_to_pow2(scales: np.ndarray) -> np.ndarray:
    """Snap each scale to the nearest power of two.

    Multiplying by a power of two is exact in binary floating point, so the
    scale-application stage stops contributing divergence. This is the mobile
    port of the mitigation in arXiv:2609.00363. It costs accuracy, because the
    representable range of each channel is now coarser, and measuring that
    cost is the point of the mitigation ablation.
    """
    scales = np.asarray(scales, dtype=np.float64)
    safe = np.where(scales > 0, scales, 1.0)
    return np.where(scales > 0, np.exp2(np.round(np.log2(safe))), scales).astype(
        np.float32
    )


def quantize(
    weight: np.ndarray, bits: int = 8, granularity: str = "per_channel"
) -> QuantizedMatrix:
    """Symmetric round-to-nearest quantization of a (K, N) weight matrix.

    This runs exactly once, on the host, and the result is what gets hashed
    and shipped. Every backend reads these same integer codes.
    """
    if granularity not in {"per_channel", "per_tensor"}:
        raise ValueError(f"unknown granularity {granularity!r}")
    weight = np.asarray(weight, dtype=np.float32)
    qmax = float(2 ** (bits - 1) - 1)

    if granularity == "per_channel":
        amax = np.abs(weight).max(axis=0)
    else:
        amax = np.array([np.abs(weight).max()])
    scales = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)

    codes = np.rint(weight / scales).clip(-qmax - 1, qmax).astype(np.int32)
    return QuantizedMatrix(codes=codes, scales=scales, bits=bits, granularity=granularity)


def _read_scales(qm: QuantizedMatrix, spec: KernelSpec) -> np.ndarray:
    """Re-derive the scales the way this backend chooses to read them.

    A per-tensor backend handed a per-channel checkpoint collapses the scales
    to a single value. This is a real and common mismatch: the checkpoint says
    per-channel, the runtime's fast path only supports per-tensor, and nobody
    reports it.
    """
    scales = qm.scales.astype(np.float32)
    if spec.granularity == "per_tensor" and scales.size > 1:
        scales = np.array([float(scales.max())], dtype=np.float32)
    if spec.pow2_scales:
        scales = round_to_pow2(scales)
    return scales


def _quantize_activations(x: np.ndarray, bits: int) -> np.ndarray:
    """Dynamic per-tensor activation quantization, then immediate dequant.

    An NPU does this for real in the integer domain. We model only its
    numerical effect, which is a rounding step the float path never performs.
    """
    amax = float(np.abs(x).max())
    if amax == 0.0:
        return x.astype(np.float32)
    qmax = float(2 ** (bits - 1) - 1)
    scale = np.float32(amax / qmax)
    return (np.rint(x / scale).clip(-qmax - 1, qmax) * scale).astype(np.float32)


def _sum_sequential(terms: np.ndarray) -> np.ndarray:
    """Accumulate along axis 0 strictly left to right, in the given dtype."""
    acc = np.zeros(terms.shape[1:], dtype=terms.dtype)
    for k in range(terms.shape[0]):
        acc = acc + terms[k]
    return acc


def _sum_blocked(terms: np.ndarray, block: int, reverse: bool) -> np.ndarray:
    """Tile the reduction: partial sums per tile, then sum the partials.

    This is what a real tiled GEMM does. ``reverse`` models a backend that
    walks its tiles in the opposite order, which is numerically different and
    equally correct.
    """
    partials = [
        _sum_sequential(terms[start : start + block])
        for start in range(0, terms.shape[0], block)
    ]
    stacked = np.stack(partials[::-1] if reverse else partials)
    return _sum_sequential(stacked)


def _reduce(terms: np.ndarray, spec: KernelSpec) -> np.ndarray:
    if spec.reduction == "pairwise":
        # numpy's default, a pairwise tree reduction
        return terms.sum(axis=0)
    if spec.reduction == "sequential":
        return _sum_sequential(terms)
    if spec.reduction == "blocked":
        return _sum_blocked(terms, spec.block, reverse=False)
    if spec.reduction == "blocked_reverse":
        return _sum_blocked(terms, spec.block, reverse=True)
    raise ValueError(f"unknown reduction {spec.reduction!r}")


def matvec(x: np.ndarray, qm: QuantizedMatrix, spec: KernelSpec) -> np.ndarray:
    """Compute ``x @ dequant(qm)`` the way ``spec`` says to compute it.

    Returns float32 regardless of the accumulator width, because that is what
    the next layer receives on every real stack.
    """
    if spec.accum_dtype not in _ACCUM_DTYPES:
        raise ValueError(f"unknown accum_dtype {spec.accum_dtype!r}")
    dtype = _ACCUM_DTYPES[spec.accum_dtype]

    x = np.asarray(x, dtype=np.float32)
    if spec.act_bits is not None:
        x = _quantize_activations(x, spec.act_bits)

    scales = _read_scales(qm, spec)
    codes = qm.codes.astype(dtype)

    # Scale applied before accumulation, which is the ordering that
    # arXiv:2608.13756 identifies as the divergence site. Applying it after
    # accumulation (the other legal choice) is the `late_scale` option.
    if spec.options.get("late_scale", False):
        terms = x[:, None].astype(dtype) * codes
        acc = _reduce(terms, spec)
        out = acc * scales.astype(dtype)
    else:
        terms = x[:, None].astype(dtype) * codes * scales.astype(dtype)
        out = _reduce(terms, spec)

    return np.asarray(out, dtype=np.float32)


# Named kernel specs used by the simulated backends. Each one is a plausible
# mobile implementation, and no two of them are bit-identical.
KERNEL_PRESETS: dict[str, KernelSpec] = {
    "ref": KernelSpec(
        name="ref",
        reduction="sequential",
        accum_dtype="float64",
        granularity="per_channel",
        notes="host FP64 reference, stands in for the ground-truth trace",
    ),
    "cpu_neon": KernelSpec(
        name="cpu_neon",
        reduction="blocked",
        accum_dtype="float32",
        block=16,
        granularity="per_channel",
        notes="Arm NEON weight-only path, 16-wide tiles, float accumulate",
    ),
    "cpu_i8mm": KernelSpec(
        name="cpu_i8mm",
        reduction="blocked",
        accum_dtype="float32",
        block=64,
        granularity="per_channel",
        act_bits=8,
        notes="Arm i8mm dot-product path, wider tiles, quantized activations",
    ),
    "gpu_opencl": KernelSpec(
        name="gpu_opencl",
        reduction="blocked_reverse",
        accum_dtype="float32",
        block=32,
        granularity="per_channel",
        notes="GPU path, tiles walked in the opposite order",
    ),
    "npu_pertensor": KernelSpec(
        name="npu_pertensor",
        reduction="blocked",
        accum_dtype="float32",
        block=32,
        granularity="per_tensor",
        act_bits=8,
        notes="NPU fast path: collapses scales to per-tensor, quantizes activations",
        options={"late_scale": True},
    ),
    "npu_fp16_accum": KernelSpec(
        name="npu_fp16_accum",
        reduction="blocked",
        accum_dtype="float16",
        block=32,
        granularity="per_channel",
        act_bits=8,
        notes="NPU path with a 16-bit accumulator, the most aggressive setting",
    ),
}


def preset(name: str, **overrides: Any) -> KernelSpec:
    """Fetch a named kernel spec, optionally overriding fields.

    Overriding is how the mitigation ablation works: the same backend is run
    once as shipped and once with ``pow2_scales=True``.
    """
    if name not in KERNEL_PRESETS:
        raise KeyError(f"unknown kernel preset {name!r}; have {sorted(KERNEL_PRESETS)}")
    return dataclasses.replace(KERNEL_PRESETS[name], **overrides)
