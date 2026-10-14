"""coda: Cross-Device Output Divergence Audit for quantized SLMs.

The package is organized around one abstraction, ``Backend``, and one data
object, ``Trace``. A backend turns a prompt into a trace of per-step top-k
logits. Everything else in the package either produces backends, or reduces
traces to numbers.
"""

__version__ = "0.1.0"

__all__ = [
    "backends",
    "config",
    "datasets",
    "kernels",
    "metrics",
    "thermal",
    "tracing",
]
