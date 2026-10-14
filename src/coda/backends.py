"""Execution backends: the one abstraction the rest of the harness knows about.

A backend turns a prompt into a :class:`~coda.tracing.Trace`. Nothing outside
this module knows whether the tokens came from an NPU, a GPU, a CPU or a
simulation. Adding a phone means adding one class here plus one entry in the
config, and no changes anywhere else.

Three families live here:

``SyntheticBackend``
    A tiny recurrent model evaluated through :mod:`coda.kernels`. Runs
    everywhere with numpy alone and is how you develop the metrics before you
    own a device fleet. Different kernel presets stand in for different
    silicon, and they diverge for the same reasons real silicon diverges.

``HFReferenceBackend`` and ``LlamaCppBackend``
    Host-side real-model backends. These give you the FP32 reference trace and
    a real quantized trace, both on your laptop.

``LlamaCppAdbBackend``
    The device backend, left as a documented stub. Filling it in is the
    highest-value engineering task in the project.
"""

from __future__ import annotations

import abc
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import kernels
from .tracing import Trace, array_sha256


class Backend(abc.ABC):
    """Produce a logit trace for a prompt.

    Implementations must be deterministic given the same prompt. If a backend
    is not deterministic, that is itself a finding and belongs in the
    intra-device flakiness column, not hidden behind a seed.
    """

    config_id: str

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """Software bill of materials for this configuration."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int = 8,
        forced_tokens: Sequence[int] | None = None,
    ) -> Trace:
        """Decode greedily, logging top-k logits at every step.

        ``forced_tokens`` enables teacher forcing against a reference trace,
        which measures single-step flip rate without autoregressive
        amplification. Both modes are needed to separate the two effects.
        """

    @property
    def artifact_hash(self) -> str:
        """Hash of the weights this backend actually loaded."""
        return ""


# ---------------------------------------------------------------------------
# Synthetic model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticWeights:
    """Weights shared by every synthetic backend. Built once, never per device."""

    embed: np.ndarray  # (vocab, hidden), kept in float
    hidden: kernels.QuantizedMatrix  # (hidden, hidden)
    out: kernels.QuantizedMatrix  # (hidden, vocab)
    sha256: str


def build_synthetic_weights(
    seed: int = 0, hidden: int = 96, vocab: int = 256, bits: int = 8
) -> SyntheticWeights:
    """Construct the toy model, quantized once.

    The output matrix is built from near-duplicate column pairs whose
    separation is drawn log-uniformly between 1e-7 and 1e-2. This is not a
    trick: it reproduces the single most important property of a real
    quantized model's logit distribution, which is that many next-token
    candidates are nearly, and sometimes exactly, tied.

    Quantization collapses the closest pairs onto identical integer codes,
    because the separation falls below one quantization step. Those steps
    become exact ties, and an exact tie is maximally fragile: any difference
    in reduction order decides it. Real 4-bit checkpoints create exact ties
    the same way, which is why low-bit models are the ones that diverge.
    """
    rng = np.random.default_rng(seed)
    embed = (rng.standard_normal((vocab, hidden)) * 0.5).astype(np.float32)
    w_hidden = (rng.standard_normal((hidden, hidden)) / np.sqrt(hidden)).astype(
        np.float32
    )

    half = vocab // 2
    base = (rng.standard_normal((hidden, half)) / np.sqrt(hidden)).astype(np.float32)
    separation = 10.0 ** rng.uniform(-7.0, -2.0, size=half)
    perturb = (rng.standard_normal((hidden, half)) * separation).astype(np.float32)
    w_out = np.empty((hidden, vocab), dtype=np.float32)
    w_out[:, 0::2] = base
    w_out[:, 1::2] = base + perturb

    return SyntheticWeights(
        embed=embed,
        hidden=kernels.quantize(w_hidden, bits=bits, granularity="per_channel"),
        out=kernels.quantize(w_out, bits=bits, granularity="per_channel"),
        sha256=array_sha256(embed, w_hidden, w_out),
    )


class ByteTokenizer:
    """UTF-8 byte tokenizer. No vocabulary file, so nothing can drift."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class SyntheticBackend(Backend):
    """Toy model executed through a named kernel preset.

    Two instances built from the same weights but different presets are the
    simulation of two phones: identical artifact, different arithmetic.
    """

    def __init__(
        self,
        config_id: str,
        weights: SyntheticWeights,
        spec: kernels.KernelSpec,
        device_name: str = "host",
        soc: str = "simulated",
    ) -> None:
        self.config_id = config_id
        self.weights = weights
        self.spec = spec
        self.device_name = device_name
        self.soc = soc
        self.tokenizer = ByteTokenizer()

    @property
    def artifact_hash(self) -> str:
        return self.weights.sha256

    def describe(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "family": "synthetic",
            "device": self.device_name,
            "soc": self.soc,
            "kernel": self.spec.describe(),
            "artifact_sha256": self.weights.sha256,
            "host": platform.platform(),
            "numpy": np.__version__,
        }

    def _step(self, state: np.ndarray, token: int) -> tuple[np.ndarray, np.ndarray]:
        pre = kernels.matvec(state, self.weights.hidden, self.spec)
        state = np.tanh(pre + self.weights.embed[token % self.tokenizer.vocab_size])
        logits = kernels.matvec(state.astype(np.float32), self.weights.out, self.spec)
        return state.astype(np.float32), logits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int = 8,
        forced_tokens: Sequence[int] | None = None,
    ) -> Trace:
        ids = self.tokenizer.encode(prompt) or [0]
        state = np.zeros(self.weights.hidden.shape[0], dtype=np.float32)

        # Prefill.
        logits = np.zeros(self.tokenizer.vocab_size, dtype=np.float32)
        for token in ids:
            state, logits = self._step(state, token)

        chosen: list[int] = []
        topk_logits: list[np.ndarray] = []
        topk_ids: list[np.ndarray] = []

        for step in range(max_new_tokens):
            order = np.argsort(-logits, kind="stable")[:top_k]
            topk_ids.append(order.astype(np.int64))
            topk_logits.append(logits[order].astype(np.float32))

            # argsort with a stable sort breaks exact ties by token id, so the
            # tie-breaking rule is identical on every backend and cannot be
            # mistaken for a hardware effect.
            if forced_tokens is not None and step < len(forced_tokens):
                nxt = int(forced_tokens[step])
            else:
                nxt = int(order[0])
            chosen.append(nxt)
            state, logits = self._step(state, nxt)

        return Trace(
            prompt_id="",
            config_id=self.config_id,
            repeat=0,
            token_ids=chosen,
            topk_logits=np.stack(topk_logits),
            topk_ids=np.stack(topk_ids),
            text=self.tokenizer.decode(chosen),
            meta={"kernel": self.spec.name, "teacher_forced": forced_tokens is not None},
        )


# ---------------------------------------------------------------------------
# Real-model host backends
# ---------------------------------------------------------------------------


class HFReferenceBackend(Backend):
    """FP32 reference trace from a transformers checkpoint, on the host CPU.

    Single-threaded float32 on CPU is the closest thing to a ground truth that
    is reproducible on someone else's machine. Every device trace is compared
    against this.
    """

    def __init__(
        self,
        config_id: str,
        hf_id: str,
        dtype: str = "float32",
        device: str = "cpu",
        num_threads: int = 1,
    ) -> None:
        import torch  # imported lazily so the core harness needs no torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(num_threads)
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.config_id = config_id
        self.hf_id = hf_id
        self.device = device
        self.dtype_name = dtype
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype)
        ).to(device)
        self.model.eval()

    def describe(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "family": "transformers",
            "hf_id": self.hf_id,
            "dtype": self.dtype_name,
            "device": self.device,
            "torch": self.torch.__version__,
            "host": platform.platform(),
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int = 8,
        forced_tokens: Sequence[int] | None = None,
    ) -> Trace:
        torch = self.torch
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        chosen: list[int] = []
        topk_logits: list[np.ndarray] = []
        topk_ids: list[np.ndarray] = []
        past = None

        with torch.no_grad():
            step_input = ids
            for step in range(max_new_tokens):
                out = self.model(step_input, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float()
                values, indices = torch.topk(logits, k=top_k)
                topk_logits.append(values.cpu().numpy().astype(np.float32))
                topk_ids.append(indices.cpu().numpy().astype(np.int64))

                if forced_tokens is not None and step < len(forced_tokens):
                    nxt = int(forced_tokens[step])
                else:
                    nxt = int(indices[0])
                chosen.append(nxt)
                if nxt == self.tokenizer.eos_token_id:
                    break
                step_input = torch.tensor([[nxt]], device=self.device)

        return Trace(
            prompt_id="",
            config_id=self.config_id,
            repeat=0,
            token_ids=chosen,
            topk_logits=np.stack(topk_logits),
            topk_ids=np.stack(topk_ids),
            text=self.tokenizer.decode(chosen, skip_special_tokens=True),
            meta={"teacher_forced": forced_tokens is not None},
        )


class LlamaCppBackend(Backend):
    """Host-side GGUF trace through llama-cpp-python.

    Use this to confirm that a quantized artifact diverges from the FP32
    reference before you spend a week on adb. Set ``logits_all=True`` so the
    per-step logits are retrievable rather than only the sampled token.
    """

    def __init__(
        self,
        config_id: str,
        gguf_path: str,
        n_threads: int = 1,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
    ) -> None:
        from llama_cpp import Llama

        self.config_id = config_id
        self.gguf_path = gguf_path
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.model = Llama(
            model_path=gguf_path,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,
            verbose=False,
        )

    @property
    def artifact_hash(self) -> str:
        from .tracing import file_sha256

        return file_sha256(self.gguf_path)

    def describe(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "family": "llama.cpp",
            "gguf_path": self.gguf_path,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "artifact_sha256": self.artifact_hash,
            "host": platform.platform(),
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int = 8,
        forced_tokens: Sequence[int] | None = None,
    ) -> Trace:
        ids = self.model.tokenize(prompt.encode("utf-8"), add_bos=True)
        self.model.reset()
        self.model.eval(ids)

        chosen: list[int] = []
        topk_logits: list[np.ndarray] = []
        topk_ids: list[np.ndarray] = []

        for step in range(max_new_tokens):
            logits = np.asarray(self.model.scores[self.model.n_tokens - 1], np.float32)
            order = np.argsort(-logits, kind="stable")[:top_k]
            topk_ids.append(order.astype(np.int64))
            topk_logits.append(logits[order])

            if forced_tokens is not None and step < len(forced_tokens):
                nxt = int(forced_tokens[step])
            else:
                nxt = int(order[0])
            chosen.append(nxt)
            if nxt == self.model.token_eos():
                break
            self.model.eval([nxt])

        return Trace(
            prompt_id="",
            config_id=self.config_id,
            repeat=0,
            token_ids=chosen,
            topk_logits=np.stack(topk_logits),
            topk_ids=np.stack(topk_ids),
            text=self.model.detokenize(chosen).decode("utf-8", errors="replace"),
            meta={"teacher_forced": forced_tokens is not None},
        )


class LlamaCppAdbBackend(Backend):
    """Device backend over adb. Implement this in week 3.

    The contract is the same as every other backend, so once ``generate``
    returns a real Trace the entire downstream pipeline works unchanged.

    What you have to build, in order:

    1. Cross-compile ``llama-cli`` for ``arm64-v8a`` with the Android NDK, or
       take a prebuilt binary, and push it plus the GGUF to
       ``/data/local/tmp/coda/`` with ``adb push``.
    2. Verify the on-device SHA-256 of the GGUF against ``model.sha256`` from
       the config. ``adb shell sha256sum`` is enough. Do this on every run, not
       once: a partial push is silent and would invalidate everything.
    3. Run with a fixed prompt and a logit dump. Upstream llama.cpp does not
       print per-step logits by default, so you will either patch ``main.cpp``
       to write a binary dump of the top-k logits per step, or link against
       ``libllama`` through a small JNI or CLI shim. Budget a week for this
       step alone; the dump format is the hard part, not the plumbing.
    4. Pull the dump with ``adb pull`` and parse it into a Trace.

    Useful detail: pin the backend explicitly on the command line (threads,
    GPU layers, delegate) and record what you pinned in ``describe``. An
    unpinned run lets the runtime choose, and a run whose execution path you
    did not record is a run you cannot interpret.
    """

    def __init__(
        self,
        config_id: str,
        adb_serial: str,
        gguf_path_on_device: str,
        binary_path_on_device: str = "/data/local/tmp/coda/llama-cli",
        n_threads: int = 4,
        delegate: str = "cpu",
    ) -> None:
        self.config_id = config_id
        self.adb_serial = adb_serial
        self.gguf_path_on_device = gguf_path_on_device
        self.binary_path_on_device = binary_path_on_device
        self.n_threads = n_threads
        self.delegate = delegate

    def adb(self, *args: str, timeout: int = 120) -> str:
        """Run an adb command against this device and return stdout."""
        cmd = ["adb", "-s", self.adb_serial, *args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
        return result.stdout

    def device_artifact_sha256(self) -> str:
        """Hash the GGUF as it exists on the phone. Call this every run."""
        out = self.adb("shell", "sha256sum", self.gguf_path_on_device)
        return out.split()[0]

    def describe(self) -> dict[str, Any]:
        info = {
            "config_id": self.config_id,
            "family": "llama.cpp-adb",
            "adb_serial": self.adb_serial,
            "gguf_path_on_device": self.gguf_path_on_device,
            "n_threads": self.n_threads,
            "delegate": self.delegate,
        }
        for prop in ("ro.product.model", "ro.board.platform", "ro.build.fingerprint"):
            try:
                info[prop] = self.adb("shell", "getprop", prop).strip()
            except Exception as exc:  # device may be offline during a dry run
                info[prop] = f"unavailable: {exc}"
        return info

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int = 8,
        forced_tokens: Sequence[int] | None = None,
    ) -> Trace:
        raise NotImplementedError(
            "LlamaCppAdbBackend.generate is the week 3 task. See the class "
            "docstring for the four steps, and keep the return type a Trace so "
            "nothing downstream has to change."
        )


# ---------------------------------------------------------------------------
# Construction from config
# ---------------------------------------------------------------------------


def build_backend(device: Any, model: Any, weights: SyntheticWeights | None = None):
    """Instantiate the backend named by a DeviceConfig.

    ``device.backend`` is either a kernel preset name, which gives a synthetic
    backend, or one of the real-backend names below.
    """
    opts = dict(device.options)

    if device.backend in kernels.KERNEL_PRESETS:
        if weights is None:
            raise ValueError("synthetic backends need a SyntheticWeights bundle")
        spec = kernels.preset(device.backend, **opts)
        return SyntheticBackend(
            config_id=device.name,
            weights=weights,
            spec=spec,
            device_name=device.name,
            soc=device.soc,
        )

    if device.backend == "hf_reference":
        if not model.hf_id:
            raise ValueError("model.hf_id is required for the hf_reference backend")
        return HFReferenceBackend(config_id=device.name, hf_id=model.hf_id, **opts)

    if device.backend == "llamacpp_host":
        if not model.gguf_path:
            raise ValueError("model.gguf_path is required for llamacpp_host")
        return LlamaCppBackend(
            config_id=device.name, gguf_path=model.gguf_path, **opts
        )

    if device.backend == "llamacpp_adb":
        if not device.adb_serial:
            raise ValueError(f"device {device.name} needs an adb_serial")
        return LlamaCppAdbBackend(
            config_id=device.name,
            adb_serial=device.adb_serial,
            gguf_path_on_device=opts.pop(
                "gguf_path_on_device", "/data/local/tmp/coda/model.gguf"
            ),
            **opts,
        )

    raise ValueError(
        f"unknown backend {device.backend!r}. Kernel presets are "
        f"{sorted(kernels.KERNEL_PRESETS)}; real backends are "
        "hf_reference, llamacpp_host, llamacpp_adb."
    )
