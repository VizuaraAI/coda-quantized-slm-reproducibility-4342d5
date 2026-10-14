"""Typed configuration, loaded from YAML.

Application code never touches a raw dict. If you need a new knob, add a
field here so that every run manifest records it automatically.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Which checkpoint and which quantization recipe."""

    name: str = "toy-synthetic"
    hf_id: str | None = None
    gguf_path: str | None = None
    # Quantization recipe label, used verbatim in the results table.
    recipe: str = "int8-per-channel"
    bits: int = 8
    # Verified at load time on every backend. Empty string skips the check.
    sha256: str = ""


@dataclass
class DeviceConfig:
    """One physical or simulated execution target."""

    name: str
    soc: str = "host"
    backend: str = "ref"
    # adb serial for real phones; None for host-side backends.
    adb_serial: str | None = None
    # Backend-specific knobs, recorded in the manifest as given.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """The prompt suite."""

    tasks: list[str] = field(
        default_factory=lambda: ["gsm8k", "arc_easy", "mmlu", "ifeval"]
    )
    n_per_task: int = 8
    max_new_tokens: int = 48
    # Offline mode uses the built-in prompt list instead of downloading.
    offline: bool = True
    seed: int = 0


@dataclass
class ThermalConfig:
    """Thermal protocol. See thermal.py for how states are reached."""

    states: list[str] = field(default_factory=lambda: ["cold"])
    cold_idle_seconds: int = 600
    warm_load_seconds: int = 300
    throttled_load_seconds: int = 1200
    # Refuse to record a cell whose measured temperature misses the target.
    enforce: bool = False


@dataclass
class TraceConfig:
    """What we log per decoding step."""

    top_k: int = 8
    # Teacher forcing on the reference trace isolates single-step flip rate
    # from autoregressive amplification. Run both.
    teacher_forced: bool = False


@dataclass
class AuditConfig:
    """The whole experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    devices: list[DeviceConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    # Within-device repeats. Needed before any between-device claim.
    repeats: int = 3
    output_root: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, raw: Any) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys loudly."""
    if raw is None:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__} got unknown config keys: {sorted(unknown)}. "
            "Add a typed field in config.py rather than passing it through."
        )
    return cls(**raw)


def load_config(path: str | Path) -> AuditConfig:
    """Load and validate an audit config from YAML."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    devices = [_build(DeviceConfig, d) for d in raw.pop("devices", [])]
    nested = {
        "model": _build(ModelConfig, raw.pop("model", None)),
        "dataset": _build(DatasetConfig, raw.pop("dataset", None)),
        "thermal": _build(ThermalConfig, raw.pop("thermal", None)),
        "trace": _build(TraceConfig, raw.pop("trace", None)),
    }
    scalars = {k: raw.pop(k) for k in list(raw) if k in {"repeats", "output_root"}}
    if raw:
        raise ValueError(f"Unknown top-level config keys: {sorted(raw)}")
    return AuditConfig(devices=devices, **nested, **scalars)
