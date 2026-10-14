"""Thermal state control and verification.

Thermal state is a first-class experimental condition here, not a nuisance.
The hypothesis the paper tests is sharp: heat by itself does not change
arithmetic, but a thermal governor that migrates work off the NPU does.
Testing that requires knowing which state a measurement was taken in, which
means reading the sensors rather than trusting a stopwatch.

On Android, thermal zones live under /sys/class/thermal/thermal_zone*/temp
and report millidegrees Celsius. Zone naming is vendor specific, so we read
every zone, keep the maximum, and record the full vector.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ThermalState(str, Enum):
    COLD = "cold"
    WARM = "warm"
    THROTTLED = "throttled"

    @classmethod
    def parse(cls, name: str) -> "ThermalState":
        try:
            return cls(name)
        except ValueError as exc:
            raise ValueError(
                f"unknown thermal state {name!r}; expected one of "
                f"{[s.value for s in cls]}"
            ) from exc


# Target windows in degrees Celsius for the hottest zone. These are starting
# points: measure your own device idle and saturated temperatures first, then
# set the windows from the data. A window copied from another phone is a
# window that silently mislabels your runs.
DEFAULT_WINDOWS: dict[ThermalState, tuple[float, float]] = {
    ThermalState.COLD: (0.0, 36.0),
    ThermalState.WARM: (36.0, 44.0),
    ThermalState.THROTTLED: (44.0, 120.0),
}


@dataclass
class ThermalReading:
    """One sensor sample, recorded alongside every trace."""

    state: ThermalState
    max_celsius: float
    zones: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    in_window: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "max_celsius": self.max_celsius,
            "zones": self.zones,
            "timestamp": self.timestamp,
            "in_window": self.in_window,
        }


def read_zones_host() -> dict[str, float]:
    """Read thermal zones on a Linux host. Returns an empty dict elsewhere."""
    from pathlib import Path

    zones: dict[str, float] = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            milli = int((zone / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        name = zone.name
        type_file = zone / "type"
        if type_file.exists():
            try:
                name = f"{zone.name}:{type_file.read_text().strip()}"
            except OSError:
                pass
        zones[name] = milli / 1000.0
    return zones


def read_zones_adb(adb_serial: str) -> dict[str, float]:
    """Read thermal zones on a phone over adb."""
    cmd = [
        "adb", "-s", adb_serial, "shell",
        "for z in /sys/class/thermal/thermal_zone*; do "
        'echo "$z $(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    zones: dict[str, float] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        path, kind, raw = parts[0], parts[1], parts[-1]
        try:
            value = float(raw)
        except ValueError:
            continue
        # Some vendors report degrees, others millidegrees.
        zones[f"{path.rsplit('/', 1)[-1]}:{kind}"] = (
            value / 1000.0 if abs(value) > 200 else value
        )
    return zones


def sample(
    target: ThermalState,
    reader: Callable[[], dict[str, float]] = read_zones_host,
    windows: dict[ThermalState, tuple[float, float]] | None = None,
) -> ThermalReading:
    """Take one reading and check it against the target window."""
    windows = windows or DEFAULT_WINDOWS
    zones = reader()
    hottest = max(zones.values()) if zones else float("nan")
    lo, hi = windows[target]
    ok = bool(zones) and lo <= hottest < hi
    return ThermalReading(
        state=target,
        max_celsius=hottest,
        zones=zones,
        timestamp=time.time(),
        in_window=ok,
    )


def reach_state(
    target: ThermalState,
    cfg,
    reader: Callable[[], dict[str, float]] = read_zones_host,
    load: Callable[[], None] | None = None,
    enforce: bool | None = None,
) -> ThermalReading:
    """Drive the device into ``target`` and verify that it got there.

    ``load`` is a callable that generates one burst of inference. For COLD we
    simply idle; for WARM and THROTTLED we call ``load`` repeatedly until the
    configured duration has elapsed. If ``enforce`` is set and the sensors
    disagree with the target window, this raises rather than silently
    recording a mislabelled cell.
    """
    enforce = cfg.enforce if enforce is None else enforce
    seconds = {
        ThermalState.COLD: cfg.cold_idle_seconds,
        ThermalState.WARM: cfg.warm_load_seconds,
        ThermalState.THROTTLED: cfg.throttled_load_seconds,
    }[target]

    deadline = time.time() + seconds
    if target is ThermalState.COLD:
        # Idling is the only way to cool a passively cooled device.
        while time.time() < deadline:
            time.sleep(min(5.0, max(0.0, deadline - time.time())))
    else:
        if load is None:
            raise ValueError(f"reaching {target.value} needs a load callable")
        while time.time() < deadline:
            load()

    reading = sample(target, reader=reader)
    if enforce and not reading.in_window:
        raise RuntimeError(
            f"device did not reach {target.value}: hottest zone is "
            f"{reading.max_celsius:.1f} C, expected "
            f"{DEFAULT_WINDOWS[target]}. Recording this cell would mislabel it."
        )
    return reading
