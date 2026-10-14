"""Entry point: run the device x backend x thermal matrix and save traces.

    python -m coda.run_audit --config configs/default.yaml

Writes to runs/<timestamp>/ and repoints runs/latest. A run directory is
never overwritten, because a run you cannot go back to is a run you cannot
defend in review.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from . import backends, thermal
from .config import AuditConfig, load_config
from .datasets import Prompt, load_prompts, shape_counts
from .tracing import save_trace


def new_run_dir(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / time.strftime("%Y%m%d_%H%M%S")
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix}"
    run_dir.mkdir(parents=True)

    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        pass  # symlinks are not available everywhere; the timestamp still works
    return run_dir


def select_devices(cfg: AuditConfig, names: list[str] | None) -> list:
    if not names:
        return cfg.devices
    wanted = set(names)
    chosen = [d for d in cfg.devices if d.name in wanted or d.backend in wanted]
    missing = wanted - {d.name for d in chosen} - {d.backend for d in chosen}
    if missing:
        raise SystemExit(
            f"no device in the config matches {sorted(missing)}. "
            f"Configured devices: {[d.name for d in cfg.devices]}"
        )
    return chosen


def run(cfg: AuditConfig, device_filter: list[str] | None = None) -> Path:
    prompts: list[Prompt] = load_prompts(cfg.dataset)
    devices = select_devices(cfg, device_filter)
    if not devices:
        raise SystemExit("no devices configured; add entries under `devices`")

    states = [thermal.ThermalState.parse(s) for s in cfg.thermal.states]
    run_dir = new_run_dir(cfg.output_root)

    weights = None
    if any(d.backend in backends.kernels.KERNEL_PRESETS for d in devices):
        weights = backends.build_synthetic_weights(
            seed=cfg.dataset.seed, bits=cfg.model.bits
        )

    print(f"run directory : {run_dir}")
    print(f"prompts       : {len(prompts)}  {shape_counts(prompts)}")
    print(f"devices       : {[d.name for d in devices]}")
    print(f"thermal states: {[s.value for s in states]}")
    print(f"repeats       : {cfg.repeats}")

    manifest: dict[str, object] = {
        "config": cfg.to_dict(),
        "host": platform.platform(),
        "python": sys.version,
        "started": time.time(),
        "configurations": {},
        "prompts": [
            {"prompt_id": p.prompt_id, "task": p.task, "shape": p.shape,
             "answer": p.answer}
            for p in prompts
        ],
    }

    total = len(devices) * len(states) * len(prompts) * cfg.repeats
    bar = tqdm(total=total, desc="audit", unit="trace")

    for device in devices:
        backend = backends.build_backend(device, cfg.model, weights=weights)

        # The frozen-artifact invariant. A mismatch here means the device is
        # not running the model you think it is, and every number from it is
        # meaningless, so fail loudly rather than record it.
        if cfg.model.sha256:
            actual = backend.artifact_hash
            if actual and actual != cfg.model.sha256:
                raise SystemExit(
                    f"artifact hash mismatch on {device.name}: expected "
                    f"{cfg.model.sha256[:16]}..., loaded {actual[:16]}..."
                )

        for state in states:
            config_id = f"{device.name}@{state.value}"
            warm_up = lambda: backend.generate(  # noqa: E731
                prompts[0].text, max_new_tokens=8, top_k=1
            )
            reading = thermal.reach_state(
                state,
                cfg.thermal,
                reader=(
                    (lambda s=device.adb_serial: thermal.read_zones_adb(s))
                    if device.adb_serial
                    else thermal.read_zones_host
                ),
                load=warm_up,
                # Enforcement only makes sense on a real device with sensors.
                enforce=cfg.thermal.enforce and device.adb_serial is not None,
            )

            manifest["configurations"][config_id] = {  # type: ignore[index]
                "device": device.name,
                "soc": device.soc,
                "backend": device.backend,
                "thermal_state": state.value,
                "thermal_reading": reading.to_dict(),
                "describe": backend.describe(),
            }

            for repeat in range(cfg.repeats):
                for prompt in prompts:
                    trace = backend.generate(
                        prompt.text,
                        max_new_tokens=cfg.dataset.max_new_tokens,
                        top_k=cfg.trace.top_k,
                    )
                    trace = replace(
                        trace,
                        prompt_id=prompt.prompt_id,
                        config_id=config_id,
                        repeat=repeat,
                    )
                    trace.meta.update(
                        {
                            "task": prompt.task,
                            "shape": prompt.shape,
                            "answer": prompt.answer,
                            "thermal_state": state.value,
                            "thermal_max_c": reading.max_celsius,
                        }
                    )
                    save_trace(run_dir, trace)
                    bar.update(1)

    bar.close()
    manifest["finished"] = time.time()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote {total} traces and manifest.json to {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--backends",
        default=None,
        help="comma-separated device names or backend names to run",
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--n-per-task", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.repeats is not None:
        cfg.repeats = args.repeats
    if args.n_per_task is not None:
        cfg.dataset.n_per_task = args.n_per_task
    if args.max_new_tokens is not None:
        cfg.dataset.max_new_tokens = args.max_new_tokens

    device_filter = args.backends.split(",") if args.backends else None
    run(cfg, device_filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
