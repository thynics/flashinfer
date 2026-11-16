#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run explicitly selected GLM tests on eight B300 GPUs, with per-rank JUnit.

--plan is standard-library-only and never imports Torch or launches workers.
This runner acquires no resources and performs no automatic timeout, cleanup,
renewal or retry. An interrupted/failed run remains for its controller to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = Path(__file__).resolve().parent
LAZY = "test_lazy_first_exchange.py"


def selected_files(targets):
    files = sorted(p for p in TEST_DIR.glob("test_*.py") if p.name != "test_cpu.py")
    selected = set()
    for target in targets:
        path = Path(target)
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(ROOT / "tests") or not path.exists():
            raise ValueError(f"invalid existing test target: {target}")
        selected.update(p for p in files if p == path or p.is_relative_to(path))
    return [str(p.relative_to(ROOT)) for p in sorted(selected)]


def execution_plan(targets):
    files = selected_files(targets)
    lazy = [p for p in files if Path(p).name == LAZY]
    ordinary = [p for p in files if Path(p).name != LAZY]
    return [
        {
            "suite": name,
            "world_size": 8,
            "standalone": True,
            "files": group,
            "cuda_module_loading": "LAZY" if name == "lazy" else None,
        }
        for name, group in (("lazy", lazy), ("ordinary", ordinary))
        if group
    ]


def _check_hardware():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError(
            "explicit GLM TP8 scope requires eight visible assigned B300 GPUs"
        )
    devices = [torch.cuda.get_device_properties(i) for i in range(8)]
    if any(
        (p.major, p.minor, p.multi_processor_count) != (10, 3, 148)
        or "B300" not in p.name
        or "GB300" in p.name
        for p in devices
    ):
        raise RuntimeError(
            "explicit GLM TP8 scope requires eight B300 / SM103 / 148-SM devices"
        )
    if len({str(p.uuid) for p in devices}) != 8:
        raise RuntimeError("GLM TP8 scope requires eight distinct physical GPUs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", nargs="+", required=True)
    parser.add_argument("--junit-dir", type=Path, default=Path("junit/glm5-tp8"))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=("lazy", "ordinary"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = execution_plan(args.test_path)
    if args.plan:
        print(json.dumps({"groups": plan, "gpu_work_started": False}, indent=2))
        return 0
    if args.worker:
        if int(os.environ.get("WORLD_SIZE", "0")) != 8 or not args.suite:
            parser.error("worker mode requires the eight-rank torchrun launcher")
        import pytest

        args.junit_dir.mkdir(parents=True, exist_ok=True)
        path = args.junit_dir / f"{args.suite}-rank{int(os.environ['RANK'])}.xml"
        status = int(
            pytest.main(
                [
                    "-q",
                    "-x",
                    "-p",
                    "no:cacheprovider",
                    f"--junitxml={path}",
                    *selected_files(args.test_path),
                ]
            )
        )
        if os.environ.get("GLM5_TP8_UNFINISHED_CUDA") == "1":
            marker = (
                args.junit_dir
                / f"unfinished-{args.suite}-rank{os.environ['RANK']}.json"
            )
            marker.write_text(
                json.dumps(
                    {
                        "status": "UNKNOWN_GPU_COMPLETION",
                        "pytest_status": status,
                        "pid": os.getpid(),
                        "rank": int(os.environ["RANK"]),
                        "automatic_cleanup": False,
                        "controller_action_required": True,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"UNKNOWN GPU completion; worker retained for controller review: {marker}",
                flush=True,
            )
            # Returning a nonzero worker status would make torchrun terminate
            # siblings. Preserve this specific unresolved scene until its owner
            # explicitly decides disposition; no signal/retry/cleanup is sent.
            threading.Event().wait()
        return status
    if not plan:
        print("No GLM TP8 GPU tests selected; no workers launched.")
        return 0
    if os.environ.get("GLM5_TP8_UNFINISHED_CUDA"):
        raise RuntimeError(
            "unresolved GPU completion marker is set; controller disposition required"
        )
    _check_hardware()
    args.junit_dir.mkdir(parents=True, exist_ok=True)
    for group in plan:
        env = dict(os.environ, GLM5_TP8_REQUIRE_DISTRIBUTED="1")
        if group["cuda_module_loading"]:
            env["CUDA_MODULE_LOADING"] = group["cuda_module_loading"]
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=8",
            str(Path(__file__).resolve()),
            "--worker",
            "--suite",
            group["suite"],
            "--junit-dir",
            str(args.junit_dir.resolve()),
            "--test-path",
            *group["files"],
        ]
        print(
            f"GLM TP8 {group['suite']}: {len(group['files'])} selected files, eight ranks",
            flush=True,
        )
        status = subprocess.run(command, cwd=ROOT, env=env).returncode
        if status:
            print(
                f"GLM TP8 {group['suite']} failed with exit {status}; no retry or resource cleanup",
                flush=True,
            )
            return status if status > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
