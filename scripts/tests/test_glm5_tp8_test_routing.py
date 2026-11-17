# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""No-GPU checks of the real GLM scope planner and internal CI enrollment."""

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
PATH = "tests/experimental/glm5_tp8_decode"
spec = importlib.util.spec_from_file_location(
    "glm_tp8_launcher", ROOT / PATH / "run_tp8.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class RoutingTests(unittest.TestCase):
    def test_non_dry_run_unsupported_hardware_skips_before_setup_or_workers(self):
        # Keep these tiny diagnostics; never delete a failure scene automatically.
        fixture = Path(tempfile.mkdtemp(prefix="flashinfer-glm-routing-"))
        stub = fixture / "python"
        calls = fixture / "calls.log"
        stub.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent("""\
                import os, sys
                from pathlib import Path
                from types import SimpleNamespace
                if len(sys.argv) != 3 or sys.argv[1] != '-c' or 'GLM5_TP8_HARDWARE_PROBE' not in sys.argv[2]:
                    raise SystemExit('unexpected interpreter/worker invocation')
                with Path(os.environ['GLM_ROUTE_CALLS']).open('a') as out:
                    out.write('probe\\n')
                major, minor, count = map(int, os.environ['GLM_ROUTE_HARDWARE'].split(','))
                cuda = SimpleNamespace(
                    is_available=lambda: count > 0, device_count=lambda: count,
                    get_device_properties=lambda i: SimpleNamespace(
                        major=major, minor=minor, multi_processor_count=148,
                        name='NVIDIA B300' if (major, minor) == (10, 3) else 'other GPU', uuid=str(i)))
                sys.modules['torch'] = SimpleNamespace(cuda=cuda)
                exec(compile(sys.argv[2], '<actual CI hardware probe>', 'exec'))
                """)
        )
        stub.chmod(0o700)
        guard = fixture / "bash_env.sh"
        guard.write_text("""source() {
  if [[ "$1" == */setup_test_env.sh ]]; then
    printf 'setup-reached\\n' >> "$GLM_ROUTE_CALLS"
    return 99
  fi
  builtin source "$@"
}
""")
        env = dict(
            os.environ,
            PYTHON=str(stub),
            PIP_CONSTRAINT="/dev/null",
            JUNIT_DIR=str(fixture / "unused-junit"),
            BASH_ENV=str(guard),
            GLM_ROUTE_CALLS=str(calls),
            PYTHONDONTWRITEBYTECODE="1",
        )
        env.pop("UNIT_TEST_RUN_STATUS_DIR", None)
        for hardware in ("0,0,0", "9,0,8", "10,0,8", "10,3,4"):
            with self.subTest(hardware=hardware):
                result = subprocess.run(
                    [
                        "bash",
                        "scripts/task_test_multi_gpu_comm_kernels.sh",
                        "--test-path",
                        PATH,
                    ],
                    cwd=ROOT,
                    env=env | {"GLM_ROUTE_HARDWARE": hardware},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("Skipping GLM TP8 suite", result.stdout)
                self.assertIn("No multi-GPU files overlap", result.stdout)
                self.assertNotIn("eight ranks", result.stdout)
        # Positive control: an eligible stub reaches the blocked setup boundary.
        # The guard prevents environment installation even if routing regresses.
        supported = subprocess.run(
            [
                "bash",
                "scripts/task_test_multi_gpu_comm_kernels.sh",
                "--test-path",
                PATH,
            ],
            cwd=ROOT,
            env=env | {"GLM_ROUTE_HARDWARE": "10,3,8"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(supported.returncode, 99)
        self.assertNotIn("Skipping GLM TP8 suite", supported.stdout)
        self.assertEqual(
            calls.read_text().splitlines(), ["probe"] * 5 + ["setup-reached"]
        )
        self.assertFalse((fixture / "unused-junit").exists())

    def test_unknown_gpu_completion_parks_worker_instead_of_exiting(self):
        class Parked(BaseException):
            pass

        reports = []
        event = SimpleNamespace(wait=Mock(side_effect=Parked))
        argv = [
            "run_tp8.py",
            "--worker",
            "--suite",
            "ordinary",
            "--test-path",
            f"{PATH}/test_moe.py",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(
                os.environ, WORLD_SIZE="8", RANK="3", GLM5_TP8_UNFINISHED_CUDA="1"
            ),
            patch.dict(sys.modules, pytest=SimpleNamespace(main=lambda args: 1)),
            patch.object(Path, "mkdir"),
            patch.object(
                Path, "write_text", lambda path, text: reports.append(json.loads(text))
            ),
            patch("builtins.print"),
            patch.object(launcher.threading, "Event", return_value=event),
            self.assertRaises(Parked),
        ):
            launcher.main()
        event.wait.assert_called_once()
        self.assertEqual(reports[0]["status"], "UNKNOWN_GPU_COMPLETION")
        self.assertEqual(reports[0]["rank"], 3)
        self.assertEqual(reports[0]["pytest_status"], 1)
        self.assertFalse(reports[0]["automatic_cleanup"])

    def test_real_internal_launcher_dry_run_needs_no_torch_or_setup(self):
        env = dict(
            os.environ,
            PYTHON=sys.executable,
            PIP_CONSTRAINT="/dev/null",
            JUNIT_DIR="/tmp/glm5-routing-plan-only",
            PYTHONDONTWRITEBYTECODE="1",
        )
        env.pop("UNIT_TEST_RUN_STATUS_DIR", None)
        for target, expected in (
            (PATH, True),
            ("tests/experimental/test_experimental_api.py", False),
        ):
            with self.subTest(target=target):
                result = subprocess.run(
                    [
                        "bash",
                        "scripts/task_test_multi_gpu_comm_kernels.sh",
                        "--dry-run",
                        "--test-path",
                        target,
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if expected:
                    payload, _ = json.JSONDecoder().raw_decode(
                        result.stdout[result.stdout.index("{\n") :]
                    )
                    self.assertEqual(
                        [g["world_size"] for g in payload["groups"]], [8, 8]
                    )
                    self.assertFalse(payload["gpu_work_started"])
                else:
                    self.assertIn("No multi-GPU files overlap", result.stdout)
                    self.assertNotIn("GLM TP8", result.stdout)

    def test_directory_has_one_lazy_and_one_eight_rank_ordinary_group(self):
        groups = launcher.execution_plan([PATH])
        self.assertEqual([g["suite"] for g in groups], ["lazy", "ordinary"])
        self.assertEqual(groups[0]["files"], [f"{PATH}/test_lazy_first_exchange.py"])
        self.assertEqual(groups[0]["cuda_module_loading"], "LAZY")
        self.assertTrue(all(g["world_size"] == 8 and g["standalone"] for g in groups))
        selected = [p for g in groups for p in g["files"]]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertNotIn(f"{PATH}/test_cpu.py", selected)
        self.assertIn(f"{PATH}/test_selector.py", selected)
        self.assertIn(f"{PATH}/test_moe_exceptional.py", selected)

    def test_exact_file_scope_is_not_broadened(self):
        target = f"{PATH}/test_moe.py"
        groups = launcher.execution_plan([target, target])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["files"], [target])
        self.assertEqual(launcher.execution_plan([f"{PATH}/test_cpu.py"]), [])
        self.assertEqual(launcher.execution_plan(["tests/comm"]), [])

    def test_invalid_or_foreign_paths_fail_before_launch(self):
        for target in ["tests/not_existing.py", "scripts", "../outside"]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                launcher.execution_plan([target])

    def test_internal_ci_routes_only_explicit_selected_glm_files(self):
        script = (ROOT / "scripts/task_test_multi_gpu_comm_kernels.sh").read_text()
        enrolled = (
            re.search(r'^GLM5_TP8_TEST_FILES="([^"]+)"$', script, re.M).group(1).split()
        )
        planned = {p for g in launcher.execution_plan([PATH]) for p in g["files"]}
        self.assertEqual(set(enrolled), planned)
        # Inspect the actual routing stanza, not a second implementation of its policy.
        self.assertRegex(
            script,
            r'if \[ -n "\$\{TEST_PATH:-\}" \]; then\s+GLM5_TP8_TEST_FILES="\$\(filter_files_by_test_path "\$GLM5_TP8_TEST_FILES"\)"\s+else\s+GLM5_TP8_TEST_FILES=""',
        )
        self.assertIn('"${glm5_command[@]}" --plan', script)
        self.assertIn('--test-path "${glm5_files[@]}"', script)


if __name__ == "__main__":
    unittest.main()
