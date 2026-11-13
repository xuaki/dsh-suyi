# -*- coding: utf-8 -*-
"""最终验收：所有测试 + 数据卫生 + 服务状态。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = Path(sys.executable)

SUITES = [
    ("smoke   (分子/人格/准入)", "test_smoke.py"),
    ("emotion (情绪误判修正)", "test_emotion.py"),
    ("persist (人格持久化)", "test_persist.py"),
    ("e2e     (端到端)", "test_e2e.py"),
    ("api     (REST/中文)", "test_api.py"),
    ("garbled (乱码闸门)", "test_garbled.py"),
]


def main() -> int:
    print("=" * 60)
    print("persona-memory 最终验收")
    print("=" * 60)
    failed = []
    for label, script in SUITES:
        out = subprocess.run(
            [str(PY), str(ROOT / script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
        )
        result = [ln for ln in out.stdout.splitlines() if "RESULT" in ln]
        tail = result[-1] if result else f"NO RESULT (exit {out.returncode})"
        print(f"  {label:<28} {tail}")
        if "ALL PASS" not in tail:
            failed.append(script)

    print("\n--- 数据卫生 ---")
    for script in ("cleanup_junk.py", "rebuild_index.py"):
        out = subprocess.run(
            [str(PY), str(ROOT / script), "--apply"] if script == "cleanup_junk.py" else [str(PY), str(ROOT / script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
        )
        tail = [ln for ln in out.stdout.splitlines() if ln.strip()][-2:]
        print(f"  {script}: " + " | ".join(tail))

    print("\n--- 结论 ---")
    if failed:
        print(f"  FAILED: {failed}")
        return 1
    print("  全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
