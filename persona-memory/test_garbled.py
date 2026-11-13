# -*- coding: utf-8 -*-
"""乱码闸门验证：写入闸门必须比读取容错更严。"""
from __future__ import annotations

import sys

import requests

BASE = "http://127.0.0.1:8770"
FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if cond:
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def main() -> int:
    def observe(text):
        return requests.post(
            f"{BASE}/observe", json={"text": text, "source_kind": "dialog"}, timeout=15
        ).json()

    cases = [
        ("???? ????", False, "rejected:garbled_input"),
        ("\ufffd\ufffd\ufffd\ufffd", False, "rejected:garbled_input"),
        ("嗯", False, "deny_rule"),
        ("abcdef", False, "deny_rule"),
        ("这是正常的中文句子，应该被接受", True, None),
        ("I write code at night, it is quiet", True, None),
    ]
    for text, expect_admit, expect_reason in cases:
        r = observe(text)
        ok = r["admitted"] is expect_admit and (expect_reason is None or r["reason"] == expect_reason)
        check(f"observe({text!r}) -> admitted={r['admitted']} reason={r['reason']}", ok)
        if expect_admit:
            check(f"  admitted has memory_id for {text[:6]!r}", bool(r.get("memory_id")))

    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
