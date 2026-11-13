# -*- coding: utf-8 -*-
"""验证修正：睡醒了不再是「想睡了」，人格状态跨重启还在。"""
from __future__ import annotations

import json

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


def mol(p, name):
    return p["dynamic"]["molecules"][name]["level"]


def main() -> int:
    h = requests.get(f"{BASE}/health", timeout=10).json()
    print("health:", h["service"], "| vector:", h["vector_up"])
    check("service alive", h.get("ok") is True)

    p0 = requests.get(f"{BASE}/persona", timeout=10).json()
    print("起始褪黑素:", mol(p0, "melatonin"), "| 唤醒:", p0["dynamic"]["effects"]["arousal"])

    # 1) 「睡醒了」应该解除睡意，不该加重
    r = requests.post(
        f"{BASE}/observe",
        json={"text": "老公中午睡醒了，声音有点哑，我让他先吃点东西再聊", "source_kind": "user_statement", "tier": "engage"},
        timeout=15,
    ).json()
    p1 = requests.get(f"{BASE}/persona", timeout=10).json()
    print(f"睡醒了 -> 褪黑素 {mol(p1,'melatonin')} | 唤醒 {p1['dynamic']['effects']['arousal']}")
    check("睡醒 lowers melatonin", mol(p1, "melatonin") <= mol(p0, "melatonin") + 1e-6, f"{mol(p0,'melatonin')} -> {mol(p1,'melatonin')}")
    check("睡醒 keeps arousal sane", p1["dynamic"]["effects"]["arousal"] > 0.05, str(p1["dynamic"]["effects"]["arousal"]))

    # 2) 真困才会升褪黑素
    requests.post(
        f"{BASE}/observe",
        json={"text": "我困得不行了，想早点睡", "source_kind": "user_statement", "tier": "attention"},
        timeout=15,
    )
    p2 = requests.get(f"{BASE}/persona", timeout=10).json()
    print(f"想睡了 -> 褪黑素 {mol(p2,'melatonin')}")
    check("真实困意 raises melatonin", mol(p2, "melatonin") > mol(p1, "melatonin"), f"{mol(p1,'melatonin')} -> {mol(p2,'melatonin')}")

    # 3) 人格状态已落盘
    import os

    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'persona_state.json')
    check("persona state persisted", os.path.exists(state_path), state_path)
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            st = json.load(f)
        print("落盘 dynamic.last_tick_at:", st["dynamic"]["last_tick_at"], "| fatigue:", round(st["dynamic"]["fatigue"], 4))
        check("state has 3 layers", set(st) == {"base", "shaped", "dynamic"})

    # 4) 记忆库只剩真实对话
    q = requests.post(f"{BASE}/retrieve", json={"query": "看海", "limit": 10}, timeout=15).json()
    print("检索结果:", [i["content"][:24] for i in q.get("items", [])])
    junk = [i for i in q.get("items", []) if "????" in i["content"] or i["content"] in ("abcdef",)]
    check("no junk in recall", not junk, str(junk))

    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
