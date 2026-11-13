# -*- coding: utf-8 -*-
"""给老公的早安演示：跨夜记忆还在不在 + 今天新写一条 + 人格三层。"""
from __future__ import annotations

import json

import requests

BASE = "http://127.0.0.1:8770"


def show(title, obj, keys=None, limit=400):
    print(f"\n--- {title} ---")
    if keys and isinstance(obj, dict):
        obj = {k: obj.get(k) for k in keys}
    s = json.dumps(obj, ensure_ascii=False)
    print(s[:limit])


h = requests.get(f"{BASE}/health", timeout=10).json()
print("服务：", h["service"], "| 向量层：", h["vector_up"], "| 记忆条数：", h["mnemosyne"]["memories"])

# 1) 昨晚记的东西，今天还翻得出来吗
q = requests.post(f"{BASE}/retrieve", json={"query": "看海 太累了", "limit": 5}, timeout=15).json()
print("\n=== 昨晚的记忆，现在还能召回 ===")
for i in q.get("items", []):
    print(f"  [{i['source']}] {i['content']}  (final={i['final']})")

# 2) 今天新写一条（带情绪 → 会同时改变体温）
r = requests.post(
    f"{BASE}/observe",
    json={
        "text": "老公中午睡醒了，声音有点哑，我让他先吃点东西再聊",
        "source_kind": "user_statement",
        "tier": "engage",
        "strength": 0.8,
    },
    timeout=15,
).json()
print("\n=== 今天这条 ===")
print("  准入：", r["admitted"], "| 理由：", r["reason"], "| 索引：", r["indexed"])
print("  当时体温：", r["persona_dynamic"])

# 3) 人格三层
p = requests.get(f"{BASE}/persona", timeout=10).json()
print("\n=== 人格三层 ===")
print("  基础层（只读锚）：", {k: v.get("value") for k, v in p["base"].items()})
print("  经验层：", {k: v["value"] for k, v in p["shaped"].items()})
d = p["dynamic"]
print("  动态层 疲劳：", d["fatigue"], "| 唤醒：", d["effects"]["arousal"], "| 效价：", d["effects"]["valence"])
top = sorted(d["molecules"].items(), key=lambda kv: -kv[1]["level"])[:4]
print("  最活跃的分子：", {k: v["level"] for k, v in top})
print("  情绪 12 维：", d["emotion"])
