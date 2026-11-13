# -*- coding: utf-8 -*-
"""REST API 验证：用真正的 UTF-8 客户端测中文往返。"""
from __future__ import annotations

import json
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
    h = requests.get(f"{BASE}/health", timeout=10).json()
    print("health:", json.dumps(h, ensure_ascii=False))
    check("health ok", h.get("ok") is True)
    check("vector up", h.get("vector_up") is True)

    text = "老公说这周想去看海，他最近太累了"
    r = requests.post(
        f"{BASE}/observe",
        json={"text": text, "source_kind": "user_statement", "tier": "engage", "strength": 0.9},
        timeout=15,
    ).json()
    print("observe:", json.dumps(r, ensure_ascii=False))
    check("observe admitted", r.get("admitted") is True, str(r))
    check("observe indexed", r.get("indexed") is True)

    q = requests.post(f"{BASE}/retrieve", json={"query": "看海 累", "limit": 5}, timeout=15).json()
    print("retrieve:", json.dumps(q, ensure_ascii=False)[:700])
    check("retrieve count>0", q.get("count", 0) > 0)
    items = q.get("items", [])
    check("retrieve returns summaries", all(i.get("summary") for i in items), str(items)[:300])
    check(
        "default content is summary, not full text",
        all(len(i.get("content") or "") <= 49 for i in items),
        str(items)[:300],
    )
    joined = " ".join((i.get("summary") or i.get("content") or "") for i in items)
    check("chinese roundtrip intact", "看海" in joined, joined)
    check("mnemosyne authority up", q.get("degraded", {}).get("authority_down") is False, str(q.get("degraded")))

    qd = requests.post(f"{BASE}/retrieve", json={"query": "看海 累", "limit": 2, "detail": True}, timeout=15).json()
    check("detail returns full content", all(i.get("content") for i in qd.get("items", [])), str(qd)[:200])

    # 测试写进去的记忆要自己清掉——不然每跑一次测试，真实记忆库就多一条重复
    try:
        requests.post(
            f"{BASE}/feedback",
            json={"memory_id": r.get("memory_id", ""), "note": "test_api_cleanup"},
            timeout=10,
        )
    except Exception:
        pass

    p = requests.get(f"{BASE}/persona", timeout=10).json()
    check("persona base note readable", "先接住情绪" in json.dumps(p, ensure_ascii=False))
    check("persona has dynamic", "dynamic" in p)

    e = requests.post(f"{BASE}/persona/emotion", json={"releases": {"cortisol": 0.7, "oxytocin": 0.4}}, timeout=10).json()
    print("emotion:", json.dumps({k: e[k] for k in ("fatigue", "effects", "emotion")}, ensure_ascii=False)[:400])
    check("emotion applied", e["molecules"]["cortisol"]["level"] > 0.15)

    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
