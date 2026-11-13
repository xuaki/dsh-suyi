# -*- coding: utf-8 -*-
"""端到端：mnemosyne（权威源）+ Milvus（向量索引）+ 人格三层。

前置：mnemosyne 跑在 127.0.0.1:8765，Milvus 跑在 127.0.0.1:19530。
"""
from __future__ import annotations

import sys
import time

from persona_memory.embedding import HashingEmbedder
from persona_memory.milvus_store import MilvusVectorStore
from persona_memory.mnemosyne_client import MnemosyneClient
from persona_memory.service import PersonaMemoryService

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def main() -> int:
    collection = "persona_memory_e2e"
    mn = MnemosyneClient()
    print("mnemosyne health:", mn.health())

    vs = MilvusVectorStore(collection=collection)
    # 干净起点
    try:
        from pymilvus import MilvusClient

        c = MilvusClient(uri="http://127.0.0.1:19530")
        if collection in c.list_collections():
            c.drop_collection(collection)
            print("dropped old e2e collection")
    except Exception as e:
        print("warn: cannot drop old collection:", e)

    svc = PersonaMemoryService(mnemosyne=mn, vectors=vs)
    print("vector_up:", svc._vector_up)
    check("vector layer up", svc._vector_up)

    # ---- 1. 观察：情绪 + 事实 ----
    r1 = svc.observe("今天项目终于跑通了，太好了！我太开心了", source_kind="user_statement", tier="engage", strength=0.9)
    print("observe#1:", r1)
    check("obs1 admitted", r1.admitted, r1.reason)
    check("obs1 has observation id", bool(r1.observation_id))
    check("obs1 has memory id", bool(r1.memory_id))
    check("obs1 indexed to milvus", r1.indexed)
    check("obs1 raised arousal", r1.persona_dynamic["arousal"] >= 0.0)

    r2 = svc.observe("我最近有点焦虑，记忆系统一直不稳定，怕它丢东西", source_kind="user_statement", tier="engage", strength=0.8)
    print("observe#2:", r2)
    check("obs2 admitted", r2.admitted)
    check("obs2 valence dropped (anxiety costs)", r2.persona_dynamic["valence"] < r1.persona_dynamic["valence"], f"{r1.persona_dynamic} -> {r2.persona_dynamic}")
    check("obs2 arousal rose", r2.persona_dynamic["arousal"] > r1.persona_dynamic["arousal"], f"{r1.persona_dynamic} -> {r2.persona_dynamic}")

    r3 = svc.observe("嗯", source_kind="dialog", tier="passive", strength=0.1)
    print("observe#3:", r3)
    check("trivial obs not admitted", r3.admitted is False, r3.reason)

    # ---- 2. 检索 ----
    got = svc.retrieve("记忆系统 焦虑 丢东西")
    print("retrieve:", [(i["memory_id"][:8], i["final"], i["source"]) for i in got["items"]])
    check("retrieve returns items", got["count"] >= 1)
    joined = " ".join(i["content"] for i in got["items"])
    check("retrieve finds anxiety memory", "焦虑" in joined, joined)
    check("retrieve reports source", all("source" in i for i in got["items"]))

    # ---- 3. 人格三层快照 ----
    snap = svc.snapshot()
    check("snapshot base layer", "base" in snap["persona"])
    check("snapshot shaped layer", "shaped" in snap["persona"])
    check("snapshot dynamic layer", "dynamic" in snap["persona"])
    check("molecules present", len(snap["persona"]["dynamic"]["molecules"]) == 15)
    check("vector stats", snap["vector"] is not None and snap["vector"]["rows"] >= 2, str(snap["vector"]))

    # ---- 4. 纠正后不再被召回 ----
    if r2.memory_id:
        svc.correct(r2.memory_id, note="用户说那是误会")
        got2 = svc.retrieve("记忆系统 焦虑 丢东西")
        ids = {i["memory_id"] for i in got2["items"]}
        check("invalidated memory not recalled", r2.memory_id not in ids, str(ids))

    # ---- 5. 去重（同一句话两次不应产生两条记录）----
    a = svc.observe("记住了，我喜欢喝冰美式", source_kind="user_statement", tier="explicit", strength=1.0)
    b = svc.observe("记住了，我喜欢喝冰美式", source_kind="user_statement", tier="explicit", strength=1.0)
    check("idempotent observe", a.memory_id == b.memory_id, f"{a.memory_id} vs {b.memory_id}")
    dup_rows = [m for m in mn.search(query="", limit=200) if str(m.get("id")) == str(a.memory_id)]
    check("memory stored exactly once", len(dup_rows) == 1, f"rows={len(dup_rows)}")

    # ---- 6. 晋升：跨天 + engage 才允许 ----
    prom = svc.promote_candidates(limit=20)
    print("promote:", {k: (len(v) if isinstance(v, list) else v) for k, v in prom.items()})
    check("promote ran", "promoted" in prom)

    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
