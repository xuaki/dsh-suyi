# -*- coding: utf-8 -*-
"""内存向量索引 + 字面相关性校验的回归测试（纯逻辑，不联网、不加载模型）。

覆盖 P0（2026-11-12）的三件事里最容易悄悄坏掉的部分：
  - 索引能建、能存、能直接读回（**不重新 embed**）
  - upsert / remove 之后矩阵和 id 表不错位
  - 索引挂了 / 语料取不到 → 降级为空，不抛异常
  - _literal_relevance 的中文 2-gram 与 ASCII token 切法
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

from persona_memory.embedding import HashingEmbedder
from persona_memory.service import _literal_relevance, _query_units
from persona_memory.vector_memory import MemoryVectorIndex

FAIL = 0
_TMP_DIRS = []


def _tmpdir(prefix: str) -> str:
    d = tempfile.mkdtemp(prefix=prefix)
    _TMP_DIRS.append(d)  # 跑完统一删，别在仓库根目录留一地碎片
    return d


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


class CountingEmbedder:
    def __init__(self) -> None:
        self.inner = HashingEmbedder()
        self.dim = self.inner.dim
        self.is_semantic = False
        self.model_name = "counting-hash"
        self.batch_calls = 0

    def encode(self, text: str):
        return self.inner.encode(text)

    def encode_batch(self, texts):
        self.batch_calls += 1
        return self.inner.encode_batch(texts)


ROWS = [
    {"memory_id": "m1", "content": "溯忆的向量索引今天重写了", "status": "active"},
    {"memory_id": "m2", "content": "Milvus 的 mixcoord 组件挂了", "status": "active"},
    {"memory_id": "m3", "content": "今天中午吃了拉面", "status": "active"},
]


def test_literal_relevance() -> None:
    print("[literal relevance]")
    check("中文整串命中", _literal_relevance("溯忆", "溯忆的检索") == 1.0)
    check("中文完全不沾", _literal_relevance("溯忆", "今天中午吃了拉面") == 0.0)
    # jieba 整词：『溯忆的记忆』→ 溯忆/记忆 两个词；只撞上「记忆」= 0.5
    check(
        "中文部分命中按比例打折",
        abs(_literal_relevance("溯忆的记忆", "记忆系统很重要") - 0.5) < 1e-9,
        str(_literal_relevance("溯忆的记忆", "记忆系统很重要")),
    )
    check(
        "中文全命中",
        _literal_relevance("溯忆的记忆", "溯忆的记忆很重要") == 1.0,
    )
    check("ASCII 不区分大小写", _literal_relevance("Milvus", "milvus 坏了") == 1.0)
    check("ASCII 完全无关", _literal_relevance("zzqx-9f3a", "今天中午吃了拉面") == 0.0)
    check("空 query 记 0", _literal_relevance("", "什么都行") == 0.0)
    check("单字丢弃", _query_units("忆") == [])
    check("整词切分", _query_units("溯忆") == ["溯忆"])
    check("混合切分", _query_units("Milvus 索引") == ["milvus", "索引"])
    check("连续语气词按整词", _query_units("哈哈哈") == ["哈哈哈"])


def test_build_search_persist() -> None:
    print("[build / search / persist]")
    tmp = _tmpdir("persona-vidx-")
    path = os.path.join(tmp, "vector_index.npz")
    emb = CountingEmbedder()
    idx = MemoryVectorIndex(path, emb)
    check("build 前 stats 未加载", idx.stats()["loaded"] is False)

    t0 = time.time()
    idx.build(ROWS)
    check("build 很快（3 条）", time.time() - t0 < 5.0)
    check("build 后 loaded", idx.stats()["loaded"] is True)
    check("build 后行数", idx.stats()["rows"] == 3, str(idx.stats()))
    check("build 落了盘", os.path.exists(path))

    hits = idx.search("溯忆的向量索引今天重写了", limit=1)
    check("search 命中自己", hits and hits[0]["memory_id"] == "m1", str(hits))
    check("search 带 content", hits and "溯忆" in hits[0]["content"])
    check("search 分数在 0..1", hits and 0.0 <= hits[0]["score"] <= 1.0)
    check("空 query 返回空", idx.search("   ", limit=3) == [])

    # 重新 __init__ → 直接从 npz 读，不该再 embed 一遍语料
    emb2 = CountingEmbedder()
    idx2 = MemoryVectorIndex(path, emb2)
    hits2 = idx2.search("Milvus 的 mixcoord 组件挂了", limit=1)
    check("重载命中自己", hits2 and hits2[0]["memory_id"] == "m2", str(hits2))
    check("重载没有重新 embed", emb2.batch_calls == 0, f"batch_calls={emb2.batch_calls}")
    check("重载保留 built_at", idx2.stats()["built_at"] == idx.stats()["built_at"])

    # 换成不同维度的向量器：旧文件必须作废（语义空间不能混）
    other = HashingEmbedder(dim=64)
    idx3 = MemoryVectorIndex(path, other)
    check("维度不符时返回空（等重建）", idx3.search("溯忆", limit=1) == [])

    print("[upsert / remove]")
    idx2.upsert("m4", "新加的一条：关于人格分子", status="candidate")
    check("upsert 后行数 +1", idx2.stats()["rows"] == 4)
    hit = idx2.search("人格分子", limit=1)
    check("upsert 的能搜到", hit and hit[0]["memory_id"] == "m4", str(hit))
    idx2.upsert("m4", "改过的一条：关于人格分子", status="candidate")
    check("upsert 幂等（不新增行）", idx2.stats()["rows"] == 4)
    idx2.remove("m4")
    check("remove 后行数 -1", idx2.stats()["rows"] == 3)
    check("remove 后搜不到", all(h["memory_id"] != "m4" for h in idx2.search("人格分子", limit=3)))
    check("remove 不存在的 id 不炸", (idx2.remove("nope") or True) is True)


def test_lazy_and_degrade() -> None:
    print("[lazy / degrade]")
    tmp = _tmpdir("persona-vidx-lazy-")
    path = os.path.join(tmp, "v.npz")
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        time.sleep(0.3)  # 模拟「embed 一遍要一会儿」：让「不阻塞」可被观测
        return ROWS

    idx = MemoryVectorIndex(path, CountingEmbedder(), loader=loader)
    first = idx.search("溯忆", limit=3)
    check("第一次 search 不阻塞（后台建库）", first == [] and idx.stats()["building"] is True, str(idx.stats()))
    for _ in range(100):
        time.sleep(0.05)
        if idx.stats()["loaded"] and not idx.stats()["building"]:
            break
    check("后台建完之后有结果", idx.stats()["rows"] == 3, str(idx.stats()))
    check("后台建完能搜到", bool(idx.search("溯忆", limit=3)))
    check("loader 只调用一次", calls["n"] == 1, str(calls))

    # loader 抛错 → 降级为空，不抛
    idx2 = MemoryVectorIndex(
        os.path.join(tmp, "bad.npz"),
        CountingEmbedder(),
        loader=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    check("坏 loader 的 search 不抛", idx2.search("溯忆", limit=3) == [])
    for _ in range(100):
        time.sleep(0.05)
        if idx2.stats()["loaded"]:
            break
    check("坏 loader 最终降级为空", idx2.stats()["loaded"] is True and idx2.stats()["rows"] == 0)

    # 没有 loader、也没有 npz → 空索引，不炸
    idx3 = MemoryVectorIndex(os.path.join(tmp, "none.npz"), CountingEmbedder())
    check("无 loader 无 npz 返回空", idx3.search("溯忆", limit=3) == [])


def main() -> int:
    try:
        test_literal_relevance()
        test_build_search_persist()
        test_lazy_and_degrade()
    finally:
        for d in _TMP_DIRS:
            shutil.rmtree(d, ignore_errors=True)
    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

