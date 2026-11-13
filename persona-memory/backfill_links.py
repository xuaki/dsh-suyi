# -*- coding: utf-8 -*-
"""给存量记忆补链接图（链式召回的 backfill）。

链式召回从「新记忆写入时建边」开始生效，但库里已有的记忆彼此还是孤岛。
这个脚本一次性把历史连起来，并且**双向建边**：
新记忆建的是「从现在指向过去」的单向边（写入路径要轻），
而历史记忆用这个脚本补成双向，这样从任何一条老记忆展开，都能走回最近聊过的事。

两类边：

  related_to       语义上像（同一件事换种说法 / 同一句话重复）。
                   相似度用双通道：sim = max(向量余弦相似度, bigram Jaccard)。
  temporal_neighbor 时间上紧挨着（默认 5 分钟内），基本就是同一轮对话里
                   你说的和我回的。这就是文章里的 Thread 概念——
                   同一串来回，天然属于同一条线。

用法：
    python backfill_links.py                 # dry-run：只看边数、分数分布
    python backfill_links.py --apply         # 真写回 mnemosyne
    python backfill_links.py --apply --threshold 0.55 --temporal-gap 300

没有 Milvus 也能跑（退化成纯 Jaccard），只是语义边会少一些。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone

from persona_memory.main import collection_from_env
from persona_memory.mnemosyne_client import MnemosyneClient
from persona_memory.service import (
    LINK_CANDIDATES,
    LINK_SIMILARITY,
    MAX_LINKS,
    RELATED_TO,
    _similarity,
    _summarize,
)

DEAD = {"invalidated", "superseded", "rejected", "decayed", "archived"}
TEMPORAL_REL = "temporal_neighbor"
TEMPORAL_MAX = 2  # 每条记忆最多连几条时间邻居
DEFAULT_TEMPORAL_GAP = 300  # 秒


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真写回（默认 dry-run）")
    ap.add_argument("--threshold", type=float, default=LINK_SIMILARITY)
    ap.add_argument("--max-links", type=int, default=MAX_LINKS)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--temporal-gap", type=int, default=DEFAULT_TEMPORAL_GAP)
    ap.add_argument("--no-temporal", action="store_true", help="不建时间邻居边")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    mn = MnemosyneClient()
    try:
        rows = mn.search(query="", limit=args.limit)
    except Exception as e:  # noqa: BLE001
        print(f"mnemosyne 不可用: {e}")
        return 2

    nodes = []
    for m in rows:
        if str(m.get("status", "")).lower() in DEAD:
            continue
        content = str(m.get("content", "")).strip()
        mid = str(m.get("id", ""))
        if not content or not mid:
            continue
        nodes.append({"id": mid, "content": content, "created_at": m.get("created_at")})
    by_id = {n["id"]: n for n in nodes}
    print(f"扫描 {len(nodes)} 条存活记忆（阈值 {args.threshold}，语义边最多 {args.max_links} 条）")

    # 向量层可选：拿得到就做语义召回，拿不到就只靠字面
    vectors = None
    try:
        from persona_memory.milvus_store import MilvusVectorStore

        vectors = MilvusVectorStore(collection=collection_from_env())
        vectors.connect()
        vectors.ensure_collection()
        print("向量层可用：使用「向量 + Jaccard」双通道")
    except Exception as e:  # noqa: BLE001
        print(f"向量层不可用，退化为纯 Jaccard：{e}")

    # ---------------- 语义边 ----------------
    semantic: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for n in nodes:
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        if vectors is not None:
            try:
                for h in vectors.search(n["content"], limit=LINK_CANDIDATES):
                    if not h.memory_id or h.memory_id == n["id"] or h.memory_id in seen:
                        continue
                    other = by_id.get(h.memory_id)
                    if other is None:  # Milvus 里的孤儿（记忆已删）
                        continue
                    seen.add(h.memory_id)
                    sim = max(float(h.score), _similarity(n["content"], other["content"]))
                    if sim >= args.threshold:
                        scored.append((sim, other["id"]))
            except Exception as e:  # noqa: BLE001
                print(f"  ! 向量检索失败（{n['id']}）：{e}")
        if not scored:
            for other in nodes:
                if other["id"] == n["id"]:
                    continue
                sim = _similarity(n["content"], other["content"])
                if sim >= args.threshold:
                    scored.append((sim, other["id"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        semantic[n["id"]] = scored[: args.max_links]

    # 语义边双向：A→B 有了就补 B→A
    for src, lst in list(semantic.items()):
        for sim, dst in lst:
            if dst not in by_id:
                continue
            back = semantic.get(dst, [])
            if any(t == src for _, t in back):
                continue
            back.append((round(sim, 3), src))
            back.sort(key=lambda x: x[0], reverse=True)
            semantic[dst] = back[: args.max_links]

    # ---------------- 时间邻居边 ----------------
    temporal: dict[str, list[str]] = defaultdict(list)
    if not args.no_temporal:
        ordered = sorted(nodes, key=lambda n: _ts(n.get("created_at")))
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            gap = _ts(b.get("created_at")) - _ts(a.get("created_at"))
            if 0 <= gap <= args.temporal_gap:
                temporal[a["id"]].append(b["id"])
                temporal[b["id"]].append(a["id"])
        for k in list(temporal.keys()):
            temporal[k] = temporal[k][:TEMPORAL_MAX]

    # ---------------- 汇总 ----------------
    sem_edges = sum(len(v) for v in semantic.values())
    tmp_edges = sum(len(v) for v in temporal.values())
    sem_covered = sum(1 for v in semantic.values() if v)
    tmp_only = sum(1 for k, v in temporal.items() if v and not semantic.get(k))
    covered = len({k for k, v in semantic.items() if v} | {k for k, v in temporal.items() if v})
    print(f"语义边 {sem_edges} 条（覆盖 {sem_covered} 条）；时间邻居边 {tmp_edges} 条"
          f"（其中 {tmp_only} 条靠它才不成孤岛）")
    print(f"合并后覆盖 {covered}/{len(nodes)} 条记忆")
    scores = sorted((s for v in semantic.values() for s, _ in v), reverse=True)
    if scores:
        n = len(scores)
        print(f"语义相似度：最高 {scores[0]:.3f} / 中位 {scores[n // 2]:.3f} / 最低 {scores[-1]:.3f}")

    if args.verbose:
        for mid in by_id:
            parts = [f"{t}({s:.2f})" for s, t in semantic.get(mid, [])]
            parts += [f"{t}(time)" for t in temporal.get(mid, [])]
            if parts:
                print(f"  {mid} -> " + ", ".join(parts))

    if not args.apply:
        print("这是 dry-run。确认合理后加 --apply 真写。")
        return 0

    written = 0
    for n in nodes:
        links = []
        for sim, dst in semantic.get(n["id"], []):
            if dst not in by_id:
                continue
            links.append({
                "rel": RELATED_TO,
                "target_id": dst,
                "summary": _summarize(by_id[dst]["content"]),
                "similarity": round(sim, 3),
                "via": "backfill",
            })
        for dst in temporal.get(n["id"], []):
            if dst not in by_id:
                continue
            links.append({
                "rel": TEMPORAL_REL,
                "target_id": dst,
                "summary": _summarize(by_id[dst]["content"]),
                "via": "backfill",
            })
        if not links:
            continue
        try:
            cur = mn.get_memory(n["id"])
            meta = dict(cur.get("metadata") or {})
            meta["links"] = links  # 重建，不追加：脚本要可重复跑
            meta["links_backfilled_at"] = _now()
            mn.update_memory(n["id"], metadata=meta)
            written += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {n['id']} 写回失败: {e}")
    print(f"已写回 {written} 条记忆")
    return 0


if __name__ == "__main__":
    sys.exit(main())
