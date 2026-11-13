# -*- coding: utf-8 -*-
"""清理重复记忆：同一段内容被记了两次（或多次）的情况。

怎么产生的（实测根因）：
    mnemosyne 的 `capture_observation` 会用后台 job 把 observation 自动转成
    memory，而它和我们自己的 `propose_memory` 之间存在时序缝——自动 job 还没
    跑完时 `_find_existing_memory` 找不到，于是我们又 propose 了一条。两条
    记忆最终共享同一个 `source_ids`（同一个 observation），内容一字不差。

为什么不在 observe 里即时去重：
    那要在写入路径上加后台线程或轮询，并发下更容易出错。重复本身不致命
    （只是挤占召回预算），离线清理更安全、可回滚。

保留规则（每组的胜出者）：
    1. 有 links 的（链式召回图已经接好了，删掉会断边）
    2. 有 role 的（能区分「他说的」和「我回的」）
    3. created_at 最早的（通常是 mnemosyne 自动建的那条，溯源最干净）
    其余标为 invalidated（软删，可恢复），并从向量索引里摘掉。

用法：
    python dedupe.py            # dry-run：只报告重复分组
    python dedupe.py --apply    # 真清理
    python dedupe.py --apply --keep-newest   # 反过来：保留最新的
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from persona_memory.main import collection_from_env
from persona_memory.milvus_store import MilvusVectorStore
from persona_memory.mnemosyne_client import MnemosyneClient

DEAD = ("invalidated", "superseded", "rejected", "decayed", "archived")


def find_db() -> Path:
    import os

    env = os.environ.get("MNEMOSYNE_DATA")
    candidates = []
    if env:
        candidates.append(Path(env) / "data" / "memory.db")
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "_inspect" / "mnemosyne-data" / "data" / "memory.db")
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"找不到 mnemosyne 数据库: {[str(c) for c in candidates]}")


def load_live(db: Path) -> list[dict]:
    uri = f"file:{db.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        sql = f"""
            SELECT id, content, status, created_at, metadata_json
            FROM memories
            WHERE status NOT IN ({','.join('?' * len(DEAD))})
              AND content IS NOT NULL AND content != ''
        """
        return [dict(r) for r in conn.execute(sql, DEAD).fetchall()]
    finally:
        conn.close()


def _meta(row: dict) -> dict:
    raw = row.get("metadata_json") if "metadata_json" in row else row.get("metadata")
    if isinstance(raw, dict):
        return raw
    try:
        import json

        return json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001
        return {}


def _meta_score(row: dict) -> tuple:
    """越高越该留下：有 links（图已接好）> 有 role（分得清谁说的）。"""
    meta = _meta(row)
    return (1 if meta.get("links") else 0, 1 if meta.get("role") else 0)


def _ordered(items: list[dict], keep_newest: bool = False) -> list[dict]:
    """胜出者排第一。同分时保留较早的那条（mnemosyne 自动建的，溯源更干净）。"""
    items = sorted(items, key=lambda x: str(x.get("created_at") or ""), reverse=keep_newest)
    return sorted(items, key=_meta_score, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-newest", action="store_true", help="保留最新的而不是最早的")
    args = ap.parse_args()

    db = find_db()
    rows = load_live(db)
    print(f"权威源：{db}（活记忆 {len(rows)} 条）")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        norm = " ".join(str(r.get("content", "")).split())
        groups[norm].append(r)

    # 回显去重：「对话片段：X」和「X」是同一句话的两种形态（旧客户端留下的），
    # 原文在的话，回显那条就是纯噪声——折叠掉，不留两条。
    echo_drop: list[dict] = []
    for key in list(groups.keys()):
        if not key.startswith("对话片段："):
            continue
        base = key[len("对话片段：") :].strip()
        if base in groups:
            echo_drop.extend(groups.pop(key))
    if echo_drop:
        print(f"回显重复：{len(echo_drop)} 条（原文还在，折叠掉）")

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    total_extra = sum(len(v) - 1 for v in dup_groups.values()) + len(echo_drop)
    print(f"重复分组：{len(dup_groups)} 组 + 回显 {len(echo_drop)} 条，多出来 {total_extra} 条")

    if not dup_groups and not echo_drop:
        print("没有重复记忆 🎉")
        return 0

    winners: list[str] = []
    losers: list[str] = [x["id"] for x in echo_drop]
    for norm, items in dup_groups.items():
        items = _ordered(items, args.keep_newest)
        winners.append(items[0]["id"])
        losers.extend(x["id"] for x in items[1:])
        print(f"\n  保留 {items[0]['id']}（{len(_meta(items[0]).get('links') or [])} 条边）")
        for x in items[1:]:
            print(f"   清理 {x['id']}  文:{str(x.get('content'))[:28]!r}")

    if not args.apply:
        print("\n这是 dry-run。确认后加 --apply 真清理。")
        return 0

    mn = MnemosyneClient()
    cache: dict[str, dict] = {}
    for mid in winners:
        try:
            cache[mid] = mn.get_memory(mid).get("metadata") or {}
        except Exception:  # noqa: BLE001
            cache[mid] = {}

    # 把 loser 里有价值的信息（如果有）合并进 winner
    for norm, items in dup_groups.items():
        items = _ordered(items, args.keep_newest)
        win = items[0]["id"]
        meta = dict(cache.get(win) or {})
        for x in items[1:]:
            other = _meta(x)
            for k in ("role", "tier", "strength", "salience"):
                if k not in meta and k in other:
                    meta[k] = other[k]
        if meta != (cache.get(win) or {}):
            try:
                mn.update_memory(win, metadata=meta)
            except Exception:  # noqa: BLE001
                pass

    vec = MilvusVectorStore(collection=collection_from_env())
    try:
        vec.connect()
        vec.ensure_collection()
    except Exception as e:  # noqa: BLE001
        # 向量层不可用不该挡住去重：权威源（SQLite）的状态才是真相，
        # 向量索引随时能从权威源重建。2026-11-14：Milvus 挂掉后这里一直崩。
        print(f"  ! 向量层不可用（{type(e).__name__}），跳过向量摘除，只做软删")
        vec = None
    cleaned = 0
    for mid in losers:
        try:
            mn.update_memory(mid, status="invalidated")
            cleaned += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {mid} 失效失败: {e}")
            continue
        if vec is not None:
            try:
                vec.delete(mid, flush=False)
            except Exception:  # noqa: BLE001
                pass
    if vec is not None:
        vec.flush()
    print(
        f"\n已失效 {cleaned} 条重复记忆（软删，可恢复；"
        f"{'向量已摘除' if vec is not None else '向量层跳过'}）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
