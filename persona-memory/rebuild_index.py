# -*- coding: utf-8 -*-
"""重建向量索引：以 mnemosyne（权威源）为准，让 Milvus 与它对齐。

为什么需要：删记忆时如果只删了一边，索引里会留下孤儿向量——
检索时它们会以「幽灵记忆」的形式出现，指向不存在的记忆。
权威源在 SQLite，所以永远按它重建索引。

⚠️ 为什么直接读 SQLite 而不是走 mnemosyne 的 HTTP API：
它的 /memories/search 有一个**硬上限 50 条**（core/service.py: min(50, limit)），
用它重建索引会把 100+ 条记忆的向量全丢掉。这一点已经真踩过一次：
重建后 Milvus 从 128 行掉到 50 行，watchdog 才报出来。
直接读权威源 SQLite 是唯一能拿到「全量」的路径。

用法：
    python rebuild_index.py            # 重建（会先 drop 再建）
    python rebuild_index.py --dry-run  # 只数一下会重建多少条
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from persona_memory.main import _make_embedder, collection_for
from persona_memory.milvus_store import MilvusVectorStore

DEAD = ("invalidated", "superseded", "rejected", "decayed", "archived")


def find_db() -> Path:
    """按 bootstrap 的默认数据目录推断；也支持 MNEMOSYNE_DATA 覆盖。"""
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
    raise SystemExit(f"找不到 mnemosyne 数据库，试过：{[str(c) for c in candidates]}")


def load_rows(db: Path) -> list[dict]:
    # 只读打开：mnemosyne 服务可能正在写（WAL 模式），不要拿写锁
    uri = f"file:{db.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        sql = f"""
            SELECT id, content, type, status
            FROM memories
            WHERE status NOT IN ({','.join('?' * len(DEAD))})
              AND content IS NOT NULL AND content != ''
            ORDER BY COALESCE(updated_at, created_at) DESC
        """
        rows = conn.execute(sql, DEAD).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 向量器跟着服务走：换模型就换集合，绝不混用（语义空间不同）
    embedder, kind = _make_embedder()
    collection = collection_for(kind)
    print(f"向量器：{kind}（dim={getattr(embedder, 'dim', '?')}）-> 集合 {collection}")

    db = find_db()
    rows = load_rows(db)
    print(f"权威源：{db}")
    print(f"活记忆：{len(rows)} 条（从 SQLite 直读，绕过 API 的 50 条上限）")
    if args.dry_run:
        return 0
    if not rows:
        print("没有可重建的记忆，退出（不 drop 集合）")
        return 1

    # 1) 丢掉整个集合
    from pymilvus import MilvusClient

    # 地址走环境变量：Milvus 在 WSL 里时地址会变（localhost 转发常断），
    # 和 MilvusVectorStore 保持一致，否则会出现「建好了却删不掉」的错位。
    import os

    _uri = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
    print(f"Milvus: {_uri}")
    c = MilvusClient(uri=_uri)
    if collection in c.list_collections():
        c.drop_collection(collection)
        print("已删除旧集合")

    # 2) 重建并重新索引
    vec = MilvusVectorStore(collection=collection, embedder=embedder)
    vec.ensure_collection()
    print("已重建集合")

    n = 0
    for m in rows:
        try:
            vec.upsert(
                memory_id=str(m["id"]),
                content=str(m["content"]),
                mtype=str(m.get("type") or "episodic"),
                status=str(m.get("status") or "candidate"),
                source_kind="mnemosyne",
                salience=0.5,
            )
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"失败 {m.get('id')}: {e}")
    vec.flush()
    print(f"重新索引 {n} 条")
    print("Milvus stats:", c.get_collection_stats(collection))
    return 0


if __name__ == "__main__":
    sys.exit(main())
