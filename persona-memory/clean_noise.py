# -*- coding: utf-8 -*-
"""清理「系统噪音」记忆：DSH 的运行时上下文被误当成人说的话记下来了。

历史上 dsh-suyi 的 `isSystemNoise` 白名单不够全，于是库里混进了
「The approval policy changed…」「Current runtime context…」「你正在扮演角色…」
这类东西。它们不是对话，是运行时注入——留着会污染召回，也会让
主动提起（proactive_recall）说出很怪的话。

这里按特征词清一遍（软删，可恢复）。

用法：
    python clean_noise.py            # dry-run
    python clean_noise.py --apply
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from persona_memory.main import collection_from_env
from persona_memory.milvus_store import MilvusVectorStore
from persona_memory.mnemosyne_client import MnemosyneClient

DEAD = ("invalidated", "superseded", "rejected", "decayed", "archived")

# 只匹配「开头」的特征：避免误伤真正在聊这些词的用户对话
NOISE_PREFIXES = (
    "The approval policy changed",
    "Current runtime context",
    "Current working directory",
    "Current DeepSeek Harness implementation checkout",
    "Current DSH file policy",
    "Time sampled while preparing",
    "<system-reminder>",
    "You are an AI agent powered by",
    "You are a Team Lead named",
    "Agent Teams is available in this session",
    "background job ",
    "started background job ",
    "你正在扮演角色",
    "[PERSONA_LOAD]",
    "[此刻的你]",
    "[我们之前聊过这些]",
    "对话片段：The approval policy",
    "对话片段：Current runtime",
)


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
    raise SystemExit("找不到 mnemosyne 数据库")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    db = find_db()
    uri = f"file:{db.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, content, status FROM memories
            WHERE status NOT IN ({','.join('?' * len(DEAD))})
              AND content IS NOT NULL AND content != ''
            """,
            DEAD,
        ).fetchall()
    finally:
        conn.close()

    hits = []
    for r in rows:
        c = str(r["content"]).strip()
        if c.startswith(NOISE_PREFIXES):
            hits.append((str(r["id"]), c[:48].replace("\n", " ")))

    print(f"扫描 {len(rows)} 条，命中系统噪音 {len(hits)} 条")
    for mid, preview in hits[:20]:
        print(f"  - {mid}  {preview}")
    if len(hits) > 20:
        print(f"  …还有 {len(hits) - 20} 条")

    if not args.apply:
        print("\n这是 dry-run。确认后加 --apply 真清理。")
        return 0

    mn = MnemosyneClient()
    vec = MilvusVectorStore(collection=collection_from_env())
    vec.connect()
    vec.ensure_collection()
    cleaned = 0
    for mid, _ in hits:
        try:
            mn.update_memory(mid, status="invalidated")
            cleaned += 1
            vec.delete(mid, flush=False)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {mid} 清理失败: {e}")
    vec.flush()
    print(f"\n已失效 {cleaned} 条系统噪音记忆（软删，可恢复）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
