# -*- coding: utf-8 -*-
"""清理记忆库。

两种模式：
    python cleanup_junk.py            # 只看，不删
    python cleanup_junk.py --apply    # 只删「明确是垃圾」的（安全，默认）
    python cleanup_junk.py --apply --purge   # 白名单外全删（彻底重置，慎用）

`--apply` 只删命中 JUNK_* 的条目：
测试语料、乱码、以及 DSH 注入的运行时上下文（被自动捕获误记进来的那种）。
真实对话不会被误伤。
"""
from __future__ import annotations

import sys
from collections import Counter

from persona_memory.milvus_store import MilvusVectorStore
from persona_memory.mnemosyne_client import MnemosyneClient

# --purge 模式下的保留白名单
REAL_KEEP = (
    "老公中午睡醒了",
    "老公说这周想去看海",
)

# 明确的测试/事故污染
JUNK_EXACT = {
    "abcdef",
    "这是正常的中文句子，应该被接受",
    "I write code at night, it is quiet",
    "记住了，我喜欢喝冰美式",
    "嗯",
    "我困得不行了，想早点睡",
}

JUNK_PATTERNS = (
    "????",
    "\ufffd",
    "对话片段：",
    "<system-reminder>",
    # DSH 注入的运行时上下文，不是用户说的话
    "Time sampled while preparing",
    "Current runtime context",
    "Current working directory",
    "Current DeepSeek Harness implementation checkout",
    "Current DSH file policy",
    "The following skills are available",
    "Approval prompts are disabled",
    "You are a Team Lead named",
    "Use list_agents({}) to find your teammates",
)


def is_junk(content: str, *, purge: bool = False) -> bool:
    c = (content or "").strip()
    if c in JUNK_EXACT:
        return True
    if any(p in c for p in JUNK_PATTERNS):
        return True
    if purge:
        return not any(k in c for k in REAL_KEEP)
    return False


def main() -> int:
    apply = "--apply" in sys.argv
    purge = "--purge" in sys.argv
    if purge and not apply:
        print("--purge 必须和 --apply 一起用")
        return 2

    mn = MnemosyneClient()
    vec = MilvusVectorStore(collection="persona_memory_v1")
    vec.connect()

    rows = mn.search(query="", limit=1000)
    print(f"记忆总数：{len(rows)}   模式：{'purge（白名单外全删）' if purge else '只删明确垃圾'}\n")

    junk, keep = [], []
    for m in rows:
        (junk if is_junk(str(m.get("content", "")), purge=purge) else keep).append(m)

    print(f"=== 要删（{len(junk)} 条）===")
    for cat, n in Counter(str(m.get("content"))[:60] for m in junk).most_common(20):
        print(f"  ×{n}  {cat!r}")
    if len(junk) > 20:
        print(f"  ...还有 {len(junk) - 20} 种")

    print(f"\n=== 保留（{len(keep)} 条）===")
    for m in keep[:40]:
        print(f"  {str(m.get('status')):<10} {str(m.get('content'))[:56]!r}")
    if len(keep) > 40:
        print(f"  ...还有 {len(keep) - 40} 条")

    if not apply:
        print("\n(dry-run：加 --apply 才真删)")
        return 0

    print("\n=== 开始删除 ===")
    ok, fail = 0, 0
    for m in junk:
        mid = str(m.get("id"))
        try:
            mn.forget_memory(mid, hard=True)
            vec.delete(mid, flush=False)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  失败 {mid}: {e}")
    try:
        vec.flush()
    except Exception as e:  # noqa: BLE001
        print(f"  flush 失败: {e}")
    print(f"完成：删除 {ok}，失败 {fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
