# -*- coding: utf-8 -*-
"""链式召回 + Inline & Spill 的纯逻辑测试（不需要 mnemosyne / Milvus / 网络）。

覆盖三件事：
  1. `_summarize`：摘要怎么截，不会从词中间砍断
  2. `observe`：新建与「mnemosyne 自动把 observation 转成 memory」两种情况，
     links 和 tier 都必须写进去（这曾经是一个隐藏很久的 bug）
  3. `retrieve` / `expand`：默认只给摘要 + 引用，detail 才给全文

跑法：
    python test_links.py
"""
from __future__ import annotations

import sys

from persona_memory.mnemosyne_client import MnemosyneError
from persona_memory.service import (
    LINK_SIMILARITY,
    MAX_LINKS,
    SUMMARY_CHARS,
    PersonaMemoryService,
    _summarize,
)

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


class FakeMnemosyne:
    """模拟 mnemosyne 的两个关键行为：

    1. `capture_observation` 时就自动把 observation 转成 memory
       （status=active，metadata 只有 from_observation）；
    2. `propose_memory` 遇到相同 content 会命中已有记忆，并丢弃传入的 metadata。
    """

    def __init__(self) -> None:
        self.memories = {}
        self.observations = {}
        self._seq = 0

    def _new_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq}"

    def health(self):
        return {"ok": True}

    def capture_observation(self, *, content, source_kind="dialog", occurred_at=None,
                            producer="", metadata=None, idempotency_key=None):
        oid = self._new_id("obs")
        self.observations[oid] = {"id": oid, "content": content}
        mid = self._new_id("mem")
        self.memories[mid] = {
            "id": mid,
            "type": "episodic",
            "status": "active",
            "content": content,
            "created_at": "2026-11-01T10:00:00Z",
            "metadata": {"from_observation": oid},
            "source_ids": [oid],
        }
        return {"id": oid}

    def search(self, *, query="", types=None, status=None, limit=10, include_evidence=False):
        # 忽略 query：让 _similarity / 精确匹配自己去筛（模拟「宽召回」）
        return list(self.memories.values())[:limit]

    def propose_memory(self, *, content, type="semantic", source_ids=None,
                       status="candidate", metadata=None, idempotency_key=None):
        for m in self.memories.values():
            if m["content"] == content:
                return m  # 命中已有：metadata 被丢弃（真实行为）
        mid = self._new_id("mem")
        self.memories[mid] = {
            "id": mid,
            "type": type,
            "status": status,
            "content": content,
            "created_at": "2026-11-01T10:00:00Z",
            "metadata": {},
            "source_ids": list(source_ids or []),
        }
        return self.memories[mid]

    def get_memory(self, mid, include_evidence=False):
        m = self.memories.get(mid)
        if m is None:
            raise MnemosyneError(404, {"detail": "memory not found"})
        return dict(m)

    def update_memory(self, mid, **fields):
        m = self.memories[mid]
        if "metadata" in fields:
            m["metadata"] = fields["metadata"]
        for k, v in fields.items():
            if k != "metadata":
                m[k] = v
        return m

    def forget_memory(self, mid, hard=False):
        self.memories.pop(mid, None)
        return {}

    def feedback(self, **kw):
        return {}


def build_service() -> tuple[PersonaMemoryService, FakeMnemosyne]:
    fake = FakeMnemosyne()
    svc = PersonaMemoryService(mnemosyne=fake, vectors=None, persona_path=None)
    return svc, fake


def test_summarize() -> None:
    print("[summarize]")
    short = "用户说这周想去看海"
    check("short text unchanged", _summarize(short) == short)

    long_text = "今天在做一个很难的改动。" + "中间有很多细节，" * 20
    s = _summarize(long_text)
    check("long text bounded", len(s) <= SUMMARY_CHARS + 1, f"len={len(s)}")
    check("keeps first sentence", s.startswith("今天在做一个很难的改动"))
    check("takes only first sentence", s == "今天在做一个很难的改动。", s)

    # 没有句末标点时才靠硬截断 + 省略号
    no_period = "这是一段很长很长的话" * 10
    s2 = _summarize(no_period)
    check("hard cut adds ellipsis", len(s2) <= SUMMARY_CHARS + 1 and s2.endswith("…"), f"len={len(s2)}")

    multiline = "第一行是重点。\n第二行是细节。\n第三行也是。"
    check("multiline takes first sentence", _summarize(multiline) == "第一行是重点。")

    check("empty text", _summarize("") == "")
    check("whitespace text", _summarize("   ") == "")


def test_observe_links() -> None:
    print("[observe / links]")
    svc, fake = build_service()

    # 第一条：没有邻居
    old = "用户说这周想去看海，他最近太累了"
    r1 = svc.observe(old, tier="engage", strength=0.9)
    check("first observe admitted", r1.admitted, r1.reason)
    m1 = fake.memories[r1.memory_id]
    check("tier stored on auto-created memory", m1["metadata"].get("tier") == "engage", str(m1["metadata"]))
    check("from_observation preserved", "from_observation" in m1["metadata"])

    # 第二条：与第一条高度相似 → 应该带 related_to 边
    new = "用户说这周想去看海，他最近太累了呀"
    r2 = svc.observe(new, tier="engage", strength=0.9)
    m2 = fake.memories[r2.memory_id]
    links = m2["metadata"].get("links") or []
    check("new memory has links", len(links) >= 1, str(m2["metadata"]))
    if links:
        check("link rel is related_to", links[0]["rel"] == "related_to", str(links[0]))
        check("link points at old memory", links[0]["target_id"] == m1["id"], str(links[0]))
        check("link has summary", bool(links[0].get("summary")))
        check("links capped", len(links) <= MAX_LINKS)

    # 第三条：重复文本 → mnemosyne 自动转过，走「复用」分支；
    # 这条路径必须也能补 metadata（否则 tier / links 永远进不去）
    r3 = svc.observe(old, tier="engage", strength=0.9)
    check("duplicate reuses memory", r3.memory_id == r1.memory_id, f"{r3.memory_id} vs {r1.memory_id}")
    m3 = fake.memories[r3.memory_id]
    check("duplicate keeps tier", m3["metadata"].get("tier") == "engage", str(m3["metadata"]))


def test_retrieve_inline_spill() -> None:
    print("[retrieve / inline & spill]")
    svc, fake = build_service()
    svc.observe("用户说这周想去看海，他最近太累了", tier="engage", strength=0.9)
    svc.observe("项目 deadline 压得我有点喘不过气", tier="engage", strength=0.9)

    r = svc.retrieve("想去看海", limit=5)
    check("retrieve non-empty", r["count"] > 0, str(r["count"]))
    check("detail flag echoed", r["detail"] is False)
    check("items have summary", all(i.get("summary") for i in r["items"]))
    check(
        "default content equals summary (inline)",
        all(len(i.get("content") or "") <= SUMMARY_CHARS + 1 for i in r["items"]),
    )
    check("items carry links field", all("links" in i for i in r["items"]))

    d = svc.retrieve("想去看海", limit=5, detail=True)
    check("detail returns full content", all(i.get("content") for i in d["items"]))
    check("detail costs more tokens", d["token_estimate"] >= r["token_estimate"])


def test_expand() -> None:
    print("[expand]")
    svc, fake = build_service()
    a = svc.observe("用户说这周想去看海，他最近太累了", tier="engage", strength=0.9)
    svc.observe("用户说这周想去看海，他最近太累了呀", tier="engage", strength=0.9)

    e = svc.expand(a.memory_id, depth=1)
    check("expand returns content", bool(e.get("content")))
    check("expand returns summary", bool(e.get("summary")))
    # a 是最早那条，可能还没连边（它是被指向的一方）；换成最新那条来展开
    newest = max(fake.memories.values(), key=lambda m: m["id"])
    e2 = svc.expand(newest["id"], depth=1)
    check("expand finds neighbours", e2["count"] >= 1, str(e2))
    if e2["links"]:
        check("neighbour has summary only", "summary" in e2["links"][0])
        check("neighbour hop is 1", e2["links"][0].get("hop") == 1)

    bad = svc.expand("mem_not_exist")
    check("expand missing id returns error", "error" in bad)


def test_invalidated_not_linked() -> None:
    print("[invalidated]")
    svc, fake = build_service()
    a = svc.observe("用户说这周想去看海，他最近太累了", tier="engage", strength=0.9)
    fake.memories[a.memory_id]["status"] = "invalidated"
    r = svc.observe("用户说这周想去看海，他最近太累了呀", tier="engage", strength=0.9)
    m = fake.memories[r.memory_id]
    targets = [lk["target_id"] for lk in (m["metadata"].get("links") or [])]
    check("invalidated memory not linked", a.memory_id not in targets, str(targets))


def test_proactive() -> None:
    print("[proactive]")
    svc, fake = build_service()
    r = svc.observe(
        "用户说这周想去看海，他最近太累了",
        tier="engage",
        strength=0.9,
        metadata={"role": "user"},
    )
    check("role stored on memory", fake.memories[r.memory_id]["metadata"].get("role") == "user")

    # fake 把 created_at 固定成 2026-11-01T10:00:00Z
    from datetime import datetime, timezone

    created = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc).timestamp()
    too_new = created + 3600  # 1 小时后：太新，不算「想起」
    later = created + 13 * 3600  # 13 小时后：该想起来了

    p0 = svc.proactive_recall(limit=1, now=too_new)
    check("too new is not surfaced", p0["count"] == 0, str(p0))

    p1 = svc.proactive_recall(limit=1, now=later)
    check("surfaces old user memory", p1["count"] == 1, str(p1))
    if p1["items"]:
        check("surfaced item has reason", bool(p1["items"][0].get("reason")))
        check("surfaced item has summary", bool(p1["items"][0].get("summary")))

    # 刚提过 → 冷却期内不再提
    p2 = svc.proactive_recall(limit=1, now=later + 60)
    check("cooldown prevents repeat", p2["count"] == 0, str(p2))

    # assistant 的回复不该被「想起」
    ra = svc.observe(
        "我觉得今天状态不错，可以多做一点",
        tier="attention",
        strength=0.4,
        metadata={"role": "assistant"},
    )
    fake.memories[ra.memory_id]["created_at"] = "2026-11-01T09:00:00Z"
    p3 = svc.proactive_recall(limit=1, now=later)
    ids = [it["memory_id"] for it in p3["items"]]
    check("assistant memory not surfaced", ra.memory_id not in ids, str(ids))


def main() -> int:
    print("=== 链式召回 / Inline & Spill 测试 ===\n")
    test_summarize()
    print()
    test_observe_links()
    print()
    test_retrieve_inline_spill()
    print()
    test_expand()
    print()
    test_invalidated_not_linked()
    print()
    test_proactive()
    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
