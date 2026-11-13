# -*- coding: utf-8 -*-
"""纯逻辑层 smoke test：分子 / 人格三层 / 准入与状态机。"""
from __future__ import annotations

import sys
import time

from persona_memory.admission import (
    InterestState,
    MemoryState,
    Tier,
    advance,
    aggregate_interest,
    interest_state,
    proactive_weight,
    recency_factor,
    should_buffer,
)
from persona_memory.molecules import EMOTION_DIMS, EndocrineSystem, MOLECULES
from persona_memory.persona import PersonaState

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def test_molecules() -> None:
    print("[molecules]")
    es = EndocrineSystem()
    check("15 molecules", len(MOLECULES) == 15)
    check("baseline dopamine", abs(es.get("dopamine") - 0.22) < 1e-9)

    es.release("cortisol", 0.5)
    hi = es.get("cortisol")
    check("release raises cortisol", hi > 0.15)

    es.tick(600)  # 一个半衰期
    after = es.get("cortisol")
    check("tick decays toward baseline", 0.15 < after < hi, f"{after=}")

    eff = es.effects()
    check("effect keys", {"learning_rate_mod", "arousal", "valence", "emotion"} <= set(eff))
    check("emotion 12 dims", set(eff["emotion"]) == set(EMOTION_DIMS))
    check(
        "emotion within 0..1",
        all(0.0 <= v <= 1.0 for v in eff["emotion"].values()),
        str(eff["emotion"]),
    )
    check("learning_rate_mod positive", eff["learning_rate_mod"] > 0)

    # 高皮质醇 → 唤醒升高
    es2 = EndocrineSystem()
    es2.release("cortisol", 0.95)
    check("cortisol raises arousal", es2.effects()["arousal"] > EndocrineSystem().effects()["arousal"])


def test_persona() -> None:
    print("[persona]")
    ps = PersonaState.from_base_profile(
        {"warmth": {"value": 0.85, "note": "先接情绪"}, "directness": {"value": 0.4}}
    )
    check("base anchored", ps.base_value("warmth") == 0.85)

    # 日常路径不许改基础层
    check("base write rejected without explicit", ps.set_base("warmth", 0.1) is False)
    check("base write allowed when explicit", ps.set_base("warmth", 0.9, explicit_user_command=True) is True)
    check("base changed after explicit", ps.base_value("warmth") == 0.9)

    # 经验层：证据不足不改值
    v0 = ps.shaped["warmth"].value
    r1 = ps.observe_trait("warmth", delta=0.5, evidence_id="e1")
    r2 = ps.observe_trait("warmth", delta=0.5, evidence_id="e2")
    check("no move below evidence threshold", r1["applied"] is False and r2["applied"] is False)
    check("duplicate evidence rejected", ps.observe_trait("warmth", delta=0.5, evidence_id="e1")["reason"] == "duplicate_evidence")

    r3 = ps.observe_trait("warmth", delta=0.5, evidence_id="e3")
    check("applies at 3rd evidence", r3["applied"] is True, str(r3))
    check("single step is small", abs(r3["step"]) <= 0.05 + 1e-9, str(r3))
    check("value moved", abs(ps.shaped["warmth"].value - v0) > 0)

    # 动态层
    ps.release_emotion({"cortisol": 0.8, "adrenaline": 0.6})
    eff = ps.effects()
    check("dynamic effects reachable", eff["arousal"] > 0)
    # 情绪与疲劳解耦（2026-11-11）：情绪事件不再加疲劳，
    # 疲劳只由交互量（add_fatigue）驱动，否则「晚安 mua」会比写代码更累。
    check("emotion does not add fatigue", ps.dynamic.fatigue == 0.0)
    ps.add_fatigue(0.01)
    check("fatigue rises with interaction", ps.dynamic.fatigue > 0)
    snap = ps.snapshot()
    check("snapshot has 3 layers", set(snap) == {"base", "shaped", "dynamic"})


def test_admission() -> None:
    print("[admission]")
    now = time.time()

    # 准入
    r = should_buffer({"value": 0.1, "reliability": 0.2, "novelty": 0.3, "persistence": 0.1, "explicit_feedback": 0.0})
    check("low scores rejected", r["admit"] is False, str(r))
    r = should_buffer({"value": 0.1, "reliability": 0.9, "novelty": 0.3, "persistence": 0.1, "explicit_feedback": 0.0})
    check("one strong dimension admits", r["admit"] is True, str(r))
    check("deny rule wins", should_buffer({"value": 0.99}, deny=True)["admit"] is False)

    # 状态机不能跳级
    check("OBSERVED -> BUFFERED", advance(MemoryState.OBSERVED, evidence_net=0, tiers=set(), days_span=0)["state"] is MemoryState.BUFFERED)
    check("BUFFERED -> EPISODIZED", advance(MemoryState.BUFFERED, evidence_net=0, tiers=set(), days_span=0)["state"] is MemoryState.EPISODIZED)

    # CANDIDATE 不能一天就到 ACTIVE
    r = advance(MemoryState.CANDIDATE, evidence_net=5.0, tiers={Tier.EXPLICIT}, days_span=1)
    check("candidate one-day not active", r["state"] is MemoryState.CANDIDATE, str(r))
    r = advance(MemoryState.CANDIDATE, evidence_net=5.0, tiers={Tier.EXPLICIT}, days_span=3)
    check("candidate cross-day active", r["state"] is MemoryState.ACTIVE, str(r))

    # 纠正与冲突
    r = advance(MemoryState.ACTIVE, evidence_net=5.0, tiers={Tier.EXPLICIT}, days_span=3, user_corrected=True)
    check("user correction invalidates", r["state"] is MemoryState.INVALIDATED)
    r = advance(MemoryState.ACTIVE, evidence_net=5.0, tiers={Tier.EXPLICIT}, days_span=3, has_conflict=True)
    check("conflict suspends", r["state"] is MemoryState.NEEDS_REVIEW)

    # 兴趣：单次被动暴露不得 CONFIRMED
    passive_once = [{"tier": "passive", "polarity": 1, "strength": 0.9, "occurred_at": now, "source_kind": "video"}]
    agg = aggregate_interest(passive_once, now)
    st = interest_state(agg["net"], passive_once, now)
    check("passive once not confirmed", st is not InterestState.CONFIRMED, f"{st=} {agg=}")
    check("passive once unknown/weak", st in (InterestState.UNKNOWN, InterestState.WEAK_SIGNAL), str(st))

    # 跨天 engage + 对话追问 → CANDIDATE / CONFIRMED
    strong = [
        {"tier": "engage", "polarity": 1, "strength": 1.0, "occurred_at": now, "source_kind": "chat"},
        {"tier": "engage", "polarity": 1, "strength": 1.0, "occurred_at": now - 86400, "source_kind": "chat"},
        {"tier": "explicit", "polarity": 1, "strength": 1.0, "occurred_at": now - 86400 * 2, "source_kind": "feedback"},
    ]
    agg2 = aggregate_interest(strong, now)
    st2 = interest_state(agg2["net"], strong, now)
    check("engage cross-day confirmed", st2 is InterestState.CONFIRMED, f"{st2=} {agg2=}")

    # 明确否定 → NEEDS_REVIEW
    neg = [
        {"tier": "explicit", "polarity": -1, "strength": 1.0, "occurred_at": now, "source_kind": "feedback"},
        {"tier": "engage", "polarity": 1, "strength": 1.0, "occurred_at": now, "source_kind": "chat"},
    ]
    agg3 = aggregate_interest(neg, now)
    check("explicit dislike flags review", interest_state(agg3["net"], neg, now) is InterestState.NEEDS_REVIEW, str(agg3))

    # 一次反例不抹掉大量正向证据（oppose_discount < 1 的意图）
    many_pos = [{"tier": "engage", "polarity": 1, "strength": 1.0, "occurred_at": now, "source_kind": "chat"} for _ in range(10)]
    one_neg = many_pos + [{"tier": "attention", "polarity": -1, "strength": 0.5, "occurred_at": now, "source_kind": "video"}]
    a = aggregate_interest(one_neg, now)
    check("one counterexample does not zero support", a["net"] > 0, str(a))

    # 主动提起：WEAK 不得主动
    check("weak never proactive-high", proactive_weight(InterestState.WEAK_SIGNAL, net=0.2, fatigue=0.0, arousal=0.5) <= 0.05)
    check("confirmed can be proactive", proactive_weight(InterestState.CONFIRMED, net=2.0, fatigue=0.0, arousal=0.5) > 0.5)
    check("fatigue reduces proactive", proactive_weight(InterestState.CONFIRMED, net=2.0, fatigue=1.0, arousal=0.5) < proactive_weight(InterestState.CONFIRMED, net=2.0, fatigue=0.0, arousal=0.5))

    # 时间感知：旧证据衰减
    check("recency decays", recency_factor(now - 86400 * 60, now) < 0.3)
    check("recency fresh", recency_factor(now, now) == 1.0)


if __name__ == "__main__":
    test_molecules()
    test_persona()
    test_admission()
    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        sys.exit(1)
    print("RESULT: ALL PASS")
