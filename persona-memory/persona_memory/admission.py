# -*- coding: utf-8 -*-
"""记忆准入、证据分层与状态机。

把 dsh-mnemosyne 的 STATE_MACHINE 草案（V0.1）落成可执行规则：
Observation 永不直接变成 Active Memory；兴趣必须靠分层证据累积晋升；
一次反例不能抹掉大量可靠正向证据。

核心不变量（与 mnemosyne 的「五条底线」一致）：
1. 观察 ≠ 记忆：Observation 只能进 BUFFERED，不能直接 ACTIVE。
2. 推断 ≠ 事实：agent_inference 来源的证据权重打折，且不能单独晋升。
3. 可纠错：INVALIDATED 不再作为正依据召回。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


class Tier(str, Enum):
    """证据层级。权重是默认起点，必须实验校准，不是心理学常数。"""

    PASSIVE = "passive"      # 被动暴露：自动播放、后台
    ATTENTION = "attention"  # 注意力：停留、回看、主动暂停
    ENGAGE = "engage"        # 主动参与：搜索、评论、追问
    EXPLICIT = "explicit"    # 明确反馈：喜欢/不喜欢/想深入


TIER_WEIGHT: Mapping[Tier, float] = {
    Tier.PASSIVE: 0.15,
    Tier.ATTENTION: 0.40,
    Tier.ENGAGE: 0.70,
    Tier.EXPLICIT: 1.00,
}

# 用户明确否定时，反例打折更少（更容易扭转）
OPPOSE_DISCOUNT_DEFAULT = 0.7
OPPOSE_DISCOUNT_EXPLICIT = 1.0

# 晋升门槛（默认值，写入 algorithm_version 后可比实验）
T_WEAK = 0.15
T_CANDIDATE = 0.50
T_CONFIRMED = 1.20
T_STALE = 0.20

# 时间衰减：信号按天衰减，半衰期 30 天
RECENCY_HALF_LIFE_DAYS = 30.0


class MemoryState(str, Enum):
    OBSERVED = "observed"
    BUFFERED = "buffered"
    EPISODIZED = "episodized"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    NEEDS_REVIEW = "needs_review"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    ARCHIVED = "archived"
    DECAYED = "decayed"
    REJECTED = "rejected"


class InterestState(str, Enum):
    UNKNOWN = "unknown"
    WEAK_SIGNAL = "weak_signal"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    STALE = "stale"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


INTEREST_TO_MEMORY = {
    InterestState.WEAK_SIGNAL: MemoryState.CANDIDATE,
    InterestState.CANDIDATE: MemoryState.CANDIDATE,
    InterestState.CONFIRMED: MemoryState.ACTIVE,
    InterestState.NEEDS_REVIEW: MemoryState.NEEDS_REVIEW,
}


def recency_factor(occurred_at: float, now: float) -> float:
    """0..1 的时间新鲜度。"""
    age_days = max(0.0, (now - occurred_at) / 86400.0)
    return math.pow(0.5, age_days / RECENCY_HALF_LIFE_DAYS)


def evidence_score(
    tier: Tier,
    strength: float,
    occurred_at: float,
    now: float,
    source_trust: float = 1.0,
) -> float:
    """单条证据的加权分。strength 是原始强度（0..1），不是可信度。"""
    return (
        TIER_WEIGHT[tier]
        * max(0.0, min(1.0, strength))
        * recency_factor(occurred_at, now)
        * max(0.0, min(1.0, source_trust))
    )


def aggregate_interest(
    signals: Sequence[Mapping[str, object]],
    now: float,
) -> Dict[str, object]:
    """把一组兴趣信号聚合成 support / oppose / net。

    超加性保护：一次 explicit 反例最多只让 net 掉到 0，不直接清零正向总量。
    """
    support = 0.0
    oppose = 0.0
    explicit_oppose = 0.0

    for sig in signals:
        polarity = int(sig.get("polarity", 1))  # type: ignore[arg-type]
        tier = Tier(str(sig.get("tier", Tier.PASSIVE.value)))
        s = evidence_score(
            tier,
            float(sig.get("strength", 0.5)),  # type: ignore[arg-type]
            float(sig.get("occurred_at", now)),  # type: ignore[arg-type]
            now,
            source_trust=float(sig.get("source_trust", 1.0)),  # type: ignore[arg-type]
        )
        if polarity >= 0:
            support += s
        else:
            oppose += s
            if tier is Tier.EXPLICIT:
                explicit_oppose += s

    discount = (
        OPPOSE_DISCOUNT_EXPLICIT
        if explicit_oppose > 0
        else OPPOSE_DISCOUNT_DEFAULT
    )
    net = support - discount * oppose
    return {
        "support": round(support, 4),
        "oppose": round(oppose, 4),
        "oppose_discount": discount,
        "net": round(net, 4),
    }


def interest_state(
    net: float,
    signals: Sequence[Mapping[str, object]],
    now: float,
) -> InterestState:
    """兴趣状态机：UNKNOWN → WEAK → CANDIDATE → CONFIRMED，可 STALE/NEEDS_REVIEW。"""
    tiers: Set[Tier] = {Tier(str(s.get("tier", Tier.PASSIVE.value))) for s in signals}
    days = _distinct_days(signals)
    explicit_oppose = any(
        int(s.get("polarity", 1)) < 0 and Tier(str(s.get("tier"))) is Tier.EXPLICIT
        for s in signals
    )

    if explicit_oppose:
        return InterestState.NEEDS_REVIEW
    if not signals:
        return InterestState.UNKNOWN
    if net <= 0:
        return InterestState.STALE if any(t is not Tier.PASSIVE for t in tiers) else InterestState.UNKNOWN
    if net >= T_CONFIRMED and (Tier.ENGAGE in tiers or Tier.EXPLICIT in tiers) and days >= 2:
        return InterestState.CONFIRMED
    if net >= T_CANDIDATE and (len(_distinct_sources(signals)) >= 2 or Tier.ENGAGE in tiers):
        return InterestState.CANDIDATE
    if net >= T_WEAK and (any(t is not Tier.PASSIVE for t in tiers) or days >= 2):
        return InterestState.WEAK_SIGNAL
    return InterestState.UNKNOWN


def _distinct_days(signals: Sequence[Mapping[str, object]]) -> int:
    days = set()
    for s in signals:
        ts = float(s.get("occurred_at", 0.0))  # type: ignore[arg-type]
        if ts <= 0:
            continue
        days.add(datetime.fromtimestamp(ts, timezone.utc).date())
    return len(days)


def _distinct_sources(signals: Sequence[Mapping[str, object]]) -> Set[str]:
    out: Set[str] = set()
    for s in signals:
        src = str(s.get("source_kind", ""))
        if src:
            out.add(src)
    return out


# ---------------------------------------------------------------- 准入评分

ACCEPT_KEYS = ("value", "reliability", "novelty", "persistence", "explicit_feedback")


def admission_scores(
    *,
    value: float = 0.0,
    reliability: float = 0.0,
    novelty: float = 0.0,
    persistence: float = 0.0,
    explicit_feedback: float = 0.0,
) -> Dict[str, float]:
    return {
        "value": value,
        "reliability": reliability,
        "novelty": novelty,
        "persistence": persistence,
        "explicit_feedback": explicit_feedback,
    }


def should_buffer(scores: Mapping[str, float], *, deny: bool = False) -> Dict[str, object]:
    """准入：任一维达标即可进 BUFFERED；命中 deny 直接拒绝。

    这是通往昂贵认知链的闸门——不该让每条闲聊都进情节归纳。

    一条额外规则：**新颖本身不是理由**。「没见过」不等于「值得记」，
    否则任何第一次出现的噪声都会被永久保存。
    """
    if deny:
        return {"admit": False, "reason": "deny_rule", "scores": dict(scores)}
    hit = [k for k, v in scores.items() if float(v) >= 0.6]
    if not hit:
        return {"admit": False, "reason": "no_dimension_above_threshold", "scores": dict(scores)}
    if hit == ["novelty"]:
        return {"admit": False, "reason": "novelty_alone_insufficient", "scores": dict(scores)}
    return {"admit": True, "reason": f"admitted_by:{','.join(sorted(hit))}", "scores": dict(scores)}


def advance(
    state: MemoryState,
    *,
    evidence_net: float,
    tiers: Iterable[Tier],
    days_span: int,
    has_conflict: bool = False,
    user_corrected: bool = False,
) -> Dict[str, object]:
    """通用记忆状态转移。返回 {"state": 新状态, "reason": 理由}。

    硬规则：
    - 任何状态都不能因单次观察跳到 ACTIVE；
    - 有冲突先进 NEEDS_REVIEW，不静默覆盖；
    - 用户纠正 → INVALIDATED（保留历史，不再作正依据）。
    """
    tiers = set(tiers)

    if user_corrected:
        return {"state": MemoryState.INVALIDATED, "reason": "user_correction"}
    if has_conflict and state is MemoryState.ACTIVE:
        return {"state": MemoryState.NEEDS_REVIEW, "reason": "conflict_suspend"}
    if state is MemoryState.OBSERVED:
        return {"state": MemoryState.BUFFERED, "reason": "admission_passed"}
    if state is MemoryState.BUFFERED:
        return {"state": MemoryState.EPISODIZED, "reason": "window_closed"}
    if state is MemoryState.EPISODIZED:
        if evidence_net <= 0:
            return {"state": MemoryState.REJECTED, "reason": "no_stable_assertion"}
        return {"state": MemoryState.CANDIDATE, "reason": "candidate_assertion"}
    if state is MemoryState.CANDIDATE:
        if evidence_net >= T_CONFIRMED and tiers & {Tier.ENGAGE, Tier.EXPLICIT} and days_span >= 2:
            return {"state": MemoryState.ACTIVE, "reason": "evidence_threshold_met"}
        if evidence_net < T_STALE:
            return {"state": MemoryState.DECAYED, "reason": "below_stale_threshold"}
        return {"state": MemoryState.CANDIDATE, "reason": "awaiting_more_evidence"}
    if state is MemoryState.NEEDS_REVIEW:
        if evidence_net >= T_CONFIRMED:
            return {"state": MemoryState.ACTIVE, "reason": "review_resolved"}
        if evidence_net < 0:
            return {"state": MemoryState.INVALIDATED, "reason": "review_rejected"}
        return {"state": MemoryState.NEEDS_REVIEW, "reason": "review_pending"}
    return {"state": state, "reason": "no_transition"}


# ---------------------------------------------------------------- 主动提起策略

def proactive_weight(
    interest: InterestState,
    *,
    net: float,
    fatigue: float,
    arousal: float,
) -> float:
    """决定「要不要主动提起这件事」的权重。只允许 CONFIRMED/ACTIVE 抬高。

    动态状态层在这里真正生效：疲劳高时降低主动打扰，
    高唤醒时对强记忆更敏感——但阈值由状态机定，情绪只能调，不能翻盘。
    """
    base = 0.0
    if interest is InterestState.CONFIRMED:
        base = min(1.0, net / T_CONFIRMED)
    elif interest is InterestState.CANDIDATE:
        base = 0.3 * min(1.0, net / T_CONFIRMED)
    elif interest in (InterestState.WEAK_SIGNAL, InterestState.UNKNOWN):
        # 规范：WEAK 只影响「观察灵敏度」，不参与「主动提起」。
        # 留一个极小的底值，避免把「不确定」误当成「确定不该提」。
        base = 0.02

    fatigue_penalty = 1.0 - 0.5 * max(0.0, min(1.0, fatigue))
    arousal_boost = 1.0 + 0.25 * max(0.0, min(1.0, arousal))
    return round(max(0.0, min(1.0, base * fatigue_penalty * arousal_boost)), 4)
