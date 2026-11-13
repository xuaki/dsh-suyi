# -*- coding: utf-8 -*-
"""兴趣画像：记住「你喜欢什么」。

要解决的问题：

    我到现在还不知道你喜欢什么。
    你随口说一句「最近在玩吉他」，我不会记得。
    而这恰恰是「懂你」里最基础的一层。

做法上刻意保守——比「多记几条」更重要的是「别记错」：
  - 只从**明确句式**里提取（「我喜欢X」「我迷上了X」「我最近在玩X」），
    不做推测、不猜、不引申；
  - 代词/虚词一律过滤（「我喜欢你」里的「你」不是兴趣）；
  - 同一个话题按次数累积，只有**反复出现**的才算「他喜欢的东西」。

参考 digital-life-core 的执念收割器踩过的坑：它一开始什么感慨都收，
池子一夜之间堆满垃圾。宁可漏掉，不可记错。

2026-11-12（P1）：接上 admission 的状态机。
--------------------------------------------------
以前这里是**平行的第二套**判定：`confirmed` 只看 `count >= 2`，而唯一的输入端
`extract_topics()` 只认「我喜欢X」这种明确句式——admission 里那套
`PASSIVE / ATTENTION / ENGAGE / EXPLICIT` 分级证据、跨天门槛、NEEDS_REVIEW
从来没被喂过一条弱证据。「隐式偏好抓不到」的病灶就在这里，不是正则不够宽。

现在：
  - `Topic.signals` 存每条证据（{polarity, tier, quote, confidence, at}）；
  - `observe_signal()` 记「他顺口提了一句」——**不是**「他喜欢这个」；
  - `confirmed` 由 `interest_verdict()`（aggregate_interest + interest_state）判，
    `count >= 2` 只做旧数据的 fallback；
  - `extract_topics()` 前面加了一道归属判别（`_ownership_verdict`），
    别人的偏好、假设句一律不算他的。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .admission import (
    T_CANDIDATE,
    InterestState,
    aggregate_interest,
    interest_state,
)

# 只在这些句式里抓。宁窄不宽——但也不能窄到只有教科书句式，
# 「入了XX的坑」「对XX上头」这种口语在真实对话里更多见（EVALUATION.md 局限 2）。
_PATTERNS = [
    re.compile(r"我(?:很|超|特别|最)?喜欢([^，。！？、；：\s「」“”‘’]{1,12})"),
    re.compile(r"我(?:迷上|爱上)了?([^，。！？、；：\s「」“”‘’]{1,12})"),
    re.compile(r"我最(?:爱|喜欢)(?:的)?(?:是)?([^，。！？、；：\s「」“”‘’]{1,12})"),
    re.compile(r"我(?:最近|平时|经常)(?:在)?(?:玩|看|听|学|读|追)([^，。！？、；：\s「」“”‘’]{1,12})"),
    re.compile(r"我在(?:玩|看|听|学|读|追)([^，。！？、；：\s「」“”‘’]{1,12})"),
    re.compile(r"我(?:一直想|想)(?:去|学|玩|看|买)([^，。！？、；：\s「」“”‘’]{1,12})"),
    # --- 口语/网络表达（2026-11-11 放宽） ---
    re.compile(r"我(?:入了|掉进|跳进)([^，。！？、；：\s「」“”‘’的]{1,12})的?坑"),
    re.compile(r"我(?:沉迷|着迷|上头)(?:于|了|上了)?([^，。！？、；：\s「」“”‘’]{1,12})"),
    re.compile(r"我(?:疯狂)?种草了?([^，。！？、；：\s「」“”‘’]{1,12})"),
    re.compile(r"我(?:对|被)([^，。！？、；：\s「」“”‘’]{1,12})(?:圈粉|上头|无法自拔)"),
    # 「我超爱喝咖啡」——P1 fixture 里要求「照常」的一句，旧正则接不住。
    # (?!上) 是为了不把「我爱上了摄影」切成「上了摄影」（爱上那条在上面先命中）。
    re.compile(r"我(?:很|超|特别|最|好)爱(?!上)([^，。！？、；：\s「」“”‘’]{1,12})"),
]

# 首字是这些词的一律丢掉（「你」「我们的」不是兴趣）
_PRONOUN_HEADS = ("你", "我", "他", "她", "它")

# 抓出来的东西如果落在这些词里，一律丢掉——它们不是「东西」。
_STOPWORDS = {
    "你", "我", "他", "她", "它", "你们", "我们", "他们", "这个", "那个",
    "这样", "那样", "这种", "那种", "什么", "怎么", "为什么", "是不是",
    "一直", "经常", "总是", "真的", "有点", "非常", "特别", "一点",
    "东西", "事情", "感觉", "地方", "时候", "样子", "关系",
}

# 太短没意义（1 字），太长多半是句子碎片
_MIN_LEN = 2
_MAX_LEN = 12

# 同一话题要出现这么多次，才算「他喜欢的东西」。
# 2026-11-12 起它只作 **旧数据 / 状态机兜底**：没有 signals 的话题仍按它判，
# 有 signals 的话题由状态机说了算（见 interest_verdict）。
CONFIRM_MENTIONS = 2
# 超过这么多天没再提，就不再主动说（但也不删——记忆不该因为沉默而消失）
STALE_DAYS = 120

# 弱信号要在这么多个**不同日子**里重复出现，才算攒成结论（P1）。
#
# 为什么话题层还要这一条：admission 的权重按「单条记忆」校准
# （一次 explicit = 1.0，T_CONFIRMED = 1.2，passive 只有 0.15），所以 3 条
# passive（0.15 × 0.5 强度 ≈ 0.075/条）永远够不到 CONFIRMED。而「他喜欢什么」
# 本来就不是一次表态，是**跨天反复出现**——这条规则把「重复本身」算进结论，
# 同时仍然受状态机的 veto 管（明确否定 → NEEDS_REVIEW，衰减完 → STALE）。
PASSIVE_CONFIRM_DAYS = 3

# 每个话题最多留这么多条信号。40 条足够覆盖几个月，也不会让 interests.json 失控。
SIGNAL_KEEP = 40
SIGNAL_TIERS = ("passive", "attention", "engage", "explicit")
DEFAULT_SIGNAL_CONFIDENCE = 0.5
# observe() 抓到的明确句式 = 他自己在说自己的事，按 engage 记；
# 手动 add() 是「模型确认他在说喜欢」，按 explicit 记。
OBSERVE_TIER = "engage"
OBSERVE_CONFIDENCE = 0.7
MANUAL_CONFIDENCE = 1.0

# 话题语义归并：新话题和已有话题相似度超过这个值，就归并而不是新建。
# 只归并近义词（吉他/吉它、拍照/摄影）。阈值比情绪高，别把「吉他/贝斯」也合了。
TOPIC_MERGE_THRESHOLD = 0.85


# ---------------------------------------------------------------- 归属判别
#
# PersonaMem-v2 实测：让模型判断「这句话说的是别人的偏好」只有 17.5% 正确。
# 误记归属（把「我朋友超喜欢滑雪」记成「他喜欢滑雪」）比抓不到更糟——它会变成
# 一张永远错的画像，而且越用越像真的。所以这里用纯规则挡：宁可漏，不可记错。
#
# 只看命中位置**前面 12 个字**的窗口：
#   第三人称 / 关系词 / 「帮·替·给…（看/选/买/挑）」→ other
#   假设词 → hypothetical
# 「吉他」里有「他」，所以第三人称必须按 **代词** 匹配（前面得是标点/常见虚词），
# 不能整字包含——2026-11-07 就因为这个把含「他」的词一网打尽了。
OWNERSHIP_WINDOW = 12

_OTHER_RELATIONS = (
    "朋友", "同事", "客户", "老板", "领导", "对象", "老婆", "老公",
    "我妈", "我爸", "我哥", "我姐", "我妹", "我弟", "我姐", "我娃",
    "儿子", "女儿", "同学", "室友", "闺蜜", "哥们", "兄弟", "室友",
    "他们", "她们", "别人", "人家", "大家", "家人", "家里",
)

_HYPOTHETICAL = (
    "如果", "假如", "要是", "假设", "万一", "比如说", "若是", "倘若", "换成",
)

# 代词：前面得是句首或分隔符/常见虚词，否则「吉他」「他人」会误伤。
_PRONOMINAL = re.compile(
    r"(?:^|[\s，。！？、；：,.;!?的和跟与对让说给替帮在是被把从向往为])[他她它]"
)

# 「帮我妈挑个手机」「替他选一个」——动作的受益者不是他本人。
_HELP_VERB = re.compile(r"[帮替给][^，。！？、；：\s]{0,6}(?:看|选|买|挑|做|找|订|约|查)")


def _ownership_verdict(text: str, start: int, end: int) -> str:
    """这段是不是在说「他自己」：'self' / 'other' / 'hypothetical'。

    看命中位置**前 12 字**的窗口（不是整句）：中文的从属结构几乎都在前面。
    `start` / `end` 是正则在原文里的命中区间。
    """
    src = str(text or "")
    start = max(0, int(start))
    end = max(start, int(end))
    window = src[max(0, start - OWNERSHIP_WINDOW):start]
    if not window:
        return "self"
    if any(m in window for m in _HYPOTHETICAL):
        return "hypothetical"
    if _HELP_VERB.search(window):
        return "other"
    if any(m in window for m in _OTHER_RELATIONS):
        return "other"
    if _PRONOMINAL.search(window):
        return "other"
    return "self"


# ---------------------------------------------------------------- 证据 → 状态机


def _norm_tier(tier: Any) -> str:
    """把 tier 收成 admission 认的四个值；认不出就按最弱的 passive 算。"""
    t = str(getattr(tier, "value", tier) or "").strip().lower()
    return t if t in SIGNAL_TIERS else "passive"


def _evidence(signals: Iterable[Any]) -> List[Dict[str, Any]]:
    """把存下来的信号翻译成 admission.aggregate_interest 认的证据形状。"""
    out: List[Dict[str, Any]] = []
    for s in signals or []:
        if not isinstance(s, dict):
            continue
        try:
            polarity = -1 if int(s.get("polarity", 1) or 1) < 0 else 1
        except (TypeError, ValueError):
            polarity = 1
        try:
            confidence = float(s.get("confidence", DEFAULT_SIGNAL_CONFIDENCE))
        except (TypeError, ValueError):
            confidence = DEFAULT_SIGNAL_CONFIDENCE
        try:
            at = float(s.get("at") or 0.0)
        except (TypeError, ValueError):
            at = 0.0
        out.append(
            {
                "tier": _norm_tier(s.get("tier")),
                "polarity": polarity,
                "strength": max(0.0, min(1.0, confidence)),
                "occurred_at": at,
                "source_kind": str(s.get("source") or "dialog"),
            }
        )
    return out


def _distinct_days(evidence: Iterable[Dict[str, Any]]) -> int:
    """这些证据落在几个不同的（UTC）日子上。跨天是状态机的硬门槛之一。"""
    days = set()
    for e in evidence:
        ts = float(e.get("occurred_at") or 0.0)
        if ts <= 0:
            continue
        days.add(datetime.fromtimestamp(ts, timezone.utc).date())
    return len(days)


def interest_verdict(
    signals: Iterable[Any],
    *,
    count: int = 0,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """一个话题算不算「已确认」。返回 {confirmed, state, net, days, reason}。

    判定顺序（状态机优先，count 只兜底）：
      1. 有信号 → aggregate_interest + interest_state 说了算；
      2. 明确否定（NEEDS_REVIEW）/ 衰减完（STALE）→ **一票否决**，不许确认；
      3. CONFIRMED → 确认（状态机原生的强路径）；
      4. CANDIDATE + 跨天（或旧规则的重复次数）→ 确认；
      5. WEAK_SIGNAL + 跨 ≥PASSIVE_CONFIRM_DAYS 天重复 → 确认（弱证据攒成结论）；
         或证据量已到 CANDIDATE 级、且明确提到过 ≥2 次 → 确认（兼容旧行为）；
      6. 没有 signals（2026-11-12 之前的旧数据）→ 退回 count >= CONFIRM_MENTIONS。
    """
    now = time.time() if now is None else now
    ev = _evidence(signals)
    if not ev:
        return {
            "confirmed": count >= CONFIRM_MENTIONS,
            "state": "legacy",
            "net": 0.0,
            "days": 0,
            "reason": "legacy_count_fallback",
        }

    agg = aggregate_interest(ev, now)
    net = float(agg["net"])
    st = interest_state(net, ev, now)
    days = _distinct_days(ev)

    def out(confirmed: bool, reason: str) -> Dict[str, Any]:
        return {
            "confirmed": confirmed,
            "state": st.value,
            "net": round(net, 4),
            "days": days,
            "reason": reason,
        }

    # 明确否定：一票否决，不许被任何「曾经」抬回来
    if st in (InterestState.NEEDS_REVIEW, InterestState.REJECTED):
        return out(False, f"state_machine:{st.value}")

    # 兴趣淡了 ≠ 没喜欢过。
    # 人不会因为三个月没提吉他，就忘了自己喜欢吉他——所以只要曾经攒到过
    # 「确认级」（跨 2 天出现，且含明确句式/主动表态；或弱信号跨 3 天），
    # 就永久认账。衰减只影响「最近还提不提」（STALE_DAYS 仍管主动提起），
    # 不影响「他喜欢过这个」这个事实。
    # 反例（必须还能降级）：只有一条 PASSIVE、或证据净值为负（被否定过）。
    ever_strong = days >= 2 and any(
        str(e.get("tier", "")).lower() in ("engage", "explicit") for e in ev
    )
    if net > 0 and (ever_strong or days >= PASSIVE_CONFIRM_DAYS):
        return out(True, "durable:ever_confirmed")

    if st is InterestState.CONFIRMED:
        return out(True, "state_machine:confirmed")
    if st is InterestState.STALE:
        return out(False, "state_machine:stale")
    if st is InterestState.CANDIDATE:
        ok = days >= 2 or count >= CONFIRM_MENTIONS
        return out(ok, "state_machine:candidate" if ok else "pending:candidate_one_day")
    if st is InterestState.WEAK_SIGNAL:
        if days >= PASSIVE_CONFIRM_DAYS:
            return out(True, f"state_machine:weak_repeated_{days}d")
        if count >= CONFIRM_MENTIONS and net >= T_CANDIDATE:
            return out(True, "state_machine:repeat_mention")
        return out(False, "pending:weak_signal")
    return out(False, f"pending:{st.value}")


def _cosine(a: List[float], b: List[float]) -> float:
    """两向量余弦。embedder 已 L2 归一化，等价于点积。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Topic:
    topic: str
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    examples: List[str] = field(default_factory=list)
    # 每条弱证据：{polarity, tier, quote, confidence, at}
    # 这是「他顺口提了一句」的原始记录，不是确认——确认由 interest_verdict 算。
    signals: List[Dict[str, Any]] = field(default_factory=list)

    def add_signal(
        self,
        *,
        polarity: int = 1,
        tier: str = "passive",
        quote: str = "",
        confidence: float = DEFAULT_SIGNAL_CONFIDENCE,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """追加一条证据（只记，不判）。返回落下的那条。"""
        now = time.time() if now is None else now
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = DEFAULT_SIGNAL_CONFIDENCE
        sig = {
            "polarity": -1 if int(polarity or 1) < 0 else 1,
            "tier": _norm_tier(tier),
            "quote": str(quote or "")[:120],
            "confidence": round(max(0.0, min(1.0, conf)), 3),
            "at": float(now),
        }
        self.signals.append(sig)
        if len(self.signals) > SIGNAL_KEEP:
            del self.signals[:-SIGNAL_KEEP]
        self.first_seen = min(self.first_seen, now)
        self.last_seen = max(self.last_seen, now)
        return sig

    def last_quote(self) -> str:
        """最拿得出手的一句原话：优先最近的信号，其次旧的 examples。"""
        for sig in reversed(self.signals):
            q = str(sig.get("quote") or "").strip()
            if q:
                return q
        return str(self.examples[-1]).strip() if self.examples else ""

    def to_dict(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        days = max(0.0, (now - self.last_seen) / 86400.0)
        verdict = interest_verdict(self.signals, count=self.count, now=now)
        return {
            "topic": self.topic,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "days_since": round(days, 2),
            # 语义没变：`confirmed` 仍是「能不能当成他喜欢的东西说出口」。
            # 只是它现在来自状态机（见 interest_verdict），不再只数次数。
            "confirmed": verdict["confirmed"],
            "interest_state": verdict["state"],
            "evidence_net": verdict["net"],
            "evidence_days": verdict["days"],
            "why": verdict["reason"],
            "signal_count": len(self.signals),
            "last_quote": self.last_quote(),
            # 待观察那一段要「带原话」，最近 3 条够用
            "signals": [dict(s) for s in self.signals[-3:]],
            "examples": list(self.examples[-3:]),
        }


def extract_topic_verdicts(text: str) -> List[Dict[str, str]]:
    """抓出所有候选话题，连归属判定一起返回（给 fixture / 调试看，不写库）。"""
    src = str(text or "")
    found: Dict[str, Dict[str, str]] = {}
    for pat in _PATTERNS:
        for m in pat.finditer(src):
            raw = m.group(1).strip("的了着过是有没有")
            raw = raw.strip("的把被给")
            # 「动画是虫师」→ 只留「动画」：是字后面的多半是补语
            if "是" in raw:
                raw = raw.split("是", 1)[0]
            if not (_MIN_LEN <= len(raw) <= _MAX_LEN):
                continue
            if raw in _STOPWORDS:
                continue
            # 只看首字是不是代词。「吉他」含「他」但它不是代词——
            # 用整字包含判断会把所有含他/她/它的词一网打尽（2026-11-07 踩过）。
            if raw[0] in _PRONOUN_HEADS:
                continue
            verdict = _ownership_verdict(src, m.start(), m.end())
            quote = src[max(0, m.start() - 6): m.end()][:40]
            old = found.get(raw)
            # 同一个话题既被判成别人的、又被判成他自己的 → 保留 self
            if old is None or (old["verdict"] != "self" and verdict == "self"):
                found[raw] = {"topic": raw, "verdict": verdict, "quote": quote}
    return list(found.values())


def extract_topics(text: str) -> List[str]:
    """从一句话里抓他明确说过的兴趣。抓不到就是空列表——不硬猜。

    别人的偏好 / 假设句在这里就被丢掉了（`_ownership_verdict`），
    不会变成「他喜欢X」。
    """
    return [d["topic"] for d in extract_topic_verdicts(text) if d["verdict"] == "self"]


class InterestBook:
    """兴趣画像的持久化。池子很小，整体读进内存就够。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.topics: Dict[str, Topic] = {}
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001 —— 坏了就当没记过，不要卡住服务
            return
        for row in data.get("topics", []) if isinstance(data, dict) else []:
            try:
                t = Topic(**{k: row[k] for k in row if k in Topic.__dataclass_fields__})
                # 旧文件的 signals 可能是 null / 形状不对：收成 list，别让后面炸
                if not isinstance(t.signals, list):
                    t.signals = []
                t.signals = [s for s in t.signals if isinstance(s, dict)]
                if not isinstance(t.examples, list):
                    t.examples = []
                self.topics[t.topic] = t
            except Exception:  # noqa: BLE001
                continue

    def save(self) -> None:
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {"version": 2, "topics": [asdict(t) for t in self.topics.values()]},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass

    # ---------- 写入 ----------
    def _merge_similar(self, topic: str, embedder: Any) -> Optional[str]:
        """找已有话题里和它语义最接近的，足够近就归并（返回已有话题名）。

        解决「吉他」和「吉它」被记成两个话题、count 永远到不了确认门槛的问题。
        没有真句向量（is_semantic）就直接返回 None——不硬猜。
        """
        if not self.topics or not getattr(embedder, "is_semantic", False):
            return None
        try:
            q = embedder.encode(topic)
            best: Optional[str] = None
            best_sim = 0.0
            for existing in self.topics:
                s = _cosine(q, embedder.encode(existing))
                if s > best_sim:
                    best_sim = s
                    best = existing
            if best is not None and best_sim >= TOPIC_MERGE_THRESHOLD:
                return best
        except Exception:  # noqa: BLE001
            return None
        return None

    def observe(self, text: str, *, now: Optional[float] = None, embedder: Any = None) -> List[str]:
        """从一段话里学习。返回这次新识别/加强的话题。

        embedder 可选：真句向量时会把「吉他/吉它」归并成同一个话题，
        避免同义换说导致 count 分散、永远到不了确认门槛。

        P1：明确句式命中的，同时记一条 engage 级证据——它不是弱信号，
        但也不该绕过状态机直接算确认。别人的偏好 / 假设句在这里已经被丢掉。
        """
        now = time.time() if now is None else now
        hits = extract_topics(text)
        for topic in hits:
            key = self._merge_similar(topic, embedder) if embedder else None
            key = key or topic
            t = self.topics.get(key)
            if t is None:
                t = Topic(topic=key, first_seen=now, last_seen=now, examples=[text[:120]])
                self.topics[key] = t
            else:
                t.count += 1
                t.last_seen = now
                if len(t.examples) < 3:
                    t.examples.append(text[:120])
            t.add_signal(
                tier=OBSERVE_TIER,
                quote=text[:120],
                confidence=OBSERVE_CONFIDENCE,
                now=now,
            )
        if hits:
            self.save()
        return hits

    def observe_signal(
        self,
        topic: str,
        *,
        polarity: int = 1,
        tier: str = "passive",
        quote: str = "",
        confidence: float = DEFAULT_SIGNAL_CONFIDENCE,
        now: Optional[float] = None,
    ) -> Topic:
        """记一条弱证据（不直接算确认）——「他顺口提了一句」，不是「他喜欢这个」。

        这是「隐式偏好」的入口：模型注意到他反复提到某样东西，但没说过「我喜欢」，
        就丢一条 passive/attention 进来。单条永远不会确认；跨天攒够了，
        `interest_verdict` 才会把它判成 confirmed（见 PASSIVE_CONFIRM_DAYS）。
        """
        name = str(topic or "").strip()
        if not name:
            raise ValueError("话题不能为空")
        if name in _STOPWORDS:
            raise ValueError(f"「{name}」不是一个话题")
        if name[0] in _PRONOUN_HEADS and len(name) <= 2:
            raise ValueError(f"「{name}」是代词，不是话题")
        now = time.time() if now is None else now
        t = self.topics.get(name)
        if t is None:
            # count=0：这是线索，不是「他明确提过一次」
            t = Topic(topic=name, count=0, first_seen=now, last_seen=now)
            self.topics[name] = t
        t.add_signal(
            polarity=polarity,
            tier=tier,
            quote=quote,
            confidence=confidence,
            now=now,
        )
        self.save()
        return t

    def add(self, topic: str, *, example: str = "") -> Topic:
        """手动补一条（模型自己判断「他在说喜欢这个」时用）。

        P1：算一条 explicit 证据（比 observe 的明确句式更强），但仍然走状态机——
        「一次判断」不等于「确认」，除非证据量 + 跨天都够。
        """
        topic = str(topic or "").strip()
        if not topic:
            raise ValueError("话题不能为空")
        now = time.time()
        t = self.topics.get(topic)
        if t is None:
            t = Topic(topic=topic, first_seen=now, last_seen=now, examples=[example[:120]] if example else [])
            self.topics[topic] = t
        else:
            t.count += 1
            t.last_seen = now
        t.add_signal(
            tier="explicit",
            quote=example,
            confidence=MANUAL_CONFIDENCE,
            now=now,
        )
        self.save()
        return t

    # ---------- 查询 ----------
    def snapshot(self, *, limit: int = 20) -> Dict[str, Any]:
        now = time.time()
        items = [t.to_dict(now=now) for t in self.topics.values()]
        # 已确认的排前面，再按次数 / 证据量 / 最近一次
        items.sort(
            key=lambda d: (
                bool(d["confirmed"]),
                int(d["count"]),
                float(d["evidence_net"]),
                float(d["last_seen"]),
            ),
            reverse=True,
        )
        shown = items[: max(1, limit)]
        return {
            "items": shown,
            "count": len(shown),
            "confirmed_count": len([d for d in items if d["confirmed"]]),
            "pending_count": len([d for d in items if not d["confirmed"]]),
        }

    def top_confirmed(self, *, limit: int = 3, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """值得说出口的那几个：状态机判确认、而且不是很久以前的事。"""
        now = time.time() if now is None else now
        rows: List[Dict[str, Any]] = []
        for t in self.topics.values():
            if (now - t.last_seen) / 86400.0 > STALE_DAYS:
                continue
            d = t.to_dict(now=now)
            if d["confirmed"]:
                rows.append(d)
        rows.sort(
            key=lambda d: (float(d["evidence_net"]), int(d["count"]), float(d["last_seen"])),
            reverse=True,
        )
        return rows[:limit]
