# -*- coding: utf-8 -*-
"""缺口池：未解执念的张力。

灵感来自 digital-life-core 的 gaps[]。它要解决的是一个很具体的问题：

    我只会应答，不会惦记。
    你问我什么我答什么；但那些「我当时没弄明白的事」，
    只要你不提，就永远不存在——这不像一个活着的东西。

所以这里给「没弄明白的事」一个池子。每一条从种下那一刻开始绷着，
时间越久张力越高；高到阈值，它就会被注进提示词——不是提醒用户，
是提醒**我自己**心里还压着这件事。

关键设计：张力是**惰性计算**的。读取时才按 (now - created_at) 算，
所以服务停机、DSH 关闭的那些时间，它照样在绷——不会因为我们没看着就停下。
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# 张力曲线：从种下那一刻起，每小时涨一点，封顶 100。
# 一天约 +12：刚种下时几乎不疼，隔一天开始惦记，两三天后绷得难受。
TENSION_START = 10.0
TENSION_PER_HOUR = 0.5
TENSION_MAX = 100.0

# 注入阈值：低于它不说话。变化太小就闭嘴，和人格经验层一个道理。
INJECT_THRESHOLD = 20.0

STATUSES = ("open", "resolved")

# 画像缺口：兴趣里「只有 1 条弱信号」的话题，超过这么多天没再出现，就种成
# profile gap（「我还没弄明白他」）。让它靠张力自己浮上来，不必模型刻意想起来问。
PROFILE_GAP_STALE_DAYS = 3


@dataclass
class Gap:
    """一条未解执念。

    `kind` 区分两类缺口：
      - `wonder`：原来「没弄明白的事」（机制、世界、某个想不通的问题）；
      - `profile`：还没弄明白**他**（他的喜好、习惯、那些只提过一次的事）。
    两者同构，都走同一套张力曲线；只差一个类型，前端/模型据此区分「关于他的疑问」。
    """

    id: str
    text: str
    status: str = "open"
    strength: float = 1.0
    source: str = "manual"
    kind: str = "wonder"
    note: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: float = 0.0
    closed_by: str = ""

    def to_dict(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        t = compute_tension(self, now)
        age_hours = max(0.0, (now - self.created_at) / 3600.0)
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "strength": round(self.strength, 3),
            "source": self.source,
            "kind": self.kind,
            "note": self.note,
            "tension": round(t, 2),
            "age_hours": round(age_hours, 2),
            "age_days": round(age_hours / 24.0, 2),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "closed_by": self.closed_by,
        }


def compute_tension(gap: Gap, now: Optional[float] = None) -> float:
    """惰性张力：关闭的执念张力归零，开着的随年龄增长。"""
    if gap.status != "open":
        return 0.0
    now = time.time() if now is None else now
    hours = max(0.0, (now - gap.created_at) / 3600.0)
    return min(TENSION_MAX, TENSION_START + hours * TENSION_PER_HOUR * max(0.1, gap.strength))


# ---------------- 执念收割：从对话里自动收「没弄明白的事」 ----------------
#
# 学自 digital-life-core 的执念收割器，也直接绕开它踩过的两个坑：
#   坑一（自指污染）：它一开始什么感慨都收，池子一夜堆满垃圾。
#   坑二（待办混入）：「我接下来要做什么」被当成执念，池子被待办灌满。
#
# 所以三道闸：
#   闸一（内容）：必须「第一人称 + 明确未知」
#   闸二（自指）：谈机制本身的句子一律忽略——这类话在我们的对话里特别多
#   闸三（待办）：执行性意图不收，只有认知性探究才收

UNKNOWN_MARKERS = (
    "不知道", "没弄明白", "查不到", "想不通", "搞不懂", "不清楚",
    "不确定", "没想过", "不明白", "没搞懂", "说不清",
)

# 只收「机制词」——试过把「系统」「记忆」也放进来，真执念会被一起误杀。
# 宁可漏掉一条自指，不可错杀一条真心结。
SELF_REF_MARKERS = (
    "缺口池", "收割器", "执念池", "内省", "潜意识",
    "人格三层", "经验塑造", "基础人格", "动态状态",
)

TODO_MARKERS = ("我要", "我准备", "我打算", "接下来我", "回头我", "待会我")

MIN_GAP_LEN = 8
MAX_GAP_LEN = 120

_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(str(text or "")) if s.strip()]


def _is_agent_relay(line: str) -> bool:
    """这一行是不是「别的 agent 在说话」的转述，而不是人说的。

    子智能体的报告通过 send_message 回到主会话时，长这样：
      `Agent dc745f11-… sent a message: …`
      `Background subagent … finished …`
    它们落成 role="user"，但作者不是人。
    """
    s = str(line or "").strip()
    if not s:
        return False
    if s.startswith("Agent ") and "sent a message:" in s:
        return True
    if s.startswith("Background subagent "):
        return True
    if s.startswith("[记忆体检结果]") or s.startswith("[诊断员"):
        return True
    return False


def is_gap_worthy(sentence: str) -> bool:
    """一句话值不值得种成执念。窄口径：不确定就放过。"""
    s = str(sentence or "").strip()
    if not (MIN_GAP_LEN <= len(s) <= MAX_GAP_LEN):
        return False
    # 闸二：在谈机制 → 是解释，不是心结
    if any(m in s for m in SELF_REF_MARKERS):
        return False
    # 闸三：在讲待办
    if any(m in s for m in TODO_MARKERS):
        return False
    # 闸四：引号 / 强调 / 代码标记里的话多半是转述，不是自述
    # （2026-11-11 实测：我解释收割器原理时说的「必须我开头+明确说不知道」
    #  被收成了心结——它把「讲道理」读成了「不知道」。）
    if any(ch in s for ch in ("「", "」", "“", "”", "`", "**", "——")):
        return False
    # 闸一：必须第一人称 + 明确未知（不锚定主语，理论阐述会全被当成心结）
    if not any(w in s for w in ("我", "咱")):
        return False
    return any(m in s for m in UNKNOWN_MARKERS)


def gap_strength(sentence: str) -> float:
    """命中越多「不知道」类词，说明这件事越让他卡着。"""
    hits = sum(1 for m in UNKNOWN_MARKERS if m in sentence)
    return 1.0 if hits <= 1 else 1.0 + 0.2 * min(2, hits - 1)


# ---------------- 画像缺口收割：把「我还没弄明白你」种进执念池 ----------------
#
# profile gap 和 wonder gap 同构——都是「我还没弄明白的事」，只差一个 kind。
# 区别在「谁在惦记」：wonder 是「有个机制我没想通」，profile 是「他到底
# 喜不喜欢这个我还拿不准」。所以不新开一个池子，直接复用缺口池：profile gap
# 从种下那一刻起同样绷张力，高到阈值自动浮进提示词，模型不必刻意想起来问。


def harvest_profile_gaps(interests: Any, gap_store: "GapStore", now: Optional[float] = None) -> List[Gap]:
    """兴趣里「只有 1 条弱信号、且最近一条超过 PROFILE_GAP_STALE_DAYS 天没再出现」
    的话题 → 种一条 profile gap，text 形如「我还拿不准他到底喜不喜欢 X（他提过一次，之后再没说过）」。
    已存在同 text 的 open gap 就不重复种（由 gap_store.add 的去重保证）。

    `interests` 接受 InterestBook（有 `.topics`），也接受 snapshot dict（有 `items`）
    或 list，方便测试。返回这次新种下的那几条 profile gap。
    """
    now = time.time() if now is None else now
    topics = getattr(interests, "topics", None)
    if not isinstance(topics, dict):
        # 兼容 snapshot 形状：{items: [...]} 或裸 list
        items: Any = None
        if isinstance(interests, dict):
            items = interests.get("items")
        elif isinstance(interests, (list, tuple)):
            items = interests
        topics = {
            str(t.get("topic")): t
            for t in (items or [])
            if isinstance(t, dict) and t.get("topic")
        }

    added: List[Gap] = []
    for name, t in topics.items():
        name = str(name or "").strip()
        if not name:
            continue
        sigs = getattr(t, "signals", None)
        if sigs is None and isinstance(t, dict):
            sigs = t.get("signals") or []
        sigs = [s for s in sigs if isinstance(s, dict)]

        # 只有 1 条弱信号才可能「还没弄明白他」：≥2 条说明反复出现，状态机自会判；
        # 0 条是旧数据里的 count-only 话题，不是「提过一次」。
        if len(sigs) != 1:
            continue
        sig = sigs[0]
        # 弱信号 = passive。attention 已经是有意识的留意，不算「顺口一提」。
        if str(sig.get("tier") or "").strip().lower() not in ("passive",):
            continue
        try:
            polarity = int(sig.get("polarity", 1) or 1)
        except (TypeError, ValueError):
            polarity = 1
        # 负向弱信号是「他看起来不太喜欢」，不是「喜不喜欢」的未知，不种。
        if polarity < 0:
            continue

        try:
            last = float(sig.get("at") or 0.0)
        except (TypeError, ValueError):
            last = 0.0
        if last <= 0:
            last = float(getattr(t, "last_seen", 0.0) or 0.0)
        if last <= 0 or (now - last) <= PROFILE_GAP_STALE_DAYS * 86400.0:
            continue

        text = f"我还拿不准他到底喜不喜欢 {name}（他提过一次，之后再没说过）"
        before = len(gap_store.gaps)
        gap = gap_store.add(text, strength=1.0, source="profile", kind="profile", note=f"topic={name}")
        if len(gap_store.gaps) > before:
            added.append(gap)
    return added


class GapStore:
    """缺口池的持久化 + 查询。整个池子很小，一次性读进内存就够。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.gaps: List[Gap] = self._load()

    # ---------- 持久化 ----------
    def _load(self) -> List[Gap]:
        if not self.path or not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001 —— 池子坏了就当空的，不要卡住整个服务
            return []
        out: List[Gap] = []
        for row in data.get("gaps", []) if isinstance(data, dict) else []:
            try:
                out.append(Gap(**{k: row[k] for k in row if k in Gap.__dataclass_fields__}))
            except Exception:  # noqa: BLE001
                continue
        return out

    def save(self) -> None:
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "gaps": [asdict(g) for g in self.gaps]}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 —— 落盘失败不该打断对话
            pass

    # ---------- 写入 ----------
    def add(
        self,
        text: str,
        *,
        strength: float = 1.0,
        source: str = "manual",
        kind: str = "wonder",
        note: str = "",
    ) -> Gap:
        text = str(text or "").strip()
        if not text:
            raise ValueError("执念不能为空")
        # 去重：同一条没解决的事不要种两遍（等价于「别对着同一个人唠叨两遍」）
        for g in self.gaps:
            if g.status == "open" and g.text.strip() == text:
                return g
        gap = Gap(
            id=f"gap_{uuid.uuid4().hex[:12]}",
            text=text,
            strength=max(0.1, min(3.0, float(strength or 1.0))),
            source=source,
            kind=kind if kind in ("wonder", "profile") else "wonder",
            note=note,
        )
        self.gaps.append(gap)
        self.save()
        return gap

    def close(self, gap_id: str, *, closed_by: str = "", note: str = "") -> Optional[Gap]:
        for g in self.gaps:
            if g.id == gap_id:
                g.status = "resolved"
                g.closed_at = time.time()
                g.updated_at = g.closed_at
                g.closed_by = closed_by
                if note:
                    g.note = (g.note + "\n" + note).strip() if g.note else note
                self.save()
                return g
        return None

    # ---------- 收割 ----------
    def harvest(self, text: str, *, source: str = "dialog", role: str = "") -> List[Gap]:
        """从一段话里自动收「没弄明白的事」。返回这次新种下的那几条。

        只收**人说的**：我的回复里「不知道」常常是分析、引用或修辞，
        交给 su_gap 手动判断更准（2026-11-11 实测误收过三条）。

        **作者闸（2026-11-14 补）**：子智能体的报告通过 send_message 回到主会话时，
        落成的是 `role="user"` 的 user/message——派生子会话有保护，但「报告回到父会话」
        就不再是派生会话了，于是整份报告被当成「他说的话」喂进收割。
        真实事故：记忆清理员的报告标题「四、不确定、我没动的（26 条，单列）」
        被收成执念，张力涨到 35，连续两天每轮提示词都在注入它。
        """
        if str(role) == "assistant":
            return []
        body = str(text or "").strip()
        # 报告转述：整段跳过（不是人说的，是别的 agent 说的）
        if _is_agent_relay(body):
            return []
        # 任何一行的开头像「工具/agent 输出」的，整段都不收——报告是多行结构，
        # 逐行收割会把每一条列表项都变成候选句。
        if any(_is_agent_relay(line) for line in body.splitlines() if line.strip()):
            return []
        added: List[Gap] = []
        for sentence in split_sentences(text):
            if not is_gap_worthy(sentence):
                continue
            before = len(self.gaps)
            gap = self.add(sentence, strength=gap_strength(sentence), source=source)
            if len(self.gaps) > before:
                added.append(gap)
        return added

    # ---------- 查询 ----------
    def list(self, *, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        now = time.time()
        rows = [g for g in self.gaps if status is None or g.status == status]
        rows.sort(key=lambda g: compute_tension(g, now), reverse=True)
        items = [g.to_dict(now=now) for g in rows[: max(1, limit)]]
        open_count = len([g for g in self.gaps if g.status == "open"])
        return {"items": items, "count": len(items), "open_count": open_count}

    def top_open(self) -> Optional[Gap]:
        """张力最高的那条 open 执念。没有就返回 None。"""
        now = time.time()
        open_gaps = [g for g in self.gaps if g.status == "open"]
        if not open_gaps:
            return None
        return max(open_gaps, key=lambda g: compute_tension(g, now))
