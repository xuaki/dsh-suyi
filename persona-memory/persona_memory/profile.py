# -*- coding: utf-8 -*-
"""常驻小画像：一块 100~150 token 的「他是什么样的」。

要解决的问题：

    每次回答前都要先检索一遍记忆。检索很贵——不是 token 贵，是**噪声**贵：
    翻回来一堆沾边的片段，真正该当底色的那几句反而被淹掉。

依据（O-Mem, Table 7）：去掉 persona 属性，任务性能只降 0.35，但检索长度从
6,499 涨到 28,555（4.4 倍）。换句话说，一小块常驻的画像能替掉大半检索。

所以这里把三个慢变量压成一块人类可读的短画像：
  - 已确认的兴趣（≤3）
  - 张力最高的执念（≤1）
  - 最近翻过的车（≤1）

刻意窄：只放**慢变量**。「最近聊了什么」是 su_recall 的活，不往这里塞。
空的部分整行省略——不写「暂无」，没有就是没有。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

# 整块上限。超了就说明该砍内容，不是该调大——常驻的东西越短越值钱。
PROFILE_MAX_CHARS = 400
# 每行上限。超过 40 字读起来就不再是「一眼看完」了。
PROFILE_LINE_CHARS = 40

_WS = re.compile(r"\s+")


def _clip(text: Any, limit: int = PROFILE_LINE_CHARS) -> str:
    """压成一行 + 截断。宁可截断，也不让某一行膨胀成句子。"""
    s = _WS.sub(" ", str(text or "")).strip()
    if len(s) <= limit:
        return s
    return s[: max(1, limit - 1)] + "…"


def _interest_topics(src: Any, limit: int = 3) -> List[str]:
    """拿已确认的兴趣。接受 InterestBook / snapshot dict / list，形状不对就当空。"""
    try:
        if hasattr(src, "top_confirmed"):
            rows = src.top_confirmed(limit=limit)
        elif isinstance(src, dict):
            rows = [x for x in (src.get("items") or []) if isinstance(x, dict) and x.get("confirmed") is True]
        elif isinstance(src, (list, tuple)):
            rows = [x for x in src if isinstance(x, dict) and x.get("confirmed") is True]
        else:
            rows = []
    except Exception:  # noqa: BLE001 —— 画像坏了不该拖垮 /profile
        rows = []
    out: List[str] = []
    for row in rows:
        t = str((row or {}).get("topic") or "").strip()
        if t and t not in out:
            out.append(t)
    return out[:limit]


def _top_gap(src: Any) -> Optional[Dict[str, Any]]:
    """张力最高的那条 open 执念。"""
    try:
        if hasattr(src, "top_open"):
            g = src.top_open()
            return g.to_dict() if g is not None and hasattr(g, "to_dict") else None
        if isinstance(src, dict):
            rows = [x for x in (src.get("items") or []) if isinstance(x, dict) and x.get("status") == "open"]
        elif isinstance(src, (list, tuple)):
            rows = [x for x in src if isinstance(x, dict) and x.get("status") == "open"]
        else:
            return None
        rows.sort(key=lambda x: float(x.get("tension") or 0.0), reverse=True)
        return rows[0] if rows else None
    except Exception:  # noqa: BLE001
        return None


def _recent_failure(src: Any, max_days: float = 7.0) -> Optional[Dict[str, Any]]:
    """最近翻过的那辆车（默认只看 7 天内的，旧账不翻）。"""
    try:
        if hasattr(src, "recent"):
            rows = src.recent(limit=1, max_days=max_days)
            return rows[0] if rows else None
        if isinstance(src, dict):
            rows = [x for x in (src.get("items") or []) if isinstance(x, dict)]
        elif isinstance(src, (list, tuple)):
            rows = [x for x in src if isinstance(x, dict)]
        else:
            return None
        rows = [x for x in rows if float(x.get("days_since", 0.0) or 0.0) <= max_days]
        rows.sort(key=lambda x: float(x.get("created_at") or 0.0), reverse=True)
        return rows[0] if rows else None
    except Exception:  # noqa: BLE001
        return None


def build_profile_digest(
    interests: Any,
    gaps: Any,
    failures: Any,
    recent_memories: Optional[Iterable[Any]] = None,
) -> str:
    """≤400 字的人类可读画像，每行 ≤40 字。

    参数都接受「对象或 dict」：服务里传 InterestBook / GapStore / FailureBook，
    测试里传 snapshot 形状的 dict 也行。

    `recent_memories` 暂时不用：常驻画像只放慢变量，最近聊了什么归 su_recall。
    留着这个参数是为了调用方不用为「以后要不要放一句近况」改签名。
    """
    lines: List[str] = []

    topics = _interest_topics(interests, limit=3)
    if topics:
        lines.append(_clip("喜欢：" + "、".join(topics)))

    # 执念只放绷得起来的那条：刚种下的还没到注入阈值，不必占常驻位置。
    gap = _top_gap(gaps)
    if gap:
        # 正文压到 28 字，给「惦记：」和「（绷了 N 天）」留位置，整行才不会超 40
        gap_text = _clip(gap.get("text"), 28)
        if gap_text:
            days = float(gap.get("age_days") or 0.0)
            when = f"绷了 {round(days)} 天" if days >= 1 else "刚种下"
            lines.append(_clip(f"惦记：{gap_text}（{when}）"))

    failure = _recent_failure(failures)
    if failure:
        # 同理：正文 26 字 + 理由 12 字，整行控制在 40 字内
        fail_text = _clip(failure.get("text"), 26)
        if fail_text:
            reason = _clip(failure.get("reason"), 12)
            suffix = f"（{reason}）" if reason else ""
            lines.append(_clip(f"别再犯：{fail_text}{suffix}"))

    if not lines:
        return ""

    digest = "\n".join(lines)
    if len(digest) > PROFILE_MAX_CHARS:
        # 三行 × 40 字正常到不了这里；真到了就按行砍，别留半截句子。
        kept: List[str] = []
        used = 0
        for line in lines:
            if used + len(line) + (1 if kept else 0) > PROFILE_MAX_CHARS:
                break
            kept.append(line)
            used += len(line) + (1 if kept else 0)
        digest = "\n".join(kept)
    return digest
