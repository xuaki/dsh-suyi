# -*- coding: utf-8 -*-
"""失败记忆：我自己翻过的车。

要解决的问题：

    他纠正我一次（「你记错了」），那条记忆就 invalidated 了；
    但「下次别再犯」的教训没地方存——下一轮我就忘了。
    结果同一个错，他能纠正我第二次、第三次。

记忆库里全是「关于他」的事（兴趣、偏好、对话），唯独没有
「关于我自己哪里错了」。这里补的就是这一本账。

设计上刻意窄：
  - 只记**具体的翻车句/事**，不记空泛的「我要更小心」；
  - 按 `text` 去重，同一句错犯两次就 count+1、created_at 刷新，
    而不是堆两条一模一样的教训；
  - 注入只取最近 `max_days` 天、最多 `limit` 条——旧账不翻，
    不然每轮都背着半年前的一个口误说话，人会变得畏手畏脚。

灵感来自 gaps.py：它管「没弄明白的事」，这里管「已经弄错的事」。
一个往前绷着，一个往后防着，都是为了让下一轮别再犯同样的错。
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# 池子上限：200 条够用很久。超出丢最老的——翻车账本不是档案馆。
MAX_ENTRIES = 200

# 注入时正文截断长度。太长会挤占提示词预算，太短又说不清犯了什么错。
TEXT_LIMIT = 120

# 落盘时的存储上限。比注入长得多——账本留着原句，注入时才截，
# 不然「为什么错」的上下文会被写死成半句话。
STORE_TEXT_LIMIT = 1000

# 分类。纠正自动记账走「记忆」；手动补的可以是别的。
CATEGORIES = ("记忆", "事实", "情绪", "技术", "其它")
DEFAULT_CATEGORY = "其它"

_WS = re.compile(r"\s+")


def summarize(text: str, *, limit: int = TEXT_LIMIT) -> str:
    """把一段话压成一行摘要：折叠空白 + 截断。"""
    s = _WS.sub(" ", str(text or "")).strip()
    return s[: max(1, int(limit))]


@dataclass
class Failure:
    """一条「我说错/做错的事」。"""

    text: str
    reason: str = ""
    corrected_to: str = ""
    category: str = DEFAULT_CATEGORY
    created_at: float = field(default_factory=time.time)
    # 同一句话重复翻车时累加。1 = 只犯过一次。
    count: int = 1
    id: str = ""

    def to_dict(self, *, now: Optional[float] = None, text_limit: int = 0) -> Dict[str, Any]:
        now = time.time() if now is None else now
        text = summarize(self.text, limit=text_limit) if text_limit else self.text
        days = max(0.0, (now - self.created_at) / 86400.0)
        return {
            "id": self.id,
            "text": text,
            "reason": self.reason,
            "corrected_to": self.corrected_to,
            "category": self.category,
            "count": self.count,
            "created_at": self.created_at,
            "days_since": round(days, 2),
        }


class FailureBook:
    """翻车账本的持久化。池子很小，整体读进内存就够。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.failures: List[Failure] = self._load()

    # ---------- 持久化 ----------
    def _load(self) -> List[Failure]:
        if not self.path or not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001 —— 账本坏了就当空的，不要卡住服务
            return []
        out: List[Failure] = []
        for row in data.get("failures", []) if isinstance(data, dict) else []:
            try:
                out.append(Failure(**{k: row[k] for k in row if k in Failure.__dataclass_fields__}))
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
                json.dump(
                    {"version": 1, "failures": [asdict(x) for x in self.failures]},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            # 原子替换：写一半崩了也不留半本账
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 —— 落盘失败不该打断对话
            pass

    # ---------- 写入 ----------
    def add(
        self,
        text: str,
        *,
        reason: str = "",
        corrected_to: str = "",
        category: str = DEFAULT_CATEGORY,
        now: Optional[float] = None,
    ) -> Failure:
        """记一条翻车。同一句话重复记时累加次数、刷新时间和理由。"""
        text = summarize(text, limit=STORE_TEXT_LIMIT)
        if not text:
            raise ValueError("翻车内容不能为空")
        now = time.time() if now is None else now
        category = str(category or "").strip() or DEFAULT_CATEGORY
        reason = str(reason or "").strip()
        corrected_to = str(corrected_to or "").strip()

        # 去重：同一句话不记两遍。重复犯说明这条更该被记住，所以 count+1、
        # created_at 刷新（它会重新进入「最近」的注入窗口），理由以新的为准。
        for old in self.failures:
            if old.text != text:
                continue
            old.count += 1
            old.created_at = now
            if reason:
                old.reason = reason
            if corrected_to:
                old.corrected_to = corrected_to
            self.save()
            return old

        failure = Failure(
            id=f"fail_{uuid.uuid4().hex[:12]}",
            text=text,
            reason=reason,
            corrected_to=corrected_to,
            category=category,
            created_at=now,
        )
        self.failures.append(failure)
        self._trim()
        self.save()
        return failure

    def _trim(self) -> None:
        """超过上限就丢最老的（按 created_at，不是按插入顺序）。"""
        while len(self.failures) > MAX_ENTRIES:
            oldest = min(range(len(self.failures)), key=lambda i: self.failures[i].created_at)
            del self.failures[oldest]

    # ---------- 查询 ----------
    def recent(self, *, limit: int = 5, max_days: float = 7, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """注入用：最近 `max_days` 天、最多 `limit` 条，正文截断到 120 字。

        max_days <= 0 表示不限时间（调试/看板用），默认 7 天。
        """
        now = time.time() if now is None else now
        limit = int(limit)
        if limit <= 0:
            return []
        rows = list(self.failures)
        if max_days and max_days > 0:
            cutoff = now - float(max_days) * 86400.0
            rows = [x for x in rows if x.created_at >= cutoff]
        rows.sort(key=lambda x: x.created_at, reverse=True)
        return [x.to_dict(now=now, text_limit=TEXT_LIMIT) for x in rows[:limit]]

    def snapshot(self) -> Dict[str, Any]:
        """全量（给看板/工具用）。新的在前。"""
        now = time.time()
        rows = sorted(self.failures, key=lambda x: x.created_at, reverse=True)
        items = [x.to_dict(now=now) for x in rows]
        return {"items": items, "count": len(items), "total": len(items)}
