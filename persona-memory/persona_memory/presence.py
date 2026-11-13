# -*- coding: utf-8 -*-
"""在场感：记得上一次有互动是什么时候。

要解决的问题很具体：

    我不知道「你隔了多久才回来」。
    所以你隔了一天回来，我还是用十分钟前的语气迎上去——
    那不是在陪你，是在装作一直没分开。

这里只记录「上次有互动的时间」，不记录别的。怎么开口是模型的事，
这个模块只负责让模型**知道**过了多久。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class Presence:
    """一个很小的持久化时间戳。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.last_seen_at: float = 0.0
        self.session_started_at: float = time.time()
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.last_seen_at = float(data.get("last_seen_at") or 0.0)
        except Exception:  # noqa: BLE001 —— 坏了就当没记录，不要卡住服务
            self.last_seen_at = 0.0

    def _save(self) -> None:
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"last_seen_at": self.last_seen_at}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass

    def touch(self, now: Optional[float] = None) -> None:
        """收到一次互动就刷新。"""
        self.last_seen_at = time.time() if now is None else now
        self._save()

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        """给外面看的那一份：只给「隔了多久」，不给绝对值。"""
        now = time.time() if now is None else now
        last = self.last_seen_at
        seconds = max(0.0, now - last) if last > 0 else 0.0
        return {
            "has_history": last > 0,
            "seconds_since": round(seconds, 1),
            "minutes_since": round(seconds / 60.0, 2),
            "hours_since": round(seconds / 3600.0, 3),
            "days_since": round(seconds / 86400.0, 3),
        }
