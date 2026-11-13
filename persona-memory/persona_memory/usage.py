# -*- coding: utf-8 -*-
"""记忆使用账本：只有「真的用到」才加强。

学自 returning-tide-memory 的一条精细设计：

    时间会让旧记忆变淡（我们已经有了），
    但「强化」只该来自真实使用——重申、展开、纠正。
    **被列出来看一眼不算。曝光不是使用。**

为什么这件事重要：记忆库越攒越大，如果「用过的」和「从没用过的」
在检索里一样重，那这个库就只是一堆平铺的文本，不会因为被使用而变得更像
「我当然会想起来的事」。

记账口径（刻意从严，宁可少加强）：
  expand    真的展开了这条去读   +1 次使用（强信号）
  reaffirm  又主动记了一遍同一件事 +1 次（中等信号）
  correct   用户纠正了它         +1 次纠正（反向信号）

检索时：final_score × usage_weight
  - 用过 n 次 → 1.0 + min(0.3, 0.08 * n)
  - 被纠正过 → 再乘 0.7（纠正的记忆不该继续往前排）
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

# 每被使用一次加多少权重（封顶 0.3，防止一条记忆霸榜）
USE_BONUS = 0.08
USE_BONUS_CAP = 0.3
# 被纠正过的权重乘数：它不该消失（可纠错 ≠ 删掉），但该靠后
CORRECT_PENALTY = 0.7
# 同一记忆在这个时间窗内反复展开，只算一次使用（默认 10 分钟）
REUSE_WINDOW_SECONDS = 600


class UsageLedger:
    """记忆使用的小账本。条目少、读写频繁，整本读进内存就够。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001 —— 坏了就当没记过，不要卡住检索
            return
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            self.entries = data["entries"]

    def save(self) -> None:
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "entries": self.entries}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass

    # ---------- 记账 ----------
    def _entry(self, memory_id: str) -> Dict[str, Any]:
        e = self.entries.get(memory_id)
        if e is None:
            e = {"uses": 0, "corrections": 0, "last_used_at": 0.0}
            self.entries[memory_id] = e
        return e

    def touch(self, memory_id: str, kind: str = "expand", *, now: Optional[float] = None) -> None:
        """记一次真实使用。

        同一个记忆短时间内被反复展开，只算一次——「翻来覆去看同一页」
        不是「用得多」，是同一个动作重复了。
        """
        if not memory_id:
            return
        now = time.time() if now is None else now
        e = self._entry(memory_id)
        if kind == "correct":
            e["corrections"] = int(e.get("corrections", 0)) + 1
        else:
            last = float(e.get("last_used_at", 0) or 0)
            if now - last < REUSE_WINDOW_SECONDS:
                return
            e["uses"] = int(e.get("uses", 0)) + 1
        e["last_used_at"] = now
        self.save()

    # ---------- 查询 ----------
    def weight(self, memory_id: str) -> float:
        """检索权重乘数。没记录过 = 1.0（不惩罚、不奖励）。"""
        e = self.entries.get(memory_id)
        if not e:
            return 1.0
        uses = int(e.get("uses", 0) or 0)
        corrections = int(e.get("corrections", 0) or 0)
        w = 1.0 + min(USE_BONUS_CAP, USE_BONUS * uses)
        if corrections > 0:
            w *= CORRECT_PENALTY
        return round(w, 4)

    def snapshot(self, *, limit: int = 50) -> Dict[str, Any]:
        rows = sorted(
            self.entries.items(),
            key=lambda kv: (int(kv[1].get("uses", 0) or 0), float(kv[1].get("last_used_at", 0) or 0)),
            reverse=True,
        )
        items = [
            {
                "memory_id": mid,
                "uses": int(e.get("uses", 0) or 0),
                "corrections": int(e.get("corrections", 0) or 0),
                "last_used_at": e.get("last_used_at", 0),
                "weight": self.weight(mid),
            }
            for mid, e in rows[: max(1, limit)]
        ]
        return {"count": len(self.entries), "items": items}
