# -*- coding: utf-8 -*-
"""纠正识别：他这句话，是不是在说「我记错了」。

要解决的问题（2026-11-14 评测）：

    `su_forget` 18 天 0 次调用，`/feedback` 那个钩子是好的，但从来没人走过。
    失败账本建好了，永远 0 条——因为「他发现我记错了」这条路只能靠模型
    自觉去调工具，而模型不会自觉。

所以改成自动认：`/observe` 里扫一眼这句话是不是一次纠正，是就记账。
这本账不是「关于他」的，是「关于我自己哪里错了」——下一轮别再犯同一个错。

三组模式（学 AIDE² 的防作弊设计：先排除，再弱确认，最后才强触发）：

  - **负模式**（优先压制）：他只是在开玩笑 / 打比方 / 自嘲 / 说「我是说…」，
    命中就返回 None。宁可漏记一次，也不要往账本里塞一条假账——
    假账会被注入提示词，让我为一件没发生的事道歉。
  - **弱模式**：「不对」+ 后面跟着「是 / 应该 / 其实」，才可能是纠正。
    单一个「不对」太容易是口头语。
  - **强模式**：直接触发（你记错了 / 我没说过 / 不是这样的 / 你搞错了 …），
    但前面紧挨着「没 / 未 / 别」就不算（「你没记错」不是纠正，是安慰）。

只认「他说的」——调用方负责（role != assistant）。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# 负模式：命中任意一个就闭嘴。这些词一出现，句子多半不是真的在纠正我。
NEGATIVE_MARKERS: Tuple[str, ...] = (
    "开玩笑",
    "逗你",
    "逗我",
    "说笑",
    "随口说说",
    "没关系的",
    "别在意",
    "我是说",
    "假设",
    "如果",
    "假如",
    "要是",
    "万一",
)

# 强模式：(正则, 命中词, 一句人话理由, 要不要判主语)。
# 理由会进翻车账本、再进常驻画像（profile 里只留 12 字），所以写短、写清。
#
# 「X 错」后面不一定是「了」：「你老是看错时间」「你搞错方向了」「说错话」。
# 所以后缀收一小组常见宾语；缺了它，真实语料里那几个最像纠正的句子全会漏。
_CUO = r"错(?:了|吧|啦|咯|话|时间|日期|日子|数|字|人|方向|地方)"

_STRONG: List[Tuple[re.Pattern, str, str, bool]] = [
    # 「你没记错」是安慰，不是纠正——用 lookbehind 挡掉否定的那几种说法
    (re.compile(rf"(?<![没未别])记{_CUO}"), "记错了", "他说我记错了", True),
    # 「我没说过」本身就是「你在瞎记」，不需要判主语
    (re.compile(r"我没(?:有)?(?:说|提)过|我从(?:来)?没(?:有)?(?:说|提)过"), "我没说过", "他说他没说过", False),
    # 「是不是这样的？」是在问我，不是在纠正我——(?<!是) 挡掉疑问句
    (re.compile(r"(?<![是])不是这样的|不是这样吧|不是这个意思"), "不是这样的", "他说不是这样的", False),
    (re.compile(rf"(?<![没未别])(?:搞|弄){_CUO}"), "搞错了", "他说我搞错了", True),
    (re.compile(rf"(?<![没未别])说{_CUO}"), "说错了", "他说我说错了", True),
    # 「你老是看错时间」是真实的抱怨；「我刚才看错了」是他自陈
    (re.compile(rf"(?<![没未别])看{_CUO}"), "看错了", "他说我看错了", True),
    (re.compile(rf"(?<![没未别])算{_CUO}"), "算错了", "他说我算错了", True),
]

# 弱模式：「不对」+ 后接「是 / 应该 / 其实」。中间隔一个逗号也算。
_WEAK = re.compile(r"不对[，,、]?\s*(?:[^，。！？]{0,6})?(?:是|应该|其实)")

# 判主语用的窗口：「我刚才说错了 / 我前面那句记错了」
_SELF_WRONG_WINDOW = 8


def _wrong_is_self(src: str, start: int) -> bool:
    """「X 错了」的主语是不是他自己。

    「不好意思我刚才说错了」→ 他在改自己的口，不是纠正我 → 放过。
    「你刚才说错了」→ 有我 → 记一条。
    「我记错了，其实是周四」→ 自陈记错，也不是我的账 → 放过。
    窗口里既有「我」又有「你」时按「有你」算——「我说你记错了」那种还是冲我来的。
    """
    prefix = src[max(0, start - _SELF_WRONG_WINDOW):start]
    if "你" in prefix:
        return False
    return "我" in prefix


def detect_correction(text: str) -> Optional[Dict[str, str]]:
    """判断这句话是不是「用户在纠正我」。

    返回 `{"reason": str, "strength": "strong"|"weak", "matched": str}` 或 None。
    `reason` 是一句人话（「他说我记错了」），直接进失败账本的 reason 字段。

    判定顺序：负模式 → 强模式 → 弱模式。
    """
    src = str(text or "").strip()
    if not src:
        return None

    # ① 负模式优先：开玩笑 / 打比方 / 我是说… 一律不记账
    if any(m in src for m in NEGATIVE_MARKERS):
        return None

    # ② 强模式
    for pattern, matched, reason, check_subject in _STRONG:
        for m in pattern.finditer(src):
            if check_subject and _wrong_is_self(src, m.start()):
                continue  # 主语是他自己：他在改自己的口，不算纠正我
            return {"reason": reason, "strength": "strong", "matched": matched}

    # ③ 弱模式：只做提示，不进账本（调用方只看 strength == strong）
    if _WEAK.search(src):
        return {"reason": "他说「不对」，可能是在纠正", "strength": "weak", "matched": "不对"}

    return None
