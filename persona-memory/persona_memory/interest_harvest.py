# -*- coding: utf-8 -*-
"""隐式兴趣收割：从「顺口一提」里抽出**可能是在说兴趣**的话题。

要解决的问题（2026-11-14 评测）：

    481 条真实消息里，「我喜欢X」这类明确句式**命中 0 条**。
    人不会每天宣布自己喜欢什么——他会说「我最近又去滑雪了」「还在玩吉他」。
    `interests.extract_topics()` 只认前一种，`observe_signal()` 那条弱证据
    通道建好了却 18 天 0 次调用。两头都不通，画像就永远是空的。

所以这里补中间那条路：**抽 passive 线索**，不是确认。

设计原则：宁漏不错。
  - 必须有**隐式信号词**（又 / 还在 / 最近 / 开始 / 迷上 / 每天…）——
    没有它，这句话就只是在陈述，不是在说他爱干嘛；
  - 名词必须落在**动作动词**后面（去滑雪 / 玩吉他 / 学摄影）——
    这样「我最近在公司开会」里的「公司」不会被当成兴趣
    （「在公司」是介词短语，不是他在做的事）；
  - 词性只认 jieba 的 n/nr/ns/nz；代词、虚词、单字、纯数字、超过 12 字的一律丢；
  - 归属判别沿用 `interests._ownership_verdict`：别人在玩、假设句，都不算他的。

抽出来的是 `Topic(count=0)` + 一条 `passive` 信号，单条**永远不会**变成
「他喜欢X」——要跨天反复出现，状态机才认（见 interests.interest_verdict）。
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .interests import (
    _MAX_LEN,
    _MIN_LEN,
    _PRONOUN_HEADS,
    _STOPWORDS,
    _ownership_verdict,
)

# 隐式信号词。出现它，这句话才**可能**在说「我最近在干嘛」。
# 刻意只有这些：每一个都是「反复/开始/沉迷」这类**持续性**的线索，
# 「今天」「刚才」这种一次性的不算——一次不做数，这是这本账的前提。
SIGNAL_WORDS: Tuple[str, ...] = (
    "又",
    "还在",
    "最近",
    "开始",
    "迷上",
    "入坑",
    "上头",
    "每天",
    "经常",
    "终于",
    "戒了",
    "停不下来",
    "上瘾",
    "沉迷",
)

# 名词前面那个词落在这里，才认它是「他正在做/玩的东西」。
# 只收动作动词，**不收「在」「和」「跟」**——「我最近在公司开会」的
# 「公司」前面是介词，那是地点不是爱好；「我最近去公司」才勉强算。
ACTIVITY_VERBS = frozenset(
    {
        "玩", "学", "去", "练", "看", "听", "读", "追", "打", "跑", "拍", "做",
        "搞", "研究", "画", "写", "弹", "唱", "游", "骑", "踢", "射", "逛", "刷",
        "录", "剪", "编", "拼", "钓", "爬", "养", "种", "收集", "收藏", "修",
        "改装", "逛", "背", "考", "钻", "折腾", "捣鼓",
        # 信号词里本身就是动作的那些（「我最近迷上露营」）
        "迷上", "沉迷", "入坑", "上头", "上瘾", "种草", "戒", "戒了", "开始",
        "学起", "玩起",
    }
)

# jieba 词性：普通名词 / 人名 / 地名 / 专名。
# 实测「滑雪」被标成 nr、「吉他」标成 ns——jieba 的 n* 细分对「是不是个东西」
# 没有区分度，所以四个全收；动词误标成 n 的（如「玩」）交给单字规则挡掉。
NOUN_FLAGS = ("n", "nr", "ns", "nz")

# 「迷上了X」里的 X 一定是他在玩的东西——哪怕 jieba 把「露营」标成动词（v）。
# 只有这几个**本身就是爱好信号**的动词能开后门；「去」不行——
# 「我最近去开会」的「开会」是差事，不是爱好。
PASSION_SIGNALS = frozenset(
    {"迷上", "沉迷", "入坑", "上头", "上瘾", "种草", "戒", "戒了", "停不下来"}
)

# 这些「名词」不是兴趣：时间 / 泛指 / 身体状态 / 抽象的「事」。
NOUN_STOPWORDS = frozenset(
    {
        "今天", "昨天", "明天", "现在", "最近", "以前", "以后", "之前", "之后",
        "时候", "时间", "日子", "周末", "早上", "晚上", "中午", "下午", "上午",
        "东西", "事情", "问题", "情况", "感觉", "地方", "样子", "关系", "意思",
        "一点", "一下", "一些", "什么", "怎么", "为什么", "大家", "自己",
        "身体", "心情", "状态", "年纪", "岁数", "辈子", "机会", "结果", "原因",
    }
)

# 小句切分：只在**句读**上切，不在「和/还」这类连词上切——
# 「我最近又去滑雪了，还在玩吉他」要切两半；「滑雪和攀岩」得留在一块。
_CLAUSE_SEP = re.compile(r"[，。！？、；：,.!?;:…—~～\n\r\t()（）\[\]【】「」“”‘’\"']+")

# 名词前的助词/副词/标点：跳过它们看「上一个实词」。
# 不跳介词（p）——「在公司」的「在」正是要挡住的那种。
_SKIP_FLAGS = ("u", "y", "e", "o", "x", "w", "d", "m", "q", "f", "c", "r")

_DIGITS = re.compile(r"^[\d.\-+/%:]+$")
_ASCII = re.compile(r"^[A-Za-z0-9_\-]+$")
_HAN = re.compile(r"[\u4e00-\u9fff]")


def _clauses(src: str) -> List[Tuple[int, str]]:
    """切成小句，带每句在原文里的起始下标（归属判别要看原文位置）。"""
    out: List[Tuple[int, str]] = []
    i = 0
    for m in _CLAUSE_SEP.finditer(src):
        if m.start() > i:
            out.append((i, src[i : m.start()]))
        i = m.end()
    if i < len(src):
        out.append((i, src[i:]))
    return out


def _has_signal(clause: str) -> bool:
    return any(w in clause for w in SIGNAL_WORDS)


def _is_topic_word(word: str) -> bool:
    """这个词像不像一个「他喜欢的东西」。"""
    w = str(word or "").strip()
    if not (_MIN_LEN <= len(w) <= _MAX_LEN):
        return False  # 单字（「玩」被误标成 n）和句子碎片
    if w in _STOPWORDS or w in NOUN_STOPWORDS:
        return False
    if w in SIGNAL_WORDS:
        return False  # 信号词自己不是兴趣
    if w[0] in _PRONOUN_HEADS:
        return False
    if _DIGITS.match(w) or _ASCII.match(w):
        return False  # 纯数字 / 纯英文：留给别的路，别混进中文话题
    if not _HAN.search(w):
        return False
    if "的" in w or "了" in w:
        return False
    return True


def _tokenize(clause: str) -> List[Tuple[str, str, int]]:
    """jieba 词性标注，带每个词在**小句内**的起始下标。

    没装 jieba 就返回空列表——调用方退回「一个都不抽」，不硬猜。
    """
    try:
        import jieba.posseg as pseg  # 延迟导入：无 jieba 的环境不该因此崩
    except Exception:  # noqa: BLE001
        return []
    try:
        tokens = list(pseg.cut(clause))
    except Exception:  # noqa: BLE001
        return []
    out: List[Tuple[str, str, int]] = []
    pos = 0
    for t in tokens:
        word = str(getattr(t, "word", "") or "")
        flag = str(getattr(t, "flag", "") or "")
        if not word:
            continue
        out.append((word, flag, pos))
        pos += len(word)
    return out


def extract_implicit_topics(text: str) -> List[str]:
    """从一段话里抽「可能是在说兴趣」的话题（passive 线索，不是确认）。

    只在高置信信号下抽，宁漏不错：

    1. 小句里必须有**隐式信号词**（SIGNAL_WORDS）；
    2. 小句里要有**名词**（jieba n/nr/ns/nz），而且它紧跟在**动作动词**后面；
    3. 过滤代词、虚词、单字、纯数字/纯英文、太长（>12 字）、时间/泛指名词；
    4. 归属不是他本人的（别人在玩 / 假设句）一律丢掉；
    5. 返回话题列表（去重保序）。

    反例（必须抽不出）：
      - 「我最近有点累」← 信号词「最近」但没有名词；
      - 「我今天在公司开会」← 没有信号词；
      - 「我最近在想要不要辞职」← 「辞职」是动词，不是他在玩的东西；
      - 「我朋友最近又去滑雪了」← 归属是他朋友，不是他。
    """
    src = str(text or "").strip()
    if not src:
        return []

    found: List[str] = []
    for _offset, clause in _clauses(src):
        if not _has_signal(clause):
            continue
        prev_word = ""
        for word, flag, start in _tokenize(clause):
            first = flag[:1]
            if first in _SKIP_FLAGS or flag.startswith("u"):
                continue  # 助词/副词/标点：不参与「上一个实词」的判断
            # 「迷上了露营」这类：信号词自己就保证了后面那个是爱好，
            # 所以词性放宽到动词（露营被 jieba 标成 v）。
            noun_flag = flag in NOUN_FLAGS or (
                prev_word in PASSION_SIGNALS and flag in ("v", "vn", "l")
            )
            if noun_flag and _is_topic_word(word):
                # 只有「动作动词 + 名词」才算他在做的事。
                # 这条也是「我最近在公司开会」抽不出「公司」的原因。
                if prev_word in ACTIVITY_VERBS:
                    if _ownership_verdict(clause, start, start + len(word)) == "self":
                        if word not in found:
                            found.append(word)
                # 名词不更新 prev_word：「学摄影和剪辑」两个都该收
                continue
            prev_word = word
    return found
