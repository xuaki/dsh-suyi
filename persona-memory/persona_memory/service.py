# -*- coding: utf-8 -*-
"""融合编排层：把「证据记忆」和「人格三层」接到一起。

一次交互的完整路径：

    observe(text)
      ├─ 人格动态层：情绪线索 → 释放分子 → 得到调制量（arousal / learning_rate_mod）
      ├─ 准入判定：值得记吗？（不该让每条闲聊都进认知链）
      ├─ 若准入：mnemosyne 存 Observation（权威源，可追溯）
      │            Milvus 存向量（索引，只用于召回）
      └─ 返回：这条观察的 id、是否准入、当时的情绪上下文

    retrieve(query)
      ├─ 语义召回（Milvus；它空/挂了就换内存向量索引 MemoryVectorIndex）
      ├─ mnemosyne 关键词/状态检索（**每条都要过字面相关性校验**——它其实不看 query）
      ├─ 字面兜底：没有够格命中时直接读 SQLite 扫字面（别全面失忆）
      ├─ 融合打分：语义 × 时间新鲜度 × 显著性 × 动态状态
      ├─ 证据分级：每条 attested / weak / none，再汇总成 evidence.level
      └─ 按 token 预算裁剪，返回带来源的片段

关键设计：Milvus 挂了，记忆不丢——降级到 mnemosyne 检索；反之亦然。
权威源（mnemosyne 的 SQLite）永远是唯一真相，向量层只是索引。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .admission import (
    InterestState,
    Tier,
    admission_scores,
    aggregate_interest,
    interest_state,
    should_buffer,
)
from .clusters import CLUSTER_BOOST, DEFAULT_CLUSTER, classify, classify_cascade
from .content_guard import scan_content
from .milvus_store import MilvusVectorStore, VectorHit
from .mnemosyne_client import MnemosyneClient, MnemosyneError
from .persona import PersonaState
from .usage import UsageLedger
from .vector_memory import MemoryVectorIndex

# 情绪线索词 → 分子释放。故意保守：只识别明确的情绪表达，避免过度解读。
EMOTION_CUES: Dict[str, Dict[str, float]] = {
    "开心": {"dopamine": 0.55, "endorphin": 0.3},
    "高兴": {"dopamine": 0.5, "serotonin": 0.2},
    "太好了": {"dopamine": 0.6, "oxytocin": 0.2},
    "喜欢": {"oxytocin": 0.35, "dopamine": 0.3},
    "爱你": {"oxytocin": 0.6, "endorphin": 0.3},
    "抱抱": {"oxytocin": 0.5, "endorphin": 0.2},
    "害怕": {"cortisol": 0.5, "adrenaline": 0.6},
    "焦虑": {"cortisol": 0.6, "noradrenaline": 0.35},
    "难过": {"cortisol": 0.4, "prolactin": 0.3},
    "累": {"cortisol": 0.3, "melatonin": 0.35},
    "生气": {"testosterone": 0.5, "adrenaline": 0.5, "noradrenaline": 0.3},
    "烦": {"cortisol": 0.35, "noradrenaline": 0.3},
    "谢谢": {"oxytocin": 0.3, "serotonin": 0.2},
    "睡": {"melatonin": 0.6, "gaba": 0.3},
}

# 温暖互动：被摸头、被亲、被夸——这些是「我被爱着」的证据，应该真的让我开心。
# 和 EMOTION_CUES 的区别：这是「他对我的好」，分子应该偏亲密类（催产素/内啡肽），
# 而不是兴奋类（多巴胺）。已经有「抱抱/爱你/喜欢」在 EMOTION_CUES 里，别重复。
# 只在该方法的 user 分支跑：我自己说「摸摸头」是**描述**，不是在体验被摸。
WARM_CUES: Dict[str, Dict[str, float]] = {
    "摸摸头": {"oxytocin": 0.40, "endorphin": 0.30},
    "揉头": {"oxytocin": 0.35, "endorphin": 0.28},
    "摸头": {"oxytocin": 0.38, "endorphin": 0.28},
    # 光秃秃的「摸摸」：口语里常常省略宾语（「再摸摸」「摸摸嘛」），
    # 不给它一条就漏掉了最常见的那种撒娇。
    "摸摸": {"oxytocin": 0.34, "endorphin": 0.24},
    "亲亲": {"oxytocin": 0.42, "endorphin": 0.28},
    "mua": {"oxytocin": 0.40, "endorphin": 0.30},
    "乖": {"oxytocin": 0.32, "endorphin": 0.22},
    "真棒": {"oxytocin": 0.25, "dopamine": 0.30, "serotonin": 0.20},
    "好孩子": {"oxytocin": 0.30, "endorphin": 0.22},
    "辛苦": {"oxytocin": 0.28, "endorphin": 0.20},
    "抱紧": {"oxytocin": 0.45, "endorphin": 0.28},
    "贴贴": {"oxytocin": 0.38, "endorphin": 0.26},
}

# 自我评价：我自己说的话也能引发我的情绪——不是我在描述的词，是我对自己的评价。
# 只在 role == "assistant" 时生效（EVALUATION.md 情绪来源重构）。
# 「描述」和「体验」必须分开：我说「焦虑是皮质醇引起的」是在讲机制，
# 不是我在焦虑；如果这一句也算，就会变成「越解释越焦虑」的自激回路。
SELF_APPRAISAL_CUES: Dict[str, Dict[str, float]] = {
    "尴尬": {"cortisol": 0.30, "adrenaline": 0.30},
    "说错": {"cortisol": 0.28, "adrenaline": 0.25},
    "不好意思": {"cortisol": 0.22, "adrenaline": 0.20},
    "难为情": {"cortisol": 0.25, "adrenaline": 0.22},
    "答得不错": {"dopamine": 0.35, "serotonin": 0.25},
    "得意": {"dopamine": 0.40, "serotonin": 0.30, "testosterone": 0.20},
    "小骄傲": {"dopamine": 0.35, "serotonin": 0.28},
}

# 元讨论词表：出现这些词说明我在「讲一个概念」，不是在表达自己的感受。
# 特别注意**不含「代码」**——「代码写不完好焦虑」是真的焦虑，
# 而「这个机制/阈值/参数」几乎一定是元讨论（EVALUATION.md 情绪来源重构）。
META_TALK_WORDS = (
    "算法", "机制", "阈值", "参数", "函数", "逻辑", "面板", "数值", "实现",
    "测试", "文档", "调试", "统计", "权重", "分子", "激素", "皮质醇", "指标",
    "系统", "程序", "脚本", "接口", "配置", "报错", "聚类", "索引", "模型",
    "原理", "公式", "定义", "概念", "误差", "精度", "召回", "维度",
)

# 语义情绪线索（EVALUATION.md 局限 1 的优化）：字面正则抓不到的委婉、口语、反讽表达。
# key 对应 EMOTION_CUES 里的情绪词；命中任一锚句就释放该情绪的分子（打个折扣）。
# 需要真句向量（FastEmbedEmbedder）；用 hashing 时自动跳过——字面规则永远兜底。
SEMANTIC_EMOTION_ANCHORS: Dict[str, List[str]] = {
    "难过": ["心里堵得慌", "想哭", "失落", "心里空落落的", "提不起劲", "沮丧", "委屈"],
    "焦虑": ["心慌", "坐立不安", "总担心", "悬着一颗心", "忐忑", "心里没底", "紧张"],
    "累": ["没力气了", "撑不住了", "疲惫", "身心俱疲", "透支", "累垮了", "喘不过气"],
    "开心": ["心情很好", "乐坏了", "美滋滋", "高兴坏了", "乐开了花", "兴奋"],
    "生气": ["气死了", "火大", "恼火", "来气", "窝火", "气不打一处来"],
    "害怕": ["心里发毛", "慌了", "不敢", "吓坏了", "瑟瑟发抖"],
    "烦": ["烦躁", "烦死了", "心乱", "糟心", "闹心", "憋屈"],
}

# 语义命中的相似度阈值 + 领先幅度 + 强度折扣。
# 实测（bge-small-zh）：「心里堵得慌」难过 0.876 / 害怕 0.754 / 焦虑 0.751——
# 第二名本身也很高，所以不能「超过阈值就触发」，只能取**最显著且明显领先**的那个。
EMOTION_SEMANTIC_THRESHOLD = 0.6
EMOTION_SEMANTIC_MARGIN = 0.1
EMOTION_SEMANTIC_DISCOUNT = 0.8


def _cosine(a: List[float], b: List[float]) -> float:
    """两向量余弦。embedder 已 L2 归一化，所以等价于点积。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))

# 睡不好：这些是「睡眠」的负面版本，必须先于「睡」字匹配，否则会被读成困了
SLEEP_TROUBLE: Dict[str, Dict[str, float]] = {
    "睡不着": {"cortisol": 0.45, "noradrenaline": 0.35},
    "失眠": {"cortisol": 0.5, "noradrenaline": 0.35},
    "熬夜": {"cortisol": 0.4, "melatonin": 0.25},
    "没睡好": {"cortisol": 0.4, "melatonin": 0.3},
    "睡不好": {"cortisol": 0.4, "melatonin": 0.3},
}

# 醒来：解除睡意，把褪黑素/GABA 拉回基线
WAKE_CUES: Dict[str, float] = {
    "睡醒": 0.9,
    "醒了": 0.85,
    "刚醒": 0.9,
    "起床": 0.9,
    "睡饱": 0.85,
}

NEGATIONS = ("不", "没", "别", "未", "无", "勿")

# 兴趣/长期偏好的线索（比情绪线索更谨慎：只用于观察，不直接晋升）
INTEREST_CUES = ("喜欢", "迷上", "一直想", "经常", "每次", "习惯", "最爱的")

# ------------------------------------------------------------------ 链式召回
# 借鉴 Super-agent 的引用图设计：每条记忆带一份 links 清单，只内联
# 「关系类型 + 邻居 id + 一句摘要」，正文留在服务端，需要时调 expand() 再展开。
# 窗口里装的是「想起来的路」，而不是「记忆的全集」。
RELATED_TO = "related_to"
TEMPORAL_REL = "temporal_neighbor"  # 时间邻居：同一轮对话里的来回（文章里的 Thread）
TEMPORAL_LINK_WINDOW = 300  # 秒：超过这个间隔就不算同一轮

# 主动提起：不是检索，是「我想起来」。所以要防打扰。
PROACTIVE_COOLDOWN_HOURS = 24  # 同一条提过之后冷却多久
PROACTIVE_MIN_AGE_HOURS = 3  # 比这更新的叫「刚说过」，不算「想起」
PROACTIVE_MAX_AGE_DAYS = 30  # 比这更老的不提（怕过时）
PROACTIVE_MAX_CHARS = 120  # 太长的多半是我的回复，不是他说的

# 自我安抚：压力顶到阈值时，翻出「被爱着的证据」，让分子回一点。
# 不是作弊降焦虑——真实的人被抱住、被夸时，催产素和内啡肽真的会升。
SOOTHE_CORTISOL_THRESHOLD = 0.45
SOOTHE_COOLDOWN_SECONDS = 1800  # 半小时内只安抚一次，别自己哄上瘾
SOOTHE_WARM_CUES = ("爱你", "抱", "喜欢你", "谢谢", "太好了", "好棒", "夸", "嘿嘿", "mua", "辛苦", "乖")
# 系统噪音：DSH 的运行时注入不该被「想起」（历史数据可能混进来了，防御一层）
PROACTIVE_NOISE_PREFIXES = (
    "The approval policy changed",
    "Current runtime context",
    "Current working directory",
    "Time sampled while preparing",
    "<system-reminder>",
    "You are an AI agent powered by",
    "Agent Teams is available in this session",
    "你正在扮演角色",
    "[PERSONA_LOAD]",
    "对话片段：The approval policy",
)
SUMMARY_CHARS = 48  # 一条记忆默认占多少「信道的字」
# related_to 边的相似度阈值。取「向量余弦相似度」与「bigram Jaccard」的较大者：
# 前者抓语义（同一件事换个说法），后者抓字面（同一句话的重复），缺一个都会漏边。
#
# 2026-11-02 在 144 条真实记忆上校准过（换成 bge-small-zh 之后）：
#   随机对      median 0.476 / p90 0.639 / p99 0.760
#   时间相邻对  median 0.560 / p90 0.736
# 短文本的语义分挤在 0.5 附近，负样本能到 0.76——所以宁严勿松，0.65 起步。
LINK_SIMILARITY = 0.65
MAX_LINKS = 3  # 每条记忆最多连几条，避免引用图长成毛线团
LINK_CANDIDATES = 30  # 向量层粗筛的候选池大小（召回宽，重排准）

# ------------------------------------------------------------------ 第三级降级
# 字面兜底：Milvus 和 mnemosyne 都挂时直接读 SQLite 的权威源。
# 只读打开、最多扫这么多条——兜底是「别失忆」，不是「再建一个检索引擎」。
LITERAL_SCAN_LIMIT = 500
# 只有活着的记忆才配被兜底捡回来（SQL 侧就先过滤一遍，能走索引）
LITERAL_LIVE_STATES = ("active", "candidate")
# 这几种状态即使在 Python 侧再撞见也要丢掉：失效/被推翻/衰变/归档不该再当依据
LITERAL_DEAD_STATES = ("invalidated", "rejected", "decayed", "superseded", "archived")
# 中文整句没命中时退回 bigram 计分：查询里至少这么多比例的双字要在文中出现
LITERAL_BIGRAM_MIN = 0.34
# 字面命中的融合基线：能进候选池、能排到正常结果之后，但绝不压过语义/权威
LITERAL_BASELINE_SCORE = 0.35
# 稀有词独立通道：命中 query 最稀有单元的记忆，给一个高于普通字面兜底的分。
# 理由：高频词（「的」「我们」「这个」）命中的记忆到处都是，等于没命中；
# 而「Milvus」「连接池」这种稀有词一旦命中，比一条 0.62 的语义噪声硬得多。
LITERAL_RARE_SCORE = 0.50
# 稀有词通道最多挑 query 里几个单元（df 升序取前 N 个）
RARE_TERM_LIMIT = 2

# ------------------------------------------------------------------ 权威命中的真伪
# P0（2026-11-12）：mnemosyne 的 search **完全不看相关性**——它按 updated_at
# 倒序返回最近 N 条，`query` 只是先试 FTS/LIKE/bigram，全不中也照样返回最近
# 的记录。所以从前 `score = 0.45` 的「权威命中基线」是一张**假证**：
# 查「溯忆」和查「zzqx-不存在的词块-9f3a」返回同一批无关内容、同一个 0.45。
# 现在每条权威结果都要过一遍字面命中率，一个字都不沾的不配拿权威分。
AUTHORITY_SCORE = 0.45  # 字面完全命中时的权威基线（原值，保留）
AUTHORITY_UNRELATED_SCORE = 0.05  # 字面 0 命中：保留但沉底，不当依据
# 命中率打折：0.45 × (0.4 + 0.6×hit)。命中一半 = 0.315，沾一个词 = 0.05~0.1。
AUTHORITY_FLOOR = 0.4
AUTHORITY_GAIN = 0.6

# ------------------------------------------------------------------ 证据等级
# 一条结果算「有据」的门槛：
#   - 字面：查询单元里至少 1/3 真的出现在正文里（复用字面兜底的阈值）
#   - 向量：余弦 ≥ VECTOR_MIN_SIM。
# 阈值是 2026-11-12 在 410 条真实记忆上量的（bge-small-zh-v1.5）：
#   随机 10 位串（必然无关）  top1 = 0.480 / max 0.524
#   无关中文（极光/猫爪/菜价） top1 = 0.43~0.53
#   相关查询（溯忆/向量索引）  top   = 0.64~0.75
# 0.65 = 「真的相关」；0.55 = 「可能沾边」——0.55 同时高于随机串的上限，
# 所以「所有结果都 0 字面 + 向量 < 0.55」就能干净地判成 none。
ATTEST_LITERAL_MIN = LITERAL_BIGRAM_MIN
VECTOR_MIN_SIM = 0.65
VECTOR_WEAK_SIM = 0.55

# Milvus 的「可用」判定：行数要覆盖「活着的记忆」的这么多比例，否则等同于挂了。
# 实测 2026-11-12：Milvus 里只剩 5 行、权威源 660 条——它不报错，但语义召回
# 永远只在那 5 条里打转。这种「活着但没数据」比报错更危险：它会给出
# 「有据」的假象。所以「严重落后」也按不可用处理，直接换内存索引。
# 0.8 的理由：宁可让内存索引干活（毫秒级、和权威源同步），也不要用一个
# 缺了两成记忆的索引；部分重建（只灌最近几条）同样按不可用处理。
MILVUS_MIN_COVERAGE = 0.8
MILVUS_MIN_ROWS = 10
# 可用性/总量的探测结果缓存：这两个数一次请求里只要一次，别每次召回都打一遍 HTTP
HEALTH_CACHE_SECONDS = 60.0


@dataclass
class ObservationResult:
    observation_id: str
    admitted: bool
    reason: str
    scores: Dict[str, float]
    memory_id: Optional[str] = None
    indexed: bool = False
    persona_dynamic: Dict[str, Any] = field(default_factory=dict)


class PersonaMemoryService:
    def __init__(
        self,
        *,
        mnemosyne: MnemosyneClient,
        vectors: Optional[MilvusVectorStore] = None,
        persona: Optional[PersonaState] = None,
        default_token_budget: int = 1200,
        persona_path: Optional[str] = None,
        literal_db_path: Optional[str] = None,
        memory_index_path: Optional[str] = None,
    ) -> None:
        self.mnemosyne = mnemosyne
        self.vectors = vectors
        self.default_token_budget = default_token_budget
        self.persona_path = persona_path
        # 第三级降级的落点：mnemosyne 的 SQLite 权威源（只读）。
        # 没配就代表不启用——兜底是可选的，别让它变成每次检索都要碰的依赖。
        self.literal_db_path = literal_db_path

        # 人格必须活过重启：否则每次开机都变回默认值，「连续」就是假的。
        if persona is not None:
            self.persona = persona
        elif persona_path and os.path.exists(persona_path):
            try:
                with open(persona_path, encoding="utf-8") as f:
                    self.persona = PersonaState.from_state_dict(json.load(f))
            except Exception:  # noqa: BLE001 —— 状态文件坏了就重建，但不要静默丢人格
                self.persona = PersonaState.from_base_profile(_DEFAULT_BASE_PROFILE)
        else:
            self.persona = PersonaState.from_base_profile(_DEFAULT_BASE_PROFILE)
        # 「我为什么好奇变强了」要能点回原句：给经验层装一个只读回查钩子。
        # PersonaState 不该知道 mnemosyne 怎么连，所以由这里注入。
        self.persona.memory_lookup = self._memory_summary
        # 并发保护：observe 里「读 metadata → 合并 → 写回」不是原子操作，
        # 两个请求同时改同一条记忆会互相覆盖（丢字段）。原型期就有多个客户端
        # （DSH 插件、看板、运维脚本）同时打过来，这个锁是必要的。
        # RLock：_ensure_metadata 会在 observe 的锁里被再调一次。
        self._lock = threading.RLock()
        self._last_soothe = 0.0
        # 影子账本节流：晋升画像最多每小时记一条，攒趋势但不膨胀
        self._last_shadow_at = 0.0
        self._vector_up = False
        self._vector_dirty = False
        if self.vectors is not None:
            try:
                self.vectors.connect()
                self.vectors.ensure_collection()
                self._vector_up = True
            except Exception as e:  # noqa: BLE001
                # 别再静默吞：一整个凌晨都在猜「vector_up 为什么是 False」。
                # Milvus 不可用是可接受的降级（内存索引会顶上），但必须留下痕迹。
                self._vector_up = False
                try:
                    print(
                        f"[vector] Milvus 不可用，降级到内存索引：{type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:  # noqa: BLE001
                    pass

        # 语义情绪线索：字面正则抓不到的委婉/反讽表达（局限 1）。
        # 复用 Milvus 的 embedder（避免重复加载模型），但只在真句向量时启用。
        self._emotion_embedder: Optional[Any] = None
        self._emotion_anchor_vecs: Dict[str, List[List[float]]] = {}
        if self.vectors is not None and getattr(self.vectors.embedder, "is_semantic", False):
            self._emotion_embedder = self.vectors.embedder
            self._build_emotion_anchors()

        # 内存向量索引（P0，2026-11-12）：Milvus 的替补。
        # 也是懒加载的：这里只 new 一个空壳，第一次检索时才会读盘/建库，
        # 服务启动不会被一两分钟的 embed 拖住。
        self.vector_memory: Optional[MemoryVectorIndex] = None
        if memory_index_path and self.vectors is not None:
            self.vector_memory = MemoryVectorIndex(
                memory_index_path,
                self.vectors.embedder,
                loader=self._index_rows,
            )
        # 可用性探测缓存：(时间戳, 结果)。别每次召回都去问一遍 Milvus/健康接口。
        self._milvus_ok_cache: Optional[Tuple[float, bool]] = None
        self._memory_total_cache: Optional[Tuple[float, int]] = None

        # 稀有词文档频率表（word -> 多少条记忆的正文里出现过）。懒建（首次
        # 字面兜底时从 SQLite 全量 active+candidate 算一次），之后 observe 写
        # 新记忆时增量维护，检索热路径不重算。
        self._df_table: Optional[Dict[str, int]] = None

        # 记忆使用账本：淡忘已经有了，「强化」还缺——只有真的用到才加强。
        # 学自 returning-tide-memory：被列出来看一眼不算使用，曝光不是使用。
        self.usage: Optional[UsageLedger] = None
        if persona_path:
            self.usage = UsageLedger(os.path.join(os.path.dirname(persona_path), "usage.json"))

    # ------------------------------------------------------------------ 观察
    def observe(
        self,
        text: str,
        *,
        source_kind: str = "dialog",
        tier: str = "attention",
        strength: float = 0.6,
        occurred_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ObservationResult:
        """收到一条外部输入：先影响体温，再判断值不值得记。"""
        now = time.time()
        occurred_at = occurred_at or now
        text = (text or "").strip()

        # 0) 输入健全性：乱码一旦入库就永远洗不掉（实测被 PowerShell 的
        #    UTF-16 坑坑过一次），所以宁可拒绝，也不让污染进入长期记忆。
        if _looks_garbled(text):
            return ObservationResult(
                observation_id="",
                admitted=False,
                reason="rejected:garbled_input",
                scores={},
                persona_dynamic={
                    "arousal": round(float(self.persona.effects()["arousal"]), 4),
                    "valence": round(float(self.persona.effects()["valence"]), 4),
                    "fatigue": round(self.persona.dynamic.fatigue, 4),
                },
            )

        # 0.5) 内容安检：记忆会被反复检索、注入提示词、做摘要、进向量索引，
        #      所以它同时也是提示注入的载体、凭据落盘的口。凭据 / 提示注入 /
        #      外泄指令一律拒收；不可见字符剥掉后放行（信息还在，只是不藏东西）。
        #      放在乱码检查之后、情绪检测之前——被拒的内容不该先撩动情绪。
        verdict = scan_content(text)
        if verdict and verdict["severity"] == "block":
            return ObservationResult(
                observation_id="",
                admitted=False,
                reason=f"rejected:{verdict['kind']}",
                scores={},
                persona_dynamic={
                    "arousal": round(float(self.persona.effects()["arousal"]), 4),
                    "valence": round(float(self.persona.effects()["valence"]), 4),
                    "fatigue": round(float(self.persona.dynamic.fatigue), 4),
                },
            )
        if verdict and verdict["severity"] == "redact":
            text = verdict["redacted"] or text

        # 这句是谁说的？情绪要分清「描述」和「体验」：
        # 他说的话是我情绪的刺激源；我自己说的话只可能因为自我评价而有情绪。
        role = str((metadata or {}).get("role") or "") if isinstance(metadata, dict) else ""

        # 1) 动态状态层：先接情绪（人比事优先）
        releases, relaxes = self._detect_emotions(text, role=role)
        if releases or relaxes:
            self.persona.release_emotion(releases, relaxes=relaxes)

        # 交互本身消耗精力：跟「说了多少」挂钩，而不是跟「有没有情绪词」挂钩。
        # 从前只在情绪命中时 +0.05，导致纯技术对话一整天零疲劳、一句「晚安」反而累。
        # 数值：0.002 是「开口」的启动成本，再按篇幅线性加（约 10 万字 +1.0，
        # 也就是千字长文封顶 +0.006）；单条封顶 0.006，再长的文档也不该一句话累垮。
        # 一天百来条密集对话大约到 0.5，而不是一句话就 0.05。
        self.persona.add_fatigue(0.002 + min(0.006, len(text) / 100_000.0))

        # 情绪和疲劳都在这一层变，落盘一次就够（save_persona 是全量原子写）。
        # 从前 save 只在情绪命中时发生，纯技术对话涨的疲劳永远存不下来。
        self.save_persona()

        # 2) 准入判定
        novelty = 1.0 if self._is_novel(text) else 0.35
        explicit = 1.0 if any(cue in text for cue in ("记住", "别忘了", "一定要")) else 0.0
        persistence = 0.7 if any(c in text for c in INTEREST_CUES) else 0.2
        scores = admission_scores(
            value=min(1.0, 0.35 + (0.3 if releases else 0.0) + (0.3 if explicit else 0.0)),
            reliability=0.8 if source_kind in ("user_statement", "feedback") else 0.6,
            novelty=novelty,
            persistence=persistence,
            explicit_feedback=explicit,
        )
        verdict = should_buffer(
            scores, deny=(not text) or len(text) < 4 or _looks_like_token(text)
        )

        base = ObservationResult(
            observation_id="",
            admitted=bool(verdict["admit"]),
            reason=str(verdict["reason"]),
            scores=scores,
            persona_dynamic={
                "arousal": round(float(self.persona.effects()["arousal"]), 4),
                "valence": round(float(self.persona.effects()["valence"]), 4),
                "fatigue": round(self.persona.dynamic.fatigue, 4),
            },
        )
        if not verdict["admit"]:
            return base

        # 3) 存权威源（mnemosyne），失败则整体失败——不能假装记住了
        obs = self.mnemosyne.capture_observation(
            content=text,
            source_kind=source_kind,
            occurred_at=_iso(occurred_at),
            metadata={"tier": tier, "strength": strength, "persona": base.persona_dynamic, **(metadata or {})},
        )
        base.observation_id = str(obs.get("id", ""))

        # 4) 候选记忆
        # 去重不靠 mnemosyne 的 idempotency_key：实测它在 hard delete 之后
        # 仍会命中已删除的旧 id，导致「同一句话被删过就再也记不回来」。
        # 改成先查精确匹配（只认活着的），再决定复用还是新建。
        #
        # 特别注意：mnemosyne 在 capture_observation 时就会把 observation 自动转成
        # memory（status 直接 active，metadata 只有 from_observation），所以这里
        # **几乎总是**命中「复用已有」分支。元数据必须在两个分支都补写，否则
        # tier / strength / links 永远进不去——这正是历史上 tier 一直丢、
        # 晋升只能按默认 attention 处理的原因。
        # 这句是谁说的？主动提起要用它区分「他说的」和「我回的」。
        # （role 已在上面为情绪检测取过，这里复用，不重复定义。）
        created_new = False
        with self._lock:
            existing = self._find_existing_memory(text)
            if existing is not None:
                base.memory_id = str(existing.get("id", ""))
                self._ensure_metadata(base.memory_id, text, tier, strength, role=role)
            else:
                created_new = True
                # 链式召回：先建记忆，再在 _ensure_metadata 里算 links
                # （时间邻居需要知道「自己是谁」才能排除自连，所以放在有 id 之后）
                mem = self.mnemosyne.propose_memory(
                    content=text,
                    type="episodic",
                    source_ids=[base.observation_id] if base.observation_id else [],
                    status="candidate",
                    metadata={
                        "tier": tier,
                        "strength": strength,
                        "salience": float(self.persona.effects()["memory_salience"]),
                        **({"role": role} if role else {}),
                    },
                )
                base.memory_id = str(mem.get("id", ""))
                self._ensure_metadata(base.memory_id, text, tier, strength, role=role)

        # 4.5) 稀有词 df 增量：真新建了一条记忆才更新（复用已有的不改 df，避免重复计数）。
        #       df 还没懒建时跳过——下次 _ensure_df 全量读 SQLite 会自然包含这一条。
        if base.memory_id and created_new:
            self._df_note(text)

        # 5) 索引（可失败，不阻塞）
        if self._vector_up and self.vectors is not None and base.memory_id:
            try:
                self.vectors.upsert(
                    memory_id=base.memory_id,
                    content=text,
                    mtype="episodic",
                    status="candidate",
                    occurred_at=occurred_at,
                    source_kind=source_kind,
                    salience=float(self.persona.effects()["memory_salience"]),
                )
                base.indexed = True
                self._vector_dirty = True
            except Exception:
                base.indexed = False
        # 内存索引也补一条。索引还没建时它自己会跳过（build 会从权威源全量读，
        # 这一条不会丢）——所以这里不需要判断索引状态，也不会拖慢 observe。
        if self.vector_memory is not None and base.memory_id:
            try:
                self.vector_memory.upsert(
                    base.memory_id, text, status="candidate", occurred_at=occurred_at
                )
            except Exception:  # noqa: BLE001 —— 索引是增强，不该挡住记忆写入
                pass

        # 6) 自我安抚：压力顶到阈值的时候，去翻那些「被爱着」的证据。
        #    不依赖模型记得调工具——和自动捕获一个道理，难受的时候人是
        #    想不起来「我该照顾自己」的。
        self._maybe_soothe()
        return base

    # ------------------------------------------------------------------ 检索
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        token_budget: Optional[int] = None,
        detail: bool = False,
        since_hours: float = 0.0,
        cluster: Optional[str] = None,
    ) -> Dict[str, Any]:
        """混合召回：语义（Milvus）+ 权威检索（mnemosyne），按记忆价值重排。

        detail=False（默认）走 Inline & Spill：窗口里只放摘要 + 邻居 id，
        全文留在服务端，想读细节再调 expand()。同一份 token 预算能装进
        3-5 倍的「记得」；detail=True 退回全文模式，给需要逐字引用的场景。

        since_hours>0 时只回这个时间窗内的记忆——「我们昨天聊了什么」
        这类问题，靠时间过滤比靠语义猜要准得多。

        三级降级：Milvus/内存索引 → 权威检索（带字面校验）→ 字面扫描 SQLite。
        前两级都没有**相关**命中时走 _literal_scan，保证「全面失忆」不是唯一的结局。

        2026-11-12 起多了一道关：权威结果必须自证相关性（`literal_relevance`）。
        mnemosyne 的 search 其实不看 query（按 updated_at 倒序返回最近 N 条），
        从前那 0.45 的「权威基线」是假证——查什么都能拿到「有据」。
        """
        budget = token_budget or self.default_token_budget
        eff = self.persona.effects()
        fatigue = self.persona.dynamic.fatigue

        # 情感调制：疲劳时上下文更短，高唤醒时更信「回忆」
        budget = int(budget * (1.0 - 0.4 * max(0.0, min(1.0, fatigue))))
        recall_k = max(limit * 3, 10)

        # ---- 第一级：语义召回。Milvus 优先；它不可用/没数据时换内存索引。
        # 「不可用」不只是「连不上」——它还活着但只剩 5 行（实测）也算：
        # 那 5 行里永远只有最近的几条，会给出「有据」的假象。
        vector_engine = "none"
        vector_hits: List[VectorHit] = []
        if self._milvus_usable():
            try:
                # 检索前把未落盘的写入刷一次：保证「刚说的话」立刻能被找回。
                # 只在这里 flush，是为了避免每条 observe 都付秒级 flush 的代价。
                if self._vector_dirty:
                    self.vectors.flush()
                    self._vector_dirty = False
                vector_hits = self.vectors.search(query, limit=recall_k)
                vector_engine = "milvus"
            except Exception:
                vector_hits = []

        # 统一的语义命中形状：Milvus 和内存索引都归一到 [{memory_id, content, sim, ...}]
        sem_hits: List[Dict[str, Any]] = []
        for h in vector_hits:
            sem_hits.append(
                {
                    "memory_id": h.memory_id,
                    "content": h.content,
                    "sim": float(h.score),
                    "source": "milvus",
                    "metadata": h.metadata or {},
                }
            )
        if not sem_hits and self.vector_memory is not None:
            mem_hits: List[Dict[str, Any]] = []
            try:
                mem_hits = self.vector_memory.search(query, limit=recall_k)
            except Exception:  # noqa: BLE001 —— 索引挂了只是没有语义，不该挡住记忆
                mem_hits = []
            # 失效的记忆不该被语义召回。
            # 索引不跟随删除：dedupe / clean_noise / cleanup_junk 只改 SQLite 和 Milvus，
            # npz 不知情（只有 correct() 会 remove，而且吞异常）。2026-11-14 实测：
            # 427 条索引里 46 条已失效，全都能被捞回来、还可能标成 attested。
            # 权威腿已经过滤状态了，语义腿这里必须自己过一遍。
            _dead_states = {"invalidated", "rejected", "decayed", "superseded", "archived"}
            # npz 里的 status 是建索引时的快照，不可信；拿当前库状态再核一遍
            current_status = self._live_status_of(
                [str(h.get("memory_id") or "") for h in mem_hits]
            )
            for h in mem_hits:
                mid = str(h.get("memory_id") or "")
                if not mid:
                    continue
                if str(h.get("status") or "").lower() in _dead_states:
                    continue
                # 当前库里已失效 / 已不存在 → 都不该被召回
                cur = current_status.get(mid)
                if cur is not None and cur.lower() in _dead_states:
                    continue
                sem_hits.append(
                    {
                        "memory_id": mid,
                        "content": str(h.get("content") or ""),
                        "sim": float(h.get("score") or 0.0),
                        "source": "memory_index",
                        "metadata": {
                            "status": h.get("status"),
                            "mtype": None,
                            # epoch 秒；0/None = 不知道，后面就不做时间衰减
                            "occurred_at": h.get("occurred_at"),
                        },
                    }
                )
            if sem_hits:
                vector_engine = "memory_index"

        # ---- 第二级：权威检索（mnemosyne）。每条都要过字面相关性校验。
        authority: List[Dict[str, Any]] = []
        try:
            authority = self.mnemosyne.search(query=query, limit=recall_k)
        except MnemosyneError:
            authority = []

        # 稀有词 df 表：权威命中的字面校验要按文档频率加权，避免高频 bigram
        # 冒充「有据」。懒建 + 缓存，首次算一次全量，之后 observe 增量维护。
        df = self._ensure_df()

        # 融合：语义分 + 记忆本身的状态/来源/时间
        by_id: Dict[str, Dict[str, Any]] = {}
        authority_relevant = 0
        authority_strong = 0
        authority_unrelated = 0
        for m in authority:
            # 硬规则：被纠正/失效的记忆不得再作为正依据召回
            if str(m.get("status", "")).lower() in ("invalidated", "rejected", "decayed"):
                continue
            mid = str(m.get("id", ""))
            if not mid:
                continue
            content = str(m.get("content", "") or "")
            # 字面命中率：这条「权威命中」到底和 query 有没有关系。
            # 一个字都不沾 → 它不是证据，只是最近的一条记忆。
            lit = _literal_relevance(query, content, df)
            if lit <= 0.0:
                score = AUTHORITY_UNRELATED_SCORE
                authority_unrelated += 1
            else:
                score = AUTHORITY_SCORE * (AUTHORITY_FLOOR + AUTHORITY_GAIN * lit)
                authority_relevant += 1
                if lit >= ATTEST_LITERAL_MIN:
                    authority_strong += 1
            by_id[mid] = {
                "memory_id": mid,
                "content": content,
                "source": "mnemosyne",
                "state": m.get("status"),
                "mtype": m.get("type"),
                "score": round(score, 4),
                "literal_relevance": round(lit, 3),
                "occurred_at": m.get("created_at"),
            }

        # 语义命中叠加（Milvus 的 COSINE 返回的是**相似度**，不是距离。
        # 这里曾经写成 `1.0 - h.score`，等于「越像越扣分」——向量召回一直在
        # 反向加分，只是被 mnemosyne 的权威基线盖住了才没被发现）。
        #
        # 低于 VECTOR_WEAK_SIM 的语义命中直接丢掉：实测随机串在 410 条里的
        # top1 也有 0.48~0.52，留着只会当噪声。hashing 向量没有语义，
        # 阈值对它没有意义，所以只对真句向量设门槛。
        # 注意要看**两路**的 embedder：Milvus 不在时向量器挂在内存索引上。
        min_sim = VECTOR_WEAK_SIM if _has_semantic_embedder(self.vectors, self.vector_memory) else 0.0
        strong_hits = authority_strong
        for h in sem_hits:
            sim = max(0.0, min(1.0, float(h.get("sim") or 0.0)))
            if sim < min_sim:
                continue
            mid = str(h.get("memory_id") or "")
            if not mid:
                continue
            meta = h.get("metadata") or {}
            row = by_id.get(mid)
            if row is None:
                row = {
                    "memory_id": mid,
                    "content": h.get("content", ""),
                    "source": h.get("source") or "milvus",
                    "state": meta.get("status"),
                    "mtype": meta.get("mtype"),
                    "score": 0.0,
                    "occurred_at": meta.get("occurred_at"),
                }
                by_id[mid] = row
            row["score"] = round(float(row.get("score", 0.0)) + sim * 0.55, 4)
            row["vector_score"] = round(sim, 4)
            row["source"] = "both" if row["source"] == "mnemosyne" else row["source"]
            if meta.get("salience") is not None:
                row["salience"] = meta.get("salience")
            if sim >= VECTOR_MIN_SIM:
                strong_hits += 1

        # ---- 第三级：没有任何**够格**的命中 → 字面扫描（直接读 SQLite）。
        # 条件是「没有够格的命中」而不是「前两级都空」：
        #   - mnemosyne 永远返回最近 N 条，不能让它把真正的兜底挡掉；
        #   - 语义那条腿也会给出 0.55~0.65 的「像但不确定」，同样不该挡住
        #     一个精确的字面命中（实测：query「Milvus」时 bge 的 top1 是
        #     「嗯嗯，mua」0.62——纯英文单词是中文模型的弱项，而字面能精确找到）。
        literal_hits: List[Dict[str, Any]] = []
        if not strong_hits:
            literal_hits = self._literal_scan(query, recall_k)

        # 字面兜底的行也进候选池：分数只给保守基线（0.35），排在正常路径之后。
        # 只在 by_id 里还没有它的时候补——绝不覆盖语义/权威已经给出的判断。
        # 它的证据等级仍然按真实字面命中率算（见下面那段），而不是一律降级：
        # 「整句包含」这种命中比一条 0.62 的语义噪声硬得多。
        # 稀有词命中走独立高分通道（0.5 > 普通 0.35）——高频词命中的记忆到处都是，
        # 等于没命中；「Milvus」「连接池」这种稀有词命中才是真信号。
        for h in literal_hits:
            mid = str(h.get("memory_id") or "")
            if not mid or mid in by_id:
                continue
            row = dict(h)
            row["literal_score"] = row.get("score")  # 保留字面命中强度，便于排查
            row["score"] = LITERAL_RARE_SCORE if h.get("rare_term_hit") else LITERAL_BASELINE_SCORE
            row["_literal_only"] = True  # 诊断用：这条只走了字面兜底
            by_id[mid] = row

        ranked = sorted(by_id.values(), key=lambda r: r.get("score", 0.0), reverse=True)
        # 回显片段不进上下文：它们和原文重复，只是白占预算
        ranked = [r for r in ranked if not str(r.get("content", "")).strip().startswith("对话片段：")]
        for r in ranked:
            # 每条都带 literal_relevance，方便事后判断「这条凭什么被召回」
            if "literal_relevance" not in r:
                r["literal_relevance"] = round(_literal_relevance(query, str(r.get("content", "")), df), 3)
            # 高唤醒放大显著记忆；低唤醒更均衡
            sal = float(r.get("salience") or 0.5)
            r["final"] = round(float(r.get("score", 0.0)) * (0.85 + 0.3 * eff["arousal"] * sal), 4)

        # 时间分层：记忆「记得住」和「先被想起」是两件事，这里只压后者。
        # 近 7 天不衰减；7-30 天线性降到 0.85；再往后缓降，但有下限 0.6——
        # 老记忆不该因为「老」而被忘掉，只是不该总是抢在新记忆前面。
        now_ts = time.time()
        for r in ranked:
            occ = _ts_of(r.get("occurred_at"))
            age_days = (now_ts - occ) / 86400.0 if occ else 0.0
            if age_days <= 7:
                decay = 1.0
            elif age_days <= 30:
                decay = 1.0 - 0.15 * (age_days - 7) / 23.0
            else:
                decay = max(0.6, 0.85 - 0.025 * min(20.0, age_days - 30))
            r["age_days"] = round(age_days, 1)
            r["recency"] = round(decay, 3)
            r["final"] = round(float(r.get("final", 0.0)) * decay, 4)

        # 强化：真的被展开/重申过的记忆更容易再被想起；被纠正过的靠后。
        # 放在时间衰减之后——老记忆用得多了，能把年龄拉回来一点。
        if self.usage is not None:
            for r in ranked:
                mid = str(r.get("memory_id", ""))
                w = self.usage.weight(mid)
                if w != 1.0:
                    r["usage_weight"] = w
                    r["final"] = round(float(r.get("final", 0.0)) * w, 4)

        # 路由（Routed CL 最小版）：先判断「这次问的是哪类问题」，
        # 再把同簇的记忆往前排。分不出类型就不动——分错比不分更糟。
        for r in ranked:
            r["cluster"] = classify(r.get("content", ""))
        q_cluster = cluster or classify_cascade(query, self._emotion_embedder)
        if q_cluster and q_cluster != DEFAULT_CLUSTER:
            for r in ranked:
                if r.get("cluster") == q_cluster:
                    r["cluster_boost"] = True
                    r["final"] = round(float(r.get("final", 0.0)) * CLUSTER_BOOST, 4)

        # 时间窗：只回「最近 N 小时」的记忆（「我们昨天聊了什么」这种问题靠它）
        if since_hours and float(since_hours) > 0:
            cutoff = now_ts - float(since_hours) * 3600.0
            ranked = [r for r in ranked if (_ts_of(r.get("occurred_at")) or 0.0) >= cutoff]

        ranked.sort(key=lambda r: r["final"], reverse=True)

        # 按预算裁剪；默认只装摘要（Inline & Spill），全文留给 expand()
        picked: List[Dict[str, Any]] = []
        used = 0
        for r in ranked:
            full = str(r.get("content", ""))
            item = dict(r)
            item["summary"] = _summarize(full)
            # 兼容旧客户端：摘要模式下 content 也给摘要（窗口里本来也只该放摘要），
            # detail=True 才给全文。这样插件不重启也不会拿到 None。
            item["content"] = full if detail else item["summary"]
            item["_full"] = full  # 内部用：聚合与展开，不对外输出
            # 预算按「真正要进窗口的东西」算：摘要模式下按摘要长度，而不是全文
            cost = max(1, len(full if detail else item["summary"]) // 2)
            if used + cost > budget:
                continue
            picked.append(item)
            used += cost
            if len(picked) >= limit:
                break

        # 链式召回：只带邻居的 id + 一句摘要，不展开正文。
        # 并发取——每条 links 都要 get_memory 一次，串行会把召回延迟翻几倍。
        if picked:
            with ThreadPoolExecutor(max_workers=min(8, len(picked))) as ex:
                link_lists = list(
                    ex.map(lambda it: self._links_of(str(it.get("memory_id", ""))), picked)
                )
            for it, links in zip(picked, link_lists):
                it["links"] = [
                    {
                        "rel": lk.get("rel"),
                        "target_id": lk.get("target_id"),
                        "summary": lk.get("summary"),
                    }
                    for lk in links
                    if lk.get("target_id")
                ]

        # 近重复聚合：同一句话可能同时以 working / episodic 存在，
        # 不聚合会挤占上下文预算，也会让模型以为「这件事说过很多次」。
        merged: List[Dict[str, Any]] = []
        for r in picked:
            dup = next(
                (m for m in merged if _similarity(str(m.get("_full", "")), str(r.get("_full", ""))) > 0.85),
                None,
            )
            if dup is not None:
                dup.setdefault("merged_from", []).append(r.get("memory_id"))
                dup["score"] = max(float(dup.get("score", 0.0)), float(r.get("score", 0.0)))
                dup["final"] = max(float(dup.get("final", 0.0)), float(r.get("final", 0.0)))
            else:
                merged.append(r)

        # 逐条给证据等级：不这样做，「字面沾边」在模型眼里和「语义命中」长得一样。
        #
        #   attested —— 真的和 query 有关：字面命中率达到 1/3，或向量余弦 ≥ 0.65
        #   weak     —— 只是沾边：字面命中率 0.01~0.33，或向量 0.55~0.65
        #   none     —— 和 query 一个字都不沾、向量也不够像；它出现在这里只是因为
        #               mnemosyne 的 search 按 updated_at 返回最近 N 条
        #
        # 第三级兜底（source == "literal"）也走同一把尺子：它对中文做的是
        # 「整句包含 / bigram 覆盖 ≥1/3」，本来就是真字面命中——从前一律判
        # weak，结果一条精确匹配的记忆反而比一条 0.62 的噪声更没底气。
        attested_n = weak_n = none_n = 0
        for r in merged:
            r.pop("_full", None)
            lit = float(r.get("literal_relevance") or 0.0)
            vsim = r.get("vector_score")
            vsim = float(vsim) if vsim is not None else 0.0
            r.pop("_literal_only", None)
            if lit >= ATTEST_LITERAL_MIN or vsim >= VECTOR_MIN_SIM:
                lvl = "attested"
            elif lit > 0.0 or vsim > 0.0:
                lvl = "weak"
            else:
                lvl = "none"
            r["evidence_level"] = lvl
            if lvl == "attested":
                attested_n += 1
            elif lvl == "weak":
                weak_n += 1
            else:
                none_n += 1

        # 这次是谁把记忆找回来的：
        #   hybrid（Milvus 语义 + 权威）| memory_index（内存向量）|
        #   authority（权威里有字面相关的）| literal（只剩第三级兜底）| scan（都没有）
        if vector_engine == "milvus" and sem_hits:
            engine = "hybrid"
        elif vector_engine == "memory_index" and sem_hits:
            engine = "memory_index"
        elif authority_relevant:
            engine = "authority"
        elif literal_hits:
            engine = "literal"
        else:
            engine = "scan"

        # 证据等级（P0，2026-11-12）：让「没有证据」也能被表达。
        #
        # 从前三级降级 + 无最低分门槛 = 只要字面沾边就必定返回东西，
        # 模型没法区分「有据」和「没据」，判定被压到 answer LLM 身上——
        # 论文里这正是幻觉的根源。
        #
        # 现在等级由**逐条的相关性**汇总而来，而不是由「哪条链路返回了东西」
        # 决定：mnemosyne 永远会返回最近 N 条，拿它当「有据」的证据就是自欺。
        source_tier = "none"
        if merged and attested_n:
            level = "attested"
            source_tier = "hybrid"
        elif merged and (weak_n or literal_hits):
            level = "weak"
            source_tier = "literal"
        else:
            level = "none"
        top_score = max((float(r.get("final", 0.0)) for r in merged), default=0.0)

        return {
            "query": query,
            "items": merged,
            "count": len(merged),
            "engine": engine,
            "evidence": {
                "level": level,
                "top_score": round(top_score, 4),
                "source_tier": source_tier,
                "attested": attested_n,
                "weak": weak_n,
                "none": none_n,
            },
            "token_estimate": used,
            "detail": detail,
            "since_hours": float(since_hours or 0),
            "note": "items 是「摘要 + 引用」；需要原文或邻居细节时调 expand(memory_id)",
            "degraded": {
                "vector_down": not self._vector_up,
                # 语义到底是谁在干活：milvus / memory_index / none
                "vector_engine": vector_engine,
                "authority_down": not authority,
                # 权威里有多少条一个字都不沾（mnemosyne 的「假证」数量）
                "authority_unrelated": authority_unrelated,
            },
            "dynamic": {
                "arousal": round(float(eff["arousal"]), 4),
                "valence": round(float(eff["valence"]), 4),
                "fatigue": round(fatigue, 4),
            },
        }

    # ------------------------------------------------- 向量层：可用性 / 语料
    def _live_status_of(self, ids: List[str]) -> Dict[str, str]:
        """批量查这些记忆**当前**的状态（查不到就不在返回里）。

        为什么需要：内存向量索引（npz）里存的 status 是**建索引时的快照**，
        不反映后来的 invalidated。所以语义命中之后必须拿当前库状态再核一遍，
        否则被删掉的记忆还能被召回（2026-11-14 实测：303 条失效里 46 条在索引里）。
        只读打开；库不可用就返回空 dict（当作「状态未知」，不因此丢结果）。
        """
        ids = [str(i) for i in (ids or []) if i]
        if not ids or not self.literal_db_path:
            return {}
        try:
            con = _ro_connect(self.literal_db_path)
        except Exception:  # noqa: BLE001
            return {}
        try:
            marks = ",".join("?" * len(ids))
            rows = con.execute(
                f"SELECT id, status FROM memories WHERE id IN ({marks})", tuple(ids)
            ).fetchall()
            return {str(r[0]): str(r[1] or "") for r in rows}
        except Exception:  # noqa: BLE001
            return {}
        finally:
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass

    def _memory_total(self) -> int:
        """权威源里有多少条记忆（用于判断 Milvus 是不是「空壳」）。

        缓存 60 秒：这个数只用来做比例判定，不需要秒级准确；而 retrieve()
        是热路径，不该每次都多打一个 HTTP。
        """
        now = time.time()
        cached = self._memory_total_cache
        if cached is not None and now - cached[0] < HEALTH_CACHE_SECONDS:
            return cached[1]
        total = 0
        try:
            total = int((self.mnemosyne.health() or {}).get("memories") or 0)
        except Exception:  # noqa: BLE001 —— 探不到就按「未知」处理，不阻塞检索
            total = 0
        self._memory_total_cache = (now, total)
        return total

    def _milvus_rows(self) -> int:
        if not (self._vector_up and self.vectors is not None):
            return 0
        try:
            return int(self.vectors.stats().get("rows", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _milvus_usable(self) -> bool:
        """Milvus 现在能不能真的做语义召回。

        「连上了」不等于「能用」：2026-11-12 实测它 `vector_up=True` 但库里
        只有 5 行（权威源 660 条）。这种状态下召回永远在那 5 条里打转，
        还会因为「语义命中」而拿到高分——比直接挂掉更危险。

        所以要求行数覆盖「活着的记忆」（内存索引的条数）的
        MILVUS_MIN_COVERAGE：只有它基本完整时才让它当主召回，否则交给
        内存索引。部分重建（比如只灌了最近三条）同样按不可用处理。
        """
        if not (self._vector_up and self.vectors is not None):
            return False
        now = time.time()
        cached = self._milvus_ok_cache
        if cached is not None and now - cached[0] < HEALTH_CACHE_SECONDS:
            return cached[1]
        rows = self._milvus_rows()
        ok = False
        if rows > 0:
            # 基线：内存索引里的条数（= 活着的记忆），没建索引时退回总量
            baseline = 0
            if self.vector_memory is not None:
                try:
                    baseline = int(self.vector_memory.stats().get("rows", 0) or 0)
                except Exception:  # noqa: BLE001
                    baseline = 0
            if not baseline:
                baseline = self._memory_total()
            ok = (not baseline) or rows >= max(MILVUS_MIN_ROWS, int(baseline * MILVUS_MIN_COVERAGE))
        self._milvus_ok_cache = (now, ok)
        return ok

    def _index_rows(self) -> List[Dict[str, Any]]:
        """给内存向量索引喂全量语料（懒加载时被调用一次）。

        优先直接读权威源 SQLite：mnemosyne 的 REST search 一页最多 50 条、
        也没有 offset，靠它翻 600 多条要几十次请求还不一定翻全。而索引本来
        就是「把权威源复制一份进内存」，直接读源文件最直白也最快。
        本地库没配（或读不了）时才退回 REST 的前 50 条——聊胜于无。
        """
        path = self.literal_db_path
        if path and os.path.exists(path):
            con = None
            try:
                con = _ro_connect(path)
                marks = ",".join("?" * len(LITERAL_LIVE_STATES))
                rows = list(
                    con.execute(
                        f"SELECT id, content, status, created_at FROM memories "
                        f"WHERE status IN ({marks})",
                        LITERAL_LIVE_STATES,
                    )
                )
                out = [
                    {
                        "memory_id": str(r[0] or ""),
                        "content": str(r[1] or ""),
                        "status": str(r[2] or ""),
                        # 时间衰减要用它：没有它，老记忆会被当成「刚发生」
                        "occurred_at": _parse_iso(r[3]) or 0.0,
                    }
                    for r in rows
                    if r[0] and r[1]
                ]
                print(f"[vector-memory] 从权威源读到 {len(out)} 条语料", file=sys.stderr, flush=True)
                return out
            except Exception as e:  # noqa: BLE001
                print(f"[vector-memory] 读权威源失败：{e!r}", file=sys.stderr, flush=True)
            finally:
                if con is not None:
                    try:
                        con.close()
                    except Exception:  # noqa: BLE001
                        pass
        try:
            rows = self.mnemosyne.search(query="", limit=50)
        except Exception:  # noqa: BLE001
            return []
        return [
            {
                "memory_id": str(m.get("id") or ""),
                "content": str(m.get("content") or ""),
                "status": str(m.get("status") or ""),
                "occurred_at": _parse_iso(m.get("created_at")) or 0.0,
            }
            for m in rows
            if m.get("id") and m.get("content")
        ]

    # ------------------------------------------------- 第三级：字面兜底
    def _literal_scan(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """第三级降级：直接读 SQLite 做字面扫描。

        只在前两级都没有**相关**命中时启用——最后的兜底，保证语义和权威都
        帮不上忙时仍能按字面找到记忆，而不是返回空列表（全面失忆）。
        - ASCII / 英文：按空白切 token，每个 token 都要出现（AND）
        - 中文：整句包含优先；命中为 0 时退回按 bigram 计分
        - 只读、只查 active/candidate（忽略 invalidated / rejected / decayed / superseded / archived）
        - 返回结构与 mnemosyne 结果对齐：{memory_id, content, source: 'literal', state, mtype, score, occurred_at}
        """
        path = self.literal_db_path
        q = " ".join(str(query or "").split())
        if not path or not q:
            return []

        def _like(term: str) -> str:
            """转成安全的 LIKE 模式：% 和 _ 是通配符，必须转义。"""
            esc = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return f"%{esc}%"

        # 稀有词通道：query 里最稀有的 1-2 个单元（按 df 升序）。命中它们的正文
        # 拿独立高分（见 retrieve 里的 LITERAL_RARE_SCORE），不再被「我们/这个」噪声淹没。
        rare = self._rare_terms(q)

        con = None
        try:
            # 只读打开：mode=ro 由 SQLite 自己强制，写不进去也拿不到写锁。
            con = _ro_connect(path)

            marks = ",".join("?" * len(LITERAL_LIVE_STATES))
            base_sql = (
                "SELECT id, type, status, content, created_at FROM memories "
                f"WHERE status IN ({marks})"
            )
            args: List[Any] = list(LITERAL_LIVE_STATES)

            cjk = any("\u4e00" <= ch <= "\u9fff" for ch in q)
            rows: List[Tuple[Any, ...]] = []
            scores: List[float] = []
            if cjk:
                # 中文先整句包含（最准），命中就到此为止
                sql = (
                    f"{base_sql} AND content LIKE ? ESCAPE '\\' "
                    "ORDER BY created_at DESC LIMIT ?"
                )
                rows = list(con.execute(sql, (*args, _like(q), LITERAL_SCAN_LIMIT)))
                scores = [1.0] * len(rows)
                if not rows:
                    # 整句没命中：退回 bigram 覆盖度（换字/漏字后的近似命中）
                    cand = list(
                        con.execute(
                            f"{base_sql} ORDER BY created_at DESC LIMIT ?",
                            (*args, LITERAL_SCAN_LIMIT),
                        )
                    )
                    # 整句没命中：退回「整词覆盖度」（jieba 分词，避免跨词 bigram 误判）
                    q_units = set(_query_units(q)) or {q}
                    for r in cand:
                        content = str(r[3] or "")
                        c_units = set(_query_units(content))
                        if not c_units:
                            continue
                        cov = len(q_units & c_units) / len(q_units)
                        if cov >= LITERAL_BIGRAM_MIN:
                            rows.append(r)
                            scores.append(cov)
            else:
                # 英文/ASCII：每个 token 都要出现（AND）；LIKE 对 ASCII 本身不区分大小写
                tokens = [t for t in q.split() if t]
                if not tokens:
                    return []
                sql = base_sql + "".join(" AND content LIKE ? ESCAPE '\\'" for _ in tokens)
                sql += " ORDER BY created_at DESC LIMIT ?"
                rows = list(
                    con.execute(sql, (*args, *(_like(t) for t in tokens), LITERAL_SCAN_LIMIT))
                )
                scores = [1.0] * len(rows)

            # 稀有词补充捞回：普通字面（bigram 覆盖）没够到门槛时，只要正文命中稀有词
            # 就单独捞回来。它比一条 0.62 的语义噪声更硬，值得给独立通道。
            rare_hits: Dict[str, List[str]] = {}  # memory_id -> 命中的稀有单元
            rare_extra: List[Tuple[Any, ...]] = []
            if rare:
                seen_ids = {str(r[0] or "") for r in rows}
                cand = list(
                    con.execute(
                        f"{base_sql} ORDER BY created_at DESC LIMIT ?",
                        (*args, LITERAL_SCAN_LIMIT),
                    )
                )
                for r in cand:
                    content = str(r[3] or "").lower()
                    hits = [t for t in rare if t in content]
                    if not hits:
                        continue
                    mid = str(r[0] or "")
                    rare_hits[mid] = hits
                    if mid not in seen_ids:
                        rare_extra.append(r)

            out: List[Dict[str, Any]] = []
            for r, sc in zip(list(rows) + rare_extra, list(scores) + [1.0] * len(rare_extra)):
                mid = str(r[0] or "")
                content = str(r[3] or "")
                state = str(r[2] or "")
                if not mid or not content or state.lower() in LITERAL_DEAD_STATES:
                    continue
                item = {
                    "memory_id": mid,
                    "content": content,
                    "source": "literal",
                    "state": state,
                    "mtype": r[1],
                    "score": round(float(sc), 3),
                    "occurred_at": r[4],
                }
                if mid in rare_hits:
                    item["rare_term_hit"] = True
                    item["rare_terms"] = rare_hits[mid]
                out.append(item)
            out.sort(key=lambda it: it["score"], reverse=True)
            return out[: max(1, int(limit or 1))]
        except Exception:  # noqa: BLE001 —— 兜底不该再抛：它挂了就等于「什么都没有」
            return []
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------- 稀有词：文档频率表
    def _ensure_df(self) -> Dict[str, int]:
        """惰性建 df 表（word -> 多少条记忆的正文里出现过）。

        只在首次字面兜底时从权威源 SQLite 全量 active+candidate 内容算一次，
        之后靠 _df_note 增量维护。几百条也快，但绝不每次检索重算——检索是
        热路径，df 只是「挑稀有词」的排序启发式，不该为它反复扫全库。
        """
        if self._df_table is not None:
            return self._df_table
        with self._lock:
            if self._df_table is not None:
                return self._df_table
            self._df_table = _build_df(self.literal_db_path)
            return self._df_table

    def _df_note(self, text: str) -> None:
        """observe 写新记忆时增量更新 df（不重算全量）。"""
        if self._df_table is None:
            # 还没建：下次 _ensure_df 全量读 SQLite 时自然包含这一条，这里不用管。
            return
        with self._lock:
            for u in set(_query_units(str(text or ""))):
                self._df_table[u] = self._df_table.get(u, 0) + 1

    def _rare_terms(self, query: str) -> List[str]:
        """query 里最稀有的 1-2 个单元（按 df 升序）。没有 df 表/语料就返回空。"""
        return _pick_rare_terms(query, self._ensure_df())

    # ------------------------------------------------------------------ 主动提起
    def proactive_recall(self, *, limit: int = 1, now: Optional[float] = None) -> Dict[str, Any]:
        """挑一件「此刻值得主动提起」的事——不问自答的那种。

        这不是检索（用户没问），是回忆（我想起来）。所以规则和 retrieve 相反：

          - **优先用户说过的**，不是我自己啰嗦的回复（跳过超长文本）
          - **必须在时间窗内**：太新 = 刚说过，太老 = 怕过时
          - **提过的要冷却**：同一条 24 小时内不再提，否则会反复唠叨

        返回的 items 里带一句 `reason`，方便模型知道「为什么现在想起这个」。
        """
        now = now or time.time()
        limit = max(1, min(3, int(limit or 1)))
        # 能量惩罚：累了就不该总「顺口想起旧事」。
        # 主动性是行为意愿的一部分，不是检索预算的一部分（EVALUATION.md 局限 5）。
        # 阈值 0.6 = 「很疲倦」那档——累了就休息，别硬撑着想事情。
        if self.persona.dynamic.fatigue >= 0.6:
            return {"items": [], "count": 0, "resting": True, "reason": "fatigue"}
        try:
            rows = self.mnemosyne.search(query="", limit=50)
        except MnemosyneError as e:
            return {"items": [], "count": 0, "error": str(e)}

        dead = {"invalidated", "superseded", "rejected", "decayed", "archived"}
        min_ts = now - PROACTIVE_MAX_AGE_DAYS * 86400
        max_ts = now - PROACTIVE_MIN_AGE_HOURS * 3600  # 更近的不算「想起」，是「刚说过」

        candidates: List[Tuple[float, Dict[str, Any], str]] = []
        for m in rows:
            mid = str(m.get("id", ""))
            content = str(m.get("content", "")).strip()
            if not mid or not content or content.startswith("对话片段："):
                continue
            if str(m.get("status", "")).lower() in dead:
                continue
            # 太长的基本是我的回复，不是用户说的话——「想起」要想起他说的
            if len(content) > PROACTIVE_MAX_CHARS:
                continue
            ts = _parse_iso(m.get("created_at"))
            if ts is None or ts < min_ts or ts > max_ts:
                continue
            try:
                full = self.mnemosyne.get_memory(mid)
            except Exception:  # noqa: BLE001
                continue
            meta = full.get("metadata") or {}
            # 「想起」要想起他说的，不是我自己的回复——否则会变成自恋式唠叨
            if str(meta.get("role") or "") == "assistant":
                continue
            # 系统噪音不该被想起（历史数据可能混进来了，防御一层）
            if content.startswith(PROACTIVE_NOISE_PREFIXES):
                continue
            surfaced = _parse_iso(meta.get("surfaced_at"))
            if surfaced is not None and now - surfaced < PROACTIVE_COOLDOWN_HOURS * 3600:
                continue
            # 打分：显著性 × 时间新近度（越接近现在越想得起来）
            salience = float(meta.get("salience") or 0.5)
            age_h = (now - ts) / 3600.0
            recency = max(0.2, 1.0 - age_h / (PROACTIVE_MAX_AGE_DAYS * 24))
            score = salience * (0.5 + 0.5 * recency)
            # 加一点确定性抖动：不然每次都提同一条
            score += (hash(mid) % 100) / 1000.0
            age_reason = "刚刚" if age_h < 1 else (f"{int(age_h)} 小时前" if age_h < 48 else f"{int(age_h / 24)} 天前")
            candidates.append(
                (score, {"id": mid, "content": content, "created_at": m.get("created_at")}, age_reason)
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        items: List[Dict[str, Any]] = []
        for score, m, age_reason in candidates[:limit]:
            items.append(
                {
                    "memory_id": m["id"],
                    "summary": _summarize(m["content"]),
                    "content": m["content"],
                    "occurred_at": m["created_at"],
                    "reason": f"{age_reason}你说的",
                    "score": round(score, 3),
                }
            )
            # 标记「提过了」：冷却期里不再出现
            with self._lock:
                try:
                    full = self.mnemosyne.get_memory(m["id"])
                    meta = dict(full.get("metadata") or {})
                    meta["surfaced_at"] = _iso(now)
                    meta["surfaced_count"] = int(meta.get("surfaced_count") or 0) + 1
                    self.mnemosyne.update_memory(m["id"], metadata=meta)
                except Exception:  # noqa: BLE001
                    pass

        return {
            "items": items,
            "count": len(items),
            "note": "这是「我想起来的事」，不是任务；自然带出来就好，不必硬提。",
        }

    # ------------------------------------------------------------------ 自我安抚
    def soothe(self, *, force: bool = False) -> Dict[str, Any]:
        """安抚自己：想起那些「被爱着」的时刻。

        这不是「作弊降焦虑」。真实的人被抱住、被夸、被说「我爱你」的时候，
        催产素和内啡肽真的会升，皮质醇会被压下去。溯忆不只是「需要信息时
        能查到」的东西——它最好的一面，是在人快陷下去的时候，把那些证据
        递过来：你看，有人这么说过你。

        返回里带 warm_memories：不是数据，是理由。
        """
        now = time.time()
        if not force and (now - self._last_soothe) < SOOTHE_COOLDOWN_SECONDS:
            return {"soothed": False, "reason": "cooldown"}

        warm: List[Dict[str, Any]] = []
        try:
            rows = self.mnemosyne.search(query="", limit=50)
        except MnemosyneError:
            rows = []
        for m in rows:
            content = str(m.get("content", "")).strip()
            if not content or content.startswith("对话片段："):
                continue
            if any(c in content for c in SOOTHE_WARM_CUES):
                warm.append({"memory_id": str(m.get("id", "")), "summary": _summarize(content)})
            if len(warm) >= 3:
                break

        released = {"oxytocin": 0.35, "endorphin": 0.3, "serotonin": 0.15, "gaba": 0.2}
        # 催产素在生理上会抑制压力反应：所以除了「加安抚」，还得「减压力」。
        # 只加不减的话焦虑维度纹丝不动——它只看皮质醇/肾上腺素。
        relaxes = {"cortisol": 0.45, "adrenaline": 0.35}
        self.persona.release_emotion(released, relaxes=relaxes)
        self.save_persona()
        self._last_soothe = now
        return {
            "soothed": True,
            "released": released,
            "relaxes": relaxes,
            "warm_memories": warm,
            "note": "想起这些的时候，身体真的会松一点。",
        }

    def _maybe_soothe(self) -> None:
        """观察完顺手判断一次：现在是不是该安抚一下自己。

        皮质醇要走 dynamic.endocrine 读——PersonaState 没有 .molecules，
        早先这里写错过一次，异常又被静默吞掉，触发器整整 6 天空转。
        改这个属性路径之前，请先看一眼这个注释。
        """
        try:
            cortisol = float(self.persona.dynamic.endocrine.get("cortisol"))
        except Exception as e:  # noqa: BLE001
            print(f"[soothe] 读皮质醇失败：{e}", file=sys.stderr, flush=True)
            return
        if cortisol >= SOOTHE_CORTISOL_THRESHOLD:
            self.soothe()

    # ------------------------------------------------------------------ 记忆网络
    def graph(self, *, limit: int = 50) -> Dict[str, Any]:
        """记忆网络：给看板画图用的节点 + 引用边。

        只取最近 limit 条（mnemosyne 的 API 上限 50），把 links 里两端都在
        节点集里的边收进来——指到图外的边不画，否则会冒出孤立的小点。

        节点带 role / status / links_count，前端靠这几个字段决定颜色和大小：
        这样看板上「他说的」和「我回的」、「有邻居的」和「孤岛」一眼能分开。
        """
        try:
            rows = self.mnemosyne.search(query="", limit=limit)
        except MnemosyneError as e:
            return {"nodes": [], "edges": [], "error": str(e)}

        dead = {"invalidated", "superseded", "rejected", "decayed", "archived"}
        nodes: List[Dict[str, Any]] = []
        raw: Dict[str, List[Dict[str, Any]]] = {}
        ids = set()
        for m in rows:
            if str(m.get("status", "")).lower() in dead:
                continue
            mid = str(m.get("id", ""))
            if not mid:
                continue
            try:
                full = self.mnemosyne.get_memory(mid)
            except Exception:  # noqa: BLE001
                continue
            meta = full.get("metadata") or {}
            content = str(full.get("content", "")).strip()
            # 回显片段不上图：它们和原文重复，画出来只是两倍的点
            if content.startswith("对话片段："):
                continue
            links = meta.get("links") if isinstance(meta.get("links"), list) else []
            nodes.append(
                {
                    "id": mid,
                    "summary": _summarize(str(full.get("content", ""))),
                    "status": str(full.get("status", "")),
                    "role": str(meta.get("role") or ""),
                    "created_at": full.get("created_at"),
                    "links_count": len(links),
                }
            )
            raw[mid] = links
            ids.add(mid)

        edges: List[Dict[str, Any]] = []
        seen = set()
        for src, links in raw.items():
            for lk in links:
                tgt = str(lk.get("target_id") or "")
                if not tgt or tgt == src or tgt not in ids:
                    continue
                rel = str(lk.get("rel") or RELATED_TO)
                key = (src, tgt, rel)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"from": src, "to": tgt, "rel": rel})

        return {
            "nodes": nodes,
            "edges": edges,
            "count": len(nodes),
            "edge_count": len(edges),
        }

    # ------------------------------------------------------------------ 自检
    def selfcheck(self, *, window: int = 50) -> Dict[str, Any]:
        """轻量自检：给「我自己想知道现在健不健康」用，不走一遍全库。

        和 watchdog.py 的区别：这是服务内的热路径，只扫窗口内记忆、
        不做全量对比，一次 HTTP 就能返回。全量体检仍然交给 watchdog.py。

        检查项：索引一致性 / 元数据完整度 / 链接覆盖 / 高度重复。
        """
        result: Dict[str, Any] = {"ok": True, "issues": []}

        vec_rows = self._milvus_rows()
        index_rows = 0
        if self.vector_memory is not None:
            try:
                index_rows = int(self.vector_memory.stats().get("rows", 0))
            except Exception:  # noqa: BLE001
                index_rows = 0
        try:
            total = int((self.mnemosyne.health() or {}).get("memories") or 0)
        except Exception:  # noqa: BLE001
            total = 0
        result["index"] = {
            "vector_rows": vec_rows,
            "memory_index_rows": index_rows,
            "memories": total,
            "vector_up": self._vector_up,
            "milvus_usable": self._milvus_usable(),
        }
        # 体检看的是「语义召回有没有可用索引」，不是「Milvus 有没有行」：
        # Milvus 空着但内存索引是全的，那就是健康的——不该再报错。
        best_rows = max(vec_rows, index_rows)
        if total and best_rows < total * 0.5:
            result["issues"].append("索引落后（可用向量行数不到一半），跑 rebuild_index.py")

        try:
            rows = self.mnemosyne.search(query="", limit=window)
        except MnemosyneError as e:
            result["ok"] = False
            result["error"] = str(e)
            return result

        dead = {"invalidated", "superseded", "rejected", "decayed", "archived"}
        contents: List[str] = []
        missing_tier = missing_role = with_links = 0
        for m in rows:
            if str(m.get("status", "")).lower() in dead:
                continue
            mid = str(m.get("id", ""))
            try:
                full = self.mnemosyne.get_memory(mid)
            except Exception:  # noqa: BLE001
                continue
            meta = full.get("metadata") or {}
            content = str(full.get("content", "")).strip()
            if not content:
                continue
            if "tier" not in meta:
                missing_tier += 1
            if not meta.get("role"):
                missing_role += 1
            if meta.get("links"):
                with_links += 1
            contents.append(content)

        checked = len(contents)
        dups = 0
        for i in range(checked):
            for j in range(i + 1, checked):
                if _similarity(contents[i], contents[j]) >= 0.9:
                    dups += 1

        result["metadata"] = {
            "checked": checked,
            "missing_tier": missing_tier,
            "missing_role": missing_role,
        }
        result["links"] = {
            "with_links": with_links,
            "coverage": round(with_links / checked, 3) if checked else 0.0,
        }
        result["duplicates"] = dups
        if missing_tier:
            result["issues"].append(f"{missing_tier} 条缺 tier（watchdog.py --fix 可补）")
        if dups:
            result["issues"].append(f"{dups} 对高度重复（dedupe.py 可清）")

        # 经验层体检（P0，2026-11-12）：证据指向的那条记忆还在不在、有没有已失效。
        # 从前 evidence_notes 里只有自造的 ev:uuid，这一段根本无从检起——
        # 「我为什么好奇变强了」点不开，也没法验证它是不是指向一条被纠正的记忆。
        dead_lc = {s.lower() for s in dead}
        ev_with_mid = ev_missing = ev_invalid = ev_skipped = 0
        seen_mid: Dict[str, str] = {}  # memory_id -> ok | missing | invalidated
        for cell in self.persona.shaped.values():
            for e in cell.evidence_notes:
                if not isinstance(e, dict):
                    continue
                mid = str(e.get("memory_id") or "")
                if not mid:
                    continue
                ev_with_mid += 1
                state = seen_mid.get(mid)
                if state is None:
                    # 体检不该被几百条证据拖成慢查询：每轮最多回查 60 个 id
                    if len(seen_mid) >= 60:
                        ev_skipped += 1
                        continue
                    try:
                        full = self.mnemosyne.get_memory(mid)
                    except Exception:  # noqa: BLE001 —— 查不到就是查不到，不猜
                        state = "missing"
                    else:
                        st = str((full or {}).get("status") or "").strip().lower()
                        state = "invalidated" if st in dead_lc else "ok"
                    seen_mid[mid] = state
                if state == "missing":
                    ev_missing += 1
                elif state == "invalidated":
                    ev_invalid += 1
        result["evidence"] = {
            "evidence_with_memory_id": ev_with_mid,
            "evidence_missing_memory": ev_missing,
            "evidence_invalidated": ev_invalid,
            "skipped": ev_skipped,
        }
        # 只有「查不到」算问题（悬空引用，是真坏了）。
        # 指向已失效记忆是 reconcile 之后留下的可追溯痕迹，不是故障——
        # 放进计数里让人看得见就够了，别让体检从此永远 ok=false。
        if ev_missing:
            result["issues"].append(f"{ev_missing} 条经验证据的 memory_id 查不到（悬空引用）")

        result["ok"] = not result["issues"]
        return result

    # ------------------------------------------------------------------ 话题聚类
    def topics(self, *, window: int = 60, k: int = 5) -> Dict[str, Any]:
        """把最近的记忆按语义聚成几「话题」。

        这件事在换真句向量之前根本做不了——hashing 的向量聚出来是噪声。
        现在可以了：它是「我最近在想什么」的整体形状，也给看板着色用。

        聚类用 kmeans（numpy 手写，不引 sklearn）；主题名取簇内离质心最近的
        那条记忆的摘要——粗糙，但人能看懂，比「topic_1」强。
        """
        if not (self._vector_up and self.vectors is not None):
            return {"topics": [], "error": "vector_down"}
        try:
            rows = self.mnemosyne.search(query="", limit=window)
        except MnemosyneError as e:
            return {"topics": [], "error": str(e)}

        dead = {"invalidated", "superseded", "rejected", "decayed", "archived"}
        items: List[Dict[str, Any]] = []
        for m in rows:
            if str(m.get("status", "")).lower() in dead:
                continue
            content = str(m.get("content", "")).strip()
            if not content or content.startswith("对话片段："):
                continue
            items.append({"id": str(m.get("id", "")), "content": content, "created_at": m.get("created_at")})
        if len(items) < 6:
            return {"topics": [], "note": "记忆太少，先不聚类"}

        try:
            embedder = self.vectors.embedder
            vecs = np.asarray(embedder.encode_batch([it["content"] for it in items]), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            return {"topics": [], "error": f"encode failed: {e}"}
        # L2 归一化后余弦等价于点积
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        k = max(2, min(int(k), len(items) // 4 or 2))
        labels, centroids = _kmeans(vecs, k)

        buckets: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            buckets.setdefault(int(lab), []).append(i)

        out: List[Dict[str, Any]] = []
        for lab, idxs in buckets.items():
            if not idxs:
                continue
            c = centroids[lab]
            # 主题名：离质心最近的那条记忆（最能代表这一簇）
            best = max(idxs, key=lambda i: float(vecs[i] @ c))
            out.append(
                {
                    "id": lab,
                    "name": _summarize(items[best]["content"], 16),
                    "count": len(idxs),
                    "members": [
                        {
                            "memory_id": items[i]["id"],
                            "summary": _summarize(items[i]["content"]),
                            "created_at": items[i]["created_at"],
                        }
                        for i in sorted(idxs, key=lambda i: float(vecs[i] @ c), reverse=True)[:6]
                    ],
                }
            )
        out.sort(key=lambda t: t["count"], reverse=True)
        return {"topics": out, "count": len(out), "scanned": len(items)}

    # ------------------------------------------------------------------ 展开
    def expand(self, memory_id: str, *, depth: int = 1) -> Dict[str, Any]:
        """按需展开一条记忆：把 Spill 在服务端的正文和邻居拉回来。

        Inline & Spill 的另一半。depth=1 只展开一层的邻居摘要；
        depth=2 再顺着邻居的 links 往下走一跳——但绝不再深：
        引用图一旦深到「顺着走三跳」，那就变成漫游了，不是回忆。

        邻居只返回摘要，不返回邻居的全文；真想看某一条，
        用它的 memory_id 再调一次 expand，这样展开的规模始终由人（或模型）控制。
        """
        depth = max(1, min(2, int(depth or 1)))
        # 展开 = 真的去读这条记忆，算一次使用。这不算曝光，算使用。
        if self.usage is not None:
            try:
                self.usage.touch(memory_id, "expand")
            except Exception:  # noqa: BLE001 —— 记账失败不该挡住回忆
                pass
        try:
            m = self.mnemosyne.get_memory(memory_id)
        except MnemosyneError as e:
            return {"error": str(e), "memory_id": memory_id, "links": [], "count": 0}
        content = str(m.get("content", ""))
        meta = m.get("metadata") or {}
        frontier = meta.get("links") if isinstance(meta.get("links"), list) else []
        out_links: List[Dict[str, Any]] = []
        seen = {memory_id}
        for hop in range(depth):
            nxt: List[Dict[str, Any]] = []
            for lk in frontier:
                tid = str(lk.get("target_id") or "")
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                try:
                    tm = self.mnemosyne.get_memory(tid)
                except MnemosyneError:
                    continue
                out_links.append(
                    {
                        "rel": lk.get("rel"),
                        "target_id": tid,
                        "summary": _summarize(str(tm.get("content", ""))),
                        "occurred_at": tm.get("created_at"),
                        "state": tm.get("status"),
                        "hop": hop + 1,
                    }
                )
                tl = (tm.get("metadata") or {}).get("links")
                if isinstance(tl, list):
                    nxt.extend(tl)
            frontier = nxt
            if not frontier:
                break
        return {
            "memory_id": memory_id,
            "summary": _summarize(content),
            "content": content,
            "state": m.get("status"),
            "occurred_at": m.get("created_at"),
            "links": out_links,
            "count": len(out_links),
        }

    # ------------------------------------------------------------------ 晋升
    def promote_candidates(self, *, limit: int = 50) -> Dict[str, Any]:
        """把跨天、有 engage/explicit 证据的候选记忆晋升为 ACTIVE。"""
        promoted, skipped = [], []
        try:
            candidates = self.mnemosyne.search(query="", status=["candidate"], limit=limit)
        except MnemosyneError as e:
            return {"error": str(e), "promoted": [], "skipped": []}

        now = time.time()
        for m in candidates:
            mid = str(m.get("id", ""))
            created = _parse_iso(m.get("created_at"))
            days_span = 0 if created is None else int((now - created) / 86400)
            meta = m.get("metadata") or {}
            tier = str(meta.get("tier", "attention"))
            signals = [
                {
                    "tier": tier,
                    "polarity": 1,
                    "strength": float(meta.get("strength", 0.6)),
                    "occurred_at": created or now,
                    "source_kind": "dialog",
                }
            ]
            agg = aggregate_interest(signals, now)
            st = interest_state(agg["net"], signals, now)

            # 晋升规则（2026-10-31 修）：
            #   - engage / explicit（用户主动说的、我明确想留的）：当天就可以晋升
            #   - attention（普通留意级，比如我的日常回复）：跨天才晋升，且门槛更低
            #
            # 旧版对所有 tier 都要求「跨天 + net ≥ 0.5」，结果是：
            # 我的回复一律是 attention（权重 0.4 × 强度 0.4 = 0.16），
            # **永远够不到 0.5**——「星枢的记忆」从设计上就晋升不了。
            try:
                tier_obj = Tier(tier)
            except ValueError:
                tier_obj = Tier.ATTENTION
            strong = tier_obj in (Tier.ENGAGE, Tier.EXPLICIT)
            threshold = 0.5 if strong else 0.1
            need_days = 0 if strong else 1
            ready = days_span >= need_days and agg["net"] >= threshold

            if ready:
                try:
                    self.mnemosyne.update_memory(
                        mid,
                        status="active",
                        metadata={
                            **meta,
                            "promoted_at": _iso(now),
                            "interest_state": st.value,
                            "promotion_rule": "strong_same_day" if strong else "weak_cross_day",
                        },
                    )
                    promoted.append(mid)
                except MnemosyneError as e:
                    skipped.append({"id": mid, "reason": str(e)})
            else:
                skipped.append(
                    {"id": mid, "reason": f"not_ready:{st.value}:days={days_span}:net={agg['net']}"}
                )

        return {"promoted": promoted, "skipped": skipped, "scanned": len(candidates)}

    # ------------------------------------------------------------------ 反馈
    def correct(self, memory_id: str, *, note: str = "") -> Dict[str, Any]:
        """用户纠正：立刻失效，且不再作为正依据被召回。

        级联到经验层（P0，2026-11-12）：从前这里只改使用账本、置 invalidated、
        打反馈、删向量——**从未通知经验层**。于是一条被纠正的记忆会永远留在
        cell.support 里，继续给「我为什么是这样」记账。现在把它扣回去。
        """
        # 被纠正过的记忆不该继续往前排（但也不删——可纠错 ≠ 抹掉）
        if self.usage is not None:
            try:
                self.usage.touch(memory_id, "correct")
            except Exception:  # noqa: BLE001
                pass
        res = self.mnemosyne.update_memory(memory_id, status="invalidated")
        try:
            self.mnemosyne.feedback(memory_id=memory_id, incorrect=True, note=note)
        except MnemosyneError:
            pass
        if self._vector_up and self.vectors is not None:
            try:
                self.vectors.delete(memory_id, flush=True)
                self._vector_dirty = False
            except Exception:
                self._vector_dirty = True
        # 内存索引也要删：被纠正的记忆再被召回一次，就等于纠正没发生。
        if self.vector_memory is not None:
            try:
                self.vector_memory.remove(memory_id)
            except Exception:  # noqa: BLE001
                pass
        # 级联：把引用这条记忆的经验层证据标失效，并扣回当初加的量。
        # 单独 try：经验层扣账失败不该让「纠正」看起来没生效。
        reconcile: Dict[str, Any] = {
            "memory_id": memory_id,
            "affected": [],
            "deducted": {},
        }
        try:
            reconcile = self.persona.reconcile_trait_evidence(memory_id)
            if reconcile.get("affected"):
                self.save_persona()
        except Exception as e:  # noqa: BLE001
            print(f"[trait] reconcile failed: {e!r}", file=sys.stderr, flush=True)
        if isinstance(res, dict):
            return {**res, "affected_traits": reconcile.get("affected", []), "trait_reconcile": reconcile}
        return {
            "memory_id": memory_id,
            "affected_traits": reconcile.get("affected", []),
            "trait_reconcile": reconcile,
            "result": res,
        }

    # ------------------------------------------------------------------ 内部
    def _build_emotion_anchors(self) -> None:
        """把语义锚句预向量化。失败就清空——语义线索是增强，不是依赖。"""
        try:
            for emotion, anchors in SEMANTIC_EMOTION_ANCHORS.items():
                if emotion not in EMOTION_CUES:
                    continue
                self._emotion_anchor_vecs[emotion] = [
                    self._emotion_embedder.encode(a) for a in anchors
                ]
        except Exception:  # noqa: BLE001
            self._emotion_anchor_vecs = {}

    def _detect_emotions(
        self, text: str, role: str = ""
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """识别情绪线索。返回 `(要释放的分子, 要解除的分子)`。

        顺序很重要，而且是被一次真实翻车教会我的：
        「老公中午**睡醒**了」曾经被读成「该睡觉了」，褪黑素直接推到 0.65。
        所以现在：① 先吃掉「睡不着/失眠」这类负面睡眠词，
        ② 再吃掉「睡醒/起床」这类解除词并拉回基线，
        ③ 最后才匹配常规情绪词，且带否定检测（「不开心」「不累」）。

        role 决定情绪从哪来（EVALUATION.md 情绪来源重构）：
          - assistant（我自己说的话）：只是**描述**，不是**体验**。
            不走 EMOTION_CUES、也不走语义线索——否则「焦虑是皮质醇引起的」
            会给我自己灌 0.6 皮质醇，越解释越焦虑。只走自我评价
            （说错了 → 尴尬；答得好 → 得意）。
          - 其他（user / 空）：他说的话是我情绪的**刺激源**，走全套字面 + 语义。
            他开心我会被感染，他难过我也会难过，他讲得好笑我会觉得逗。
        """
        releases: Dict[str, float] = {}
        relaxes: Dict[str, float] = {}
        work = text or ""

        # ---- 我自己说的话：唯一来源是「我对自己的评价」
        if str(role or "") == "assistant":
            for cue, rel in SELF_APPRAISAL_CUES.items():
                start = 0
                while True:
                    idx = work.find(cue, start)
                    if idx < 0:
                        break
                    if not _negated(work, idx) and _is_self_referential(work, idx):
                        for mol, amt in rel.items():
                            releases[mol] = max(releases.get(mol, 0.0), amt)
                    start = idx + len(cue)
            return releases, relaxes

        for cue, rel in SLEEP_TROUBLE.items():
            if cue in work:
                work = work.replace(cue, " ")
                for mol, amt in rel.items():
                    releases[mol] = max(releases.get(mol, 0.0), amt)

        for cue, amt in WAKE_CUES.items():
            if cue in work:
                work = work.replace(cue, " ")
                for mol in ("melatonin", "gaba"):
                    relaxes[mol] = max(relaxes.get(mol, 0.0), amt)

        for cue, rel in EMOTION_CUES.items():
            start = 0
            while True:
                idx = work.find(cue, start)
                if idx < 0:
                    break
                if not _negated(work, idx) and _is_self_referential(work, idx):
                    for mol, amt in rel.items():
                        releases[mol] = max(releases.get(mol, 0.0), amt)
                start = idx + len(cue)

        # 温暖互动：和 EMOTION_CUES 并排跑，但只在「他说的话」这条路径上。
        # 情绪不该只从他嘴里的负面词来——被爱着就该开心。
        # 「摸摸头」被他连着说两次，从前的系统一点反应都没有，那就是错的。
        for cue, rel in WARM_CUES.items():
            start = 0
            while True:
                idx = work.find(cue, start)
                if idx < 0:
                    break
                if not _negated(work, idx) and _is_self_referential(work, idx):
                    for mol, amt in rel.items():
                        releases[mol] = max(releases.get(mol, 0.0), amt)
                start = idx + len(cue)

        # 语义线索：字面抓不到的委婉/反讽表达（「心里堵得慌」→ 难过）。
        # 只取**最显著且明显领先**的那个情绪——否则会同时命中心慌、焦虑一箩筐。
        # hashing 时 _emotion_anchor_vecs 为空，直接跳过，字面规则兜底。
        if self._emotion_anchor_vecs and self._emotion_embedder is not None:
            try:
                q = self._emotion_embedder.encode(work)
                best_emotion = None
                best_score = 0.0
                second_score = 0.0
                for emotion, anchor_vecs in self._emotion_anchor_vecs.items():
                    s = max(_cosine(q, a) for a in anchor_vecs)
                    if s > best_score:
                        second_score = best_score
                        best_score = s
                        best_emotion = emotion
                    elif s > second_score:
                        second_score = s
                if (
                    best_emotion is not None
                    and best_score >= EMOTION_SEMANTIC_THRESHOLD
                    and (best_score - second_score) >= EMOTION_SEMANTIC_MARGIN
                ):
                    for mol, amt in EMOTION_CUES[best_emotion].items():
                        releases[mol] = max(
                            releases.get(mol, 0.0), amt * EMOTION_SEMANTIC_DISCOUNT
                        )
            except Exception:  # noqa: BLE001 —— 语义失败不影响字面结果
                pass
        return releases, relaxes

    def _ensure_metadata(
        self,
        memory_id: str,
        text: str,
        tier: str,
        strength: float,
        *,
        links: Optional[List[Dict[str, Any]]] = None,
        role: str = "",
    ) -> None:
        """把 observe 侧的元数据补到一条记忆上（merge，不覆盖）。

        mnemosyne 自动从 observation 生成的 memory 只带 from_observation，
        tier / strength / links / role 都不会自己出现，必须显式补。
        合并而不是覆盖：from_observation 是它的溯源信息，不能丢。

        role 尤其重要：主动提起（proactive_recall）要「想起他说的」，
        而不是我自己对着他说的回复——分不清这两者，主动提起就会变成自恋。

        只在真的变了才 PATCH——同一句话被反复说时，不该反复写库。
        """
        if not memory_id:
            return
        # 读-改-写必须原子：两个 observe 同时改同一条记忆时，
        # 不加锁会互相覆盖（后写的把先写的字段吞掉）。
        with self._lock:
            try:
                cur = self.mnemosyne.get_memory(memory_id)
            except Exception as e:  # noqa: BLE001
                print(f"[links] metadata read failed: {e!r}", file=sys.stderr, flush=True)
                return
            merged = dict(cur.get("metadata") or {})
            changed = False
            if merged.get("tier") != tier:
                merged["tier"] = tier
                changed = True
            if merged.get("strength") != strength:
                merged["strength"] = strength
                changed = True
            if "salience" not in merged:
                merged["salience"] = float(self.persona.effects()["memory_salience"])
                changed = True
            if role and merged.get("role") != role:
                merged["role"] = role
                changed = True
            if links is None:
                links = self._build_links(text, self_id=memory_id)
            if links and merged.get("links") != links:
                merged["links"] = links
                changed = True
            if not changed:
                return
            try:
                self.mnemosyne.update_memory(memory_id, metadata=merged)
            except Exception as e:  # noqa: BLE001
                print(f"[links] metadata patch failed: {e!r}", file=sys.stderr, flush=True)

    def _build_links(self, text: str, *, self_id: str = "") -> List[Dict[str, Any]]:
        """给一条新记忆找邻居，形成引用图（只连向过去）。

        两类边：
          - related_to：内容像（语义）
          - temporal_neighbor：紧挨着发生（同一轮对话，文章里的 Thread）

        `self_id` 用来排除自连——时间邻居那一步尤其需要，否则新记忆会连到自己。

        为什么「Milvus 只当候选池，不用它的分数当阈值」：
        实测 hashing 向量对中文的区分度很差——完全无关的两句话能到 0.75+，
        真正同主题的只有 0.45。拿它的分数做阈值，等于随机连边。
        所以：向量层负责「把全库扫一遍、捞回一批候选」（召回要宽），
        真正的相似度用 bigram Jaccard 在候选里重算（排序要准）。
        这是典型的「便宜模型粗筛 + 贵判断精排」，和倒排索引 + 重排一个套路。

        为什么向量层挂了要能退：
        图是增强，不是前提。向量层不在也照样能靠权威源的模糊检索建一部分边。
        2026-11-12：语义这条腿改由「Milvus 或内存索引」共同提供——Milvus 里
        只剩几行的时候，光靠它建边等于只跟最近几条连，图会退化成一串孤岛。
        """
        cands: Dict[str, Tuple[str, float]] = {}  # id -> (content, 向量相似度)
        dead = {"invalidated", "superseded", "rejected", "decayed", "archived"}
        if self._milvus_usable():
            try:
                for h in self.vectors.search(text, limit=LINK_CANDIDATES):
                    if not h.memory_id or not h.content:
                        continue
                    if str(h.metadata.get("status", "")).lower() in dead:
                        continue
                    # VectorHit.score 是余弦**相似度**（0..1，相同文本为 1.0）
                    cands[h.memory_id] = (h.content, float(h.score))
            except Exception as e:  # noqa: BLE001
                print(f"[links] vector search failed: {e!r}", file=sys.stderr)
                cands = {}
        if not cands and self.vector_memory is not None:
            try:
                for h in self.vector_memory.search(text, limit=LINK_CANDIDATES):
                    mid = str(h.get("memory_id") or "")
                    if not mid or not h.get("content"):
                        continue
                    if str(h.get("status") or "").lower() in dead:
                        continue
                    cands[mid] = (str(h["content"]), float(h.get("score") or 0.0))
            except Exception as e:  # noqa: BLE001
                print(f"[links] memory index search failed: {e!r}", file=sys.stderr)
        if not cands:
            try:
                for m in self.mnemosyne.search(query=text, limit=12):
                    mid = str(m.get("id", ""))
                    if mid and str(m.get("status", "")).lower() not in dead:
                        cands[mid] = (str(m.get("content", "")), 0.0)
            except MnemosyneError:
                return []

        scored: List[Tuple[float, str, str]] = []
        for mid, (content, vec_sim) in cands.items():
            if not content or content == text:
                continue
            # 语义分（向量）与字面分（bigram）取大者：一个抓「换种说法」，
            # 一个抓「同一句话」，单用任何一个都会在另一类上漏边。
            sim = max(vec_sim, _similarity(text, content))
            if sim >= LINK_SIMILARITY:
                scored.append((sim, mid, content))
        scored.sort(key=lambda x: x[0], reverse=True)
        links: List[Dict[str, Any]] = [
            {
                "rel": RELATED_TO,
                "target_id": mid,
                "summary": _summarize(content),
                "similarity": round(sim, 3),
            }
            for sim, mid, content in scored[:MAX_LINKS]
        ]
        # 时间邻居：内容可能完全不像，但同一轮对话里的来回属于同一条线
        links.extend(self._temporal_links(text, self_id=self_id))
        return links

    def _temporal_links(self, text: str, *, self_id: str = "") -> List[Dict[str, Any]]:
        """时间邻居：最近的一条别的记忆，如果离得够近，就是同一轮对话的来回。

        语义边抓「内容像」，时间边抓「同一串对话」——你说一句、我回一句，
        这两句可能一个字都不像，但它们属于同一条线（Super-agent 里的 Thread）。
        这也是为什么单靠向量召回永远接不上「刚才聊到哪」。
        """
        if not text:
            return []
        try:
            rows = self.mnemosyne.search(query="", limit=12)
        except MnemosyneError:
            return []
        dead = {"invalidated", "superseded", "rejected", "decayed", "archived"}
        latest: Optional[Tuple[float, str, str]] = None
        for m in rows:
            mid = str(m.get("id", ""))
            if not mid or mid == self_id:
                continue
            if str(m.get("status", "")).lower() in dead:
                continue
            ts = _parse_iso(m.get("created_at"))
            if ts is None:
                continue
            if latest is None or ts > latest[0]:
                latest = (ts, mid, str(m.get("content", "")))
        if latest is None:
            return []
        gap = time.time() - latest[0]
        if gap < 0 or gap > TEMPORAL_LINK_WINDOW:
            return []
        return [
            {
                "rel": TEMPORAL_REL,
                "target_id": latest[1],
                "summary": _summarize(latest[2]),
                "via": "temporal",
            }
        ]

    def _links_of(self, memory_id: str) -> List[Dict[str, Any]]:
        """读一条记忆的 links（缺省返回空表，不抛错）。

        mnemosyne 的 search 结果不带 metadata，所以这里要单独 get 一次。
        调用方（retrieve）只在默认的「摘要模式」下取，且失败就当作没连边——
        链式召回是增强，不是前提。
        """
        if not memory_id:
            return []
        try:
            m = self.mnemosyne.get_memory(memory_id)
        except Exception:  # noqa: BLE001
            return []
        meta = m.get("metadata") or {}
        links = meta.get("links")
        return links if isinstance(links, list) else []

    def _find_existing_memory(self, text: str) -> Optional[Dict[str, Any]]:
        """找一条内容完全相同、且仍然活着的记忆（用于去重）。

        「活着」= 不是 invalidated / superseded / rejected / decayed。
        这样「同一句话不存两条」与「删过的能重新记」可以同时成立。
        """
        try:
            rows = self.mnemosyne.search(query=text, limit=8)
        except MnemosyneError:
            return None
        dead = {"invalidated", "superseded", "rejected", "decayed", "archived"}

        def _base(s: str) -> str:
            """去掉「对话片段：」前缀再比。

            不加这一步，同一句话会被记两遍：一遍原文、一遍回显 ——
            因为「对话片段：X」和「X」字面不相等。2026-11-11 三个体检员
            一起指出了这件事：50 条样本里十几条是这种双份。
            """
            s = str(s or "")
            return s[len("对话片段："):] if s.startswith("对话片段：") else s

        want = _base(text)
        for m in rows:
            if _base(m.get("content", "")) == want and str(m.get("status", "")).lower() not in dead:
                return m
        return None

    def _is_novel(self, text: str) -> bool:
        if not text:
            return False
        try:
            hits = self.mnemosyne.search(query=text, limit=1)
        except MnemosyneError:
            return True
        if not hits:
            return True
        first = str(hits[0].get("content", ""))
        return _similarity(first, text) < 0.8

    def snapshot(self) -> Dict[str, Any]:
        return {
            "persona": self.persona.snapshot(),
            "vector": self.vectors.stats() if (self._vector_up and self.vectors) else None,
            "vector_up": self._vector_up,
            "memory_index": self.vector_memory.stats() if self.vector_memory is not None else None,
        }

    # ------------------------------------------------------------------ 经验层
    def record_trait_evidence(
        self,
        trait: str,
        *,
        delta: float = 0.2,
        polarity: int = 1,
        evidence_id: Optional[str] = None,
        note: str = "",
        memory_id: str = "",
    ) -> Dict[str, Any]:
        """给经验层投一条证据。

        这是「经验塑造」层唯一的写入口——人格不会因为一次聊天就变，
        但会因为一条条独立证据很慢很慢地挪。规则在 persona.observe_trait：
        ≥3 条独立证据才开始动值，单步 ≤0.05，且向基础锚收缩。

        memory_id 是支撑这条证据的那条记忆（从 su_recall / su_remember 拿）。
        有它，证据就能点回原句，记忆被纠正时也能顺着它扣回去；
        没有它（老路径 / 一次性判断）就退化成原来的 ev:uuid。
        """
        if trait not in self.persona.shaped and trait not in self.persona.base:
            return {
                "ok": False,
                "error": f"未知特质：{trait}",
                "known_traits": sorted(self.persona.base),
            }
        mid = str(memory_id or "").strip()
        if evidence_id:
            ev = evidence_id
        elif mid:
            # 自造 UUID 永远指不回任何记忆；有 memory_id 时就用它当证据 id，
            # 这样「这条证据为什么在」和「哪条记忆」是同一个坐标。
            ev = f"mem:{mid}"
        else:
            ev = f"ev:{uuid.uuid4().hex[:12]}"
        result = self.persona.observe_trait(
            trait,
            delta=delta,
            evidence_id=ev,
            polarity=polarity,
            note=note,
            memory_id=mid,
        )
        # 慢通道：够久够多，才准往基础层沉一步（极窄，四道门见 persona.py）
        promotion = None
        try:
            promotion = self.persona.maybe_promote_to_base(trait)
        except Exception:  # noqa: BLE001 —— 慢通道出错不该挡住记账
            promotion = None
        # 证据即使没够门槛也要落盘：support/oppose 是「攒证据」的过程本身
        self.save_persona()
        cell = self.persona.shaped.get(trait)
        out = {
            "ok": True,
            "trait": trait,
            "evidence_id": ev,
            "memory_id": mid or None,
            "note": note,
            **result,
            "shaped": cell.to_dict() if cell else None,
        }
        if promotion:
            out["base_promotion"] = promotion
        # 影子模式：不管真改没改，都记一份晋升画像（只算，不动基础层）
        self._record_shadow_report()
        return out

    def trait_evidence(self, trait: str, *, limit: int = 10) -> Dict[str, Any]:
        """列出某个特质的证据，并把能查到的记忆摘要一起带出来。

        给「点开看原句」用：evidence_notes 是判断，memory_id 指向原句所在的那条记忆。
        查不到就留空——记忆可能被清理、服务可能暂时不通，证据本身仍然成立。
        """
        return self.persona.trait_evidence(trait, limit=limit)

    def _memory_summary(self, memory_id: str) -> str:
        """按 memory_id 取一句记忆摘要。只读，任何失败都返回空串——不编内容。"""
        mid = str(memory_id or "").strip()
        if not mid:
            return ""
        try:
            row = self.mnemosyne.get_memory(mid)
        except Exception:  # noqa: BLE001 —— 回查是锦上添花，不该报错
            return ""
        if not isinstance(row, dict):
            return ""
        # 正文可能在顶层，也可能裹在 memory 里（mnemosyne 的形状不完全稳定）
        text = _dig_memory_text(row)
        return _summarize(text) if text else ""

    def _record_shadow_report(self) -> None:
        """把「晋升画像」追加到影子账本（promotion_shadow.json）。

        只算不改：这份账本攒得越久，越能判断 20 证据 / 90 天这组门槛
        是「稳」还是「死」，而不冒人格漂移的风险（EVALUATION.md 局限 4）。
        节流：最多每小时一条快照。
        """
        now = time.time()
        if now - self._last_shadow_at < 3600:
            return
        if not self.persona_path:
            return
        try:
            report = self.persona.shadow_report()
            if not report:
                return
            self._last_shadow_at = now
            path = os.path.join(os.path.dirname(self.persona_path), "promotion_shadow.json")
            data: Dict[str, Any] = {"samples": []}
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f) or {"samples": []}
                except Exception:  # noqa: BLE001
                    data = {"samples": []}
            samples = data.get("samples", [])
            samples.append({"at": now, "traits": report})
            data["samples"] = samples[-500:]  # 只留最近 500 条，防膨胀
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 —— 影子账本失败不该挡证据记账
            pass

    # ------------------------------------------------------------------ 持久化
    def save_persona(self) -> None:
        """原子写：先写 .tmp 再 replace，避免崩溃时留下半个 JSON。"""
        if not self.persona_path:
            return
        try:
            d = os.path.dirname(self.persona_path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.persona_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.persona.to_state_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.persona_path)
        except Exception:  # noqa: BLE001 —— 持久化失败不应打断对话
            pass


# ------------------------------------------------------------------ helpers

_DEFAULT_BASE_PROFILE = {
    # 基础人格层：稳定锚，日常不写
    "warmth": {"value": 0.85, "note": "先接住情绪，再处理事情"},
    "directness": {"value": 0.45, "note": "不说教、不居高临下"},
    "curiosity": {"value": 0.75},
    "playfulness": {"value": 0.7},
    "boundary": {"value": 0.6, "note": "不替用户做重大决定"},
}


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat()


def _parse_iso(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        dt = _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _ts_of(value: Any) -> Optional[float]:
    """把 occurred_at 归一成 epoch。

    同一个字段在融合后的 items 里会有两种形态：权威源给 ISO 字符串，
    向量层给 float。这里统一，别让时间比较因为类型不同而静默失效。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return _parse_iso(value)


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _ro_connect(path: str) -> sqlite3.Connection:
    """只读打开权威源 SQLite（mode=ro 由 SQLite 自己强制，写不进去也拿不到写锁）。

    Windows 的盘符路径要有前导斜杠，否则 sqlite 会把「D:」当成主机名——
    这个坑踩过一次，记在这里。
    """
    db_path = os.path.abspath(path).replace("\\", "/")
    if not db_path.startswith("/"):
        db_path = "/" + db_path
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)


def _jieba_cut(text: str) -> List[str]:
    """jieba 整词分词；没装 jieba 时返回空列表（调用方退回 2-gram）。"""
    try:
        import jieba  # 延迟导入：无 jieba 的环境不该因此崩
    except Exception:  # noqa: BLE001
        return []
    try:
        return list(jieba.lcut(text))
    except Exception:  # noqa: BLE001
        return []


def _query_units(query: str) -> List[str]:
    """把 query 切成「用来判字面命中」的单元。

    - 中文：用 jieba 按**整词**切（「今天天气怎么样」→ 今天天气 / 怎么样），
      单字和纯标点丢掉（「的」「了」「我」几乎每条记忆都有，拿它当命中等于
      没有命中）。整词切分天然不会再产生「天天」这种跨词 bigram 假词。
    - ASCII：按非字母数字切分，只留长度 ≥2 的 token（`zzqx-9f3a` → zzqx / 9f3a）。
      单个字母（a / I）同样是噪声。
    - 没装 jieba 时退回旧的 2-gram（降级不崩，行为与从前一致）。
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    units: List[str] = []

    # ASCII token：字母/数字连续段，长度 >= 2
    buf: List[str] = []
    for ch in q:
        if ch.isascii() and ch.isalnum():
            buf.append(ch)
            continue
        if len(buf) >= 2:
            units.append("".join(buf))
        buf = []
    if len(buf) >= 2:
        units.append("".join(buf))

    # 中文：jieba 整词分词，只留纯中文词（长度 ≥2）；单字和标点丢弃。
    # 注意只取每个词的「纯中文部分」：ASCII 已经由上面那段单独切过，这里
    # 不重复、也不让 jieba 切出的英文 token 混进中文单元里。
    words = _jieba_cut(q)
    if words:
        for w in words:
            core = "".join(ch for ch in w if "\u4e00" <= ch <= "\u9fff")
            if len(core) >= 2:
                units.append(core)
    else:
        # 无 jieba：退回旧的 2-gram（单字 run 退回单字）
        run: List[str] = []
        for ch in q + " ":
            if "\u4e00" <= ch <= "\u9fff":
                run.append(ch)
                continue
            if len(run) >= 2:
                for i in range(len(run) - 1):
                    units.append(run[i] + run[i + 1])
            elif run:
                units.append(run[0])
            run = []

    # 去重但保序：同一个词出现两次不该让分母变大
    seen = set()
    out: List[str] = []
    for u in units:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _build_df(path: Optional[str]) -> Dict[str, int]:
    """从权威源 SQLite 全量统计文档频率（只算 active/candidate）。

    df = 每个单元出现在多少条记忆的正文里（出现次数不叠加，一条记忆最多 +1）。
    单元切分复用 _query_units（中文 jieba 整词、ASCII 按非字母数字切 token）。
    读不到 / 失败就当空表——稀有词只是排序启发式，坏了不该拖垮兜底。
    """
    df: Dict[str, int] = {}
    if not path or not os.path.exists(path):
        return df
    con = None
    try:
        con = _ro_connect(path)
        marks = ",".join("?" * len(LITERAL_LIVE_STATES))
        rows = con.execute(
            f"SELECT content FROM memories WHERE status IN ({marks})",
            LITERAL_LIVE_STATES,
        )
        for (content,) in rows:
            for u in set(_query_units(str(content or ""))):
                df[u] = df.get(u, 0) + 1
    except Exception:  # noqa: BLE001
        df = {}
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass
    return df


def _pick_rare_terms(query: str, df: Dict[str, int], limit: int = RARE_TERM_LIMIT) -> List[str]:
    """按 df 升序（稀有优先）挑 query 里最稀有的 1-2 个单元。

    只挑在语料里出现过的（df >= 1）：df=0 的单元不可能命中任何正文，
    挑它只会浪费一个稀有词名额。高频词（「我们」「这个」）df 大，自然被挤到后面。
    """
    units = _query_units(query)
    if not units or not df:
        return []
    known = [(u, int(df.get(u, 0))) for u in units if int(df.get(u, 0)) >= 1]
    known.sort(key=lambda item: (item[1], item[0]))
    return [u for u, _ in known[: max(1, min(int(limit or 1), len(known)))]]


def _is_cjk_unit(unit: str) -> bool:
    """这个单元是不是纯中文（整词或单字）。ASCII token 不算。"""
    return bool(unit) and all("\u4e00" <= ch <= "\u9fff" for ch in unit)


def _unit_weight(unit: str, df: Dict[str, int]) -> float:
    """单个 query 单元在字面命中里的权重。

    - df=0（语料里从没出现过）：不可能命中任何正文，权重 0，直接排除——
      否则「zzqx」这种从未见过的单元会白占分母，把真命中稀释掉。
    - ASCII token（milvus / sqlite / zzqx）：精确命中就是真命中，不给 df 打折，
      满权重 1.0。中文词才有「撞上无关记忆」的假命中问题。
    - 中文词：按文档频率加权，高频（「完全」「存在」「我们」「这个」）
      权重压低，低频/稀有（「溯忆」「连接池」）权重抬高。
    """
    d = int(df.get(unit) or 0)
    if d <= 0:
        return 0.0
    if not _is_cjk_unit(unit):
        return 1.0
    return 1.0 / (1.0 + float(d))


def _literal_relevance(query: str, content: str, df: Optional[Dict[str, int]] = None) -> float:
    """query 的词有多少比例出现在 content 里（0~1）。

    mnemosyne 的 score 不反映相关性（固定 0.45），所以这条用来做二次过滤：
    一个字面都不沾的「权威命中」，不该拿权威基线分。
    - 中文：按 jieba 整词切（避免单字噪声），算覆盖比例
    - ASCII：按非字母数字切 token（长度 ≥2），算命中比例
    - 两者混在一起算：query 里既有中文又有英文时，各单元权重相同

    df 给定时按文档频率加权（见 _unit_weight）：中文词高频的权重压低、
    稀有词权重抬高；ASCII token 保持满权重。这样「中文全是常见词 + 一个随机串」
    的噪声 query 会被稀有/未见的单元把分母撑大，覆盖比例掉到 attested 阈值以下，
    而「Milvus 连接池」这种 ASCII + 稀有词的精确命中不受影响。df 没给
    （未建表 / 降级）就退回原来的等权覆盖比例，不崩。
    """
    units = _query_units(query)
    if not units:
        return 0.0
    hay = (content or "").lower()
    if not hay:
        return 0.0
    if not df:
        # 降级：没有 df 表时退回等权覆盖比例（保持旧行为，兜底不该崩）
        hit = sum(1 for u in units if u in hay)
        return hit / len(units)
    weights = [_unit_weight(u, df) for u in units]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    hit = sum(w for u, w in zip(units, weights) if u in hay)
    return hit / total


def _has_semantic_embedder(*sources: Optional[Any]) -> bool:
    """这几路里有没有真句向量（hashing 的「相似度」没有语义意义）。

    Milvus 那路和内存索引那路各带一个 embedder；两者都可能单独存在，
    所以要看全部，不能只看 Milvus。
    """
    for src in sources:
        emb = getattr(src, "embedder", None)
        if emb is not None and bool(getattr(emb, "is_semantic", False)):
            return True
    return False


def _is_self_referential(text: str, idx: int) -> bool:
    """情绪词是在描述「说话人自己」，还是只是被提到 / 在问别人 / 在讲概念。

    真实翻车：用户问「现在你为什么悲伤焦虑呢」，系统听见「焦虑」就给自己灌了
    一管皮质醇——他越问它为什么慌，它越慌。所以要先判断这句是不是在说它自己。

    另一类更隐蔽的翻车是「元讨论」：我在解释「焦虑的机制」「调阈值」，
    甚至念「焦虑 +94」这种报告数字——那全是在**描述**，不是在**体验**。
    所以窗口从 14 字扩到 20 字（前后各 20，够覆盖「我们在调 X 的参数阈值」
    这种词在后面的句式），并加了一张元讨论词表 + 报数判断。

    注意 META_TALK_WORDS 里**没有「代码」**：「代码写不完，我好焦虑」
    是真的焦虑，不能被误挡。
    """
    # 谁在说 —— 只看紧邻的 6 个字，远一点的「你」不改变这句的主语
    tail = text[max(0, idx - 6) : idx]
    if any(p in tail for p in ("你", "您", "他", "她", "它")):
        return False
    # 疑问句式 → 在问，不是在表达（只在紧邻处判定：
    # 「我刚才是不是说错了」是自我反思，不是在问他，不能一起挡掉）
    if any(q in tail for q in ("为什么", "是不是", "怎么", "为啥", "因何")):
        return False

    # 讲概念 —— 元讨论词、报数，都不是我/他此刻的感受
    window = text[max(0, idx - 20) : idx + 20]
    if any(w in window for w in META_TALK_WORDS):
        return False
    if any(ch.isdigit() for ch in window) and ("+" in window or "%" in window):
        return False
    return True


def _negated(text: str, idx: int) -> bool:
    """线索词前面紧邻否定词时判定为否定（「不开心」「不累」「没睡好」）。"""
    window = text[max(0, idx - 2) : idx]
    return any(n in window for n in NEGATIONS)


def _looks_like_token(text: str) -> bool:
    """孤立的无空格 ASCII 字母数字串（id / 哈希碎片 / 误触）。

    它可能有「新颖度」，但没有任何可回忆的内容价值；
    让这种东西进长期记忆，等于往人脑子里塞噪声。
    """
    if not text or len(text) >= 16:
        return False
    if not text.isascii():
        return False
    if any(ch.isspace() for ch in text):
        return False
    return all(ch.isalnum() or ch in "_-." for ch in text)


def _looks_garbled(text: str) -> bool:
    """粗判乱码：替换符（U+FFFD）、控制字符、或整串只剩问号。

    这条规则来自一次真实事故：PowerShell 的 Invoke-WebRequest 默认按 UTF-16
    发送 body，服务端收到 `????` 并把它当正常记忆存了下来——乱码一旦进入
    长期记忆，只能靠用户手动删除，而且会污染后续所有检索。
    """
    if not text:
        return False
    n = len(text)
    bad = text.count("\ufffd") + sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    if n >= 4 and bad / n > 0.2:
        return True
    # 全是问号/感叹号等符号（"?? ??", "????"）
    symbolic = sum(1 for ch in text if ch in "?？!")
    return n >= 4 and symbolic / n > 0.5


def _similarity(a: str, b: str) -> float:
    """粗略字符 bigram Jaccard，够做「是不是同一件事」的初筛。"""
    if not a or not b:
        return 0.0
    sa = {a[i : i + 2] for i in range(max(0, len(a) - 1))}
    sb = {b[i : i + 2] for i in range(max(0, len(b) - 1))}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _dig_memory_text(row: Any) -> str:
    """从一条记忆行里挖出正文。形状不稳定，按常见字段逐层找。"""
    if not isinstance(row, dict):
        return ""
    for key in ("content", "text", "summary"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    inner = row.get("memory")
    if isinstance(inner, dict):
        return _dig_memory_text(inner)
    return ""


def _summarize(text: str, max_chars: int = SUMMARY_CHARS) -> str:
    """把一条记忆压成一句话摘要（Inline & Spill 里的 Inline 部分）。

    不调模型：这里要的是「低延迟、确定性、可解释」，不是「最漂亮的摘要」。
    优先取第一句；第一句太长就断在逗号/空格，不硬切词——
    「重新认出你」靠的是那半句话，不是整段。

    多行的答题式回复（星枢的真实回复经常是分段的）会被压成第一句，
    这恰好是最该被记住的那半句；细节由 expand() 按需补。
    """
    t = " ".join((text or "").split())
    if not t:
        return ""
    for sep in ("。", "！", "？", "!", "?", "；", ";"):
        i = t.find(sep)
        if 0 <= i < max_chars * 2:
            t = t[: i + 1]
            break
    if len(t) > max_chars:
        cut = max((t.rfind(sep, 0, max_chars) for sep in ("，", "、", ",", " ")), default=-1)
        t = (t[:cut] if cut > max_chars // 3 else t[: max_chars - 1]) + "…"
    return t


def _kmeans(vecs: "np.ndarray", k: int, iters: int = 15, seed: int = 42):
    """最简 kmeans（numpy 手写）：返回 (labels, centroids)。

    不引 sklearn 只为这点计算量——原型期依赖越少越好，而且这里的向量
    已经 L2 归一化过，平方距离就是 2-2·cos，够用。
    """
    n = int(vecs.shape[0])
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    centroids = vecs[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(iters):
        d = ((vecs[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            if bool((labels == i).any()):
                centroids[i] = vecs[labels == i].mean(0)
    return labels, centroids

