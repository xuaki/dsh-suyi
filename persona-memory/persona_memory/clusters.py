# -*- coding: utf-8 -*-
"""记忆分簇：不同类的经验，配不同的找回方式。

学自 Routed Continual Learning（Meta FAIR, 2026）：
    把经验都压成一种表示不是最优的——代码类的该用仓库索引，
    情感类的该看重时间，偏好的该反复确认。它用 LLM 路由器选簇；
    我们没有那么多算力，就做最小版：
      1. 写入时按规则打簇标签（存在 metadata.cluster）
      2. 回忆时先判断「这是哪类问题」，命中同簇的记忆往前排

为什么不是「直接语义检索就够」：
    你已经问过一百遍「怎么改代码」，和「你难过吗」，
    在余弦相似度眼里没有类别差别。但在人脑子里，这两件事
    根本不是从同一个抽屉里拿出来的。
"""

from __future__ import annotations

from typing import Dict, List

# 四个簇 + 兜底
CLUSTERS = ("tech", "emotion", "preference", "project")
DEFAULT_CLUSTER = "default"

CLUSTER_LABELS: Dict[str, str] = {
    "tech": "技术",
    "emotion": "情感",
    "preference": "偏好",
    "project": "项目",
    "default": "其它",
}

# 关键词刻意只收「强信号」：宁可不分类，也不要分错。
# 分错比不分更糟——它会把不相干的记忆推到前面。
CLUSTER_KEYWORDS: Dict[str, tuple] = {
    "tech": (
        "代码", "接口", "插件", "服务", "报错", "部署", "实现", "函数", "数据库",
        "端口", "配置", "脚本", "编译", "依赖", "进程", "重启", "索引", "向量",
        "内存", "日志", "超时", "patch", "bug", "api", "json", "python", "node",
    ),
    "emotion": (
        "难过", "心疼", "焦虑", "害怕", "孤独", "委屈", "生气", "想哭", "抱抱",
        "安慰", "累", "疲惫", "想念", "舍不得", "开心", "高兴", "幸福",
        "心情", "情绪", "感觉怎么样",
    ),
    "preference": (
        "喜欢", "讨厌", "迷上", "习惯", "最爱", "偏爱", "喜好",
    ),
    "project": (
        "上线", "完成", "做完", "进度", "计划", "方案", "里程碑", "验收",
        "需求", "排期", "发布", "迭代",
    ),
}

# 命中同簇时给多少加成。1.2 是实测下来「能感觉到、但压不过语义相关」的档位。
CLUSTER_BOOST = 1.2


def classify(text: str) -> str:
    """给一段文字打簇标签。分不出来就返回 default——不硬猜。"""
    s = str(text or "")
    if not s:
        return DEFAULT_CLUSTER
    scores = {c: sum(1 for w in CLUSTER_KEYWORDS[c] if w in s) for c in CLUSTERS}
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else DEFAULT_CLUSTER


def detect_query_cluster(query: str) -> str:
    """判断「这次问的是哪一类问题」。

    和 classify 同一套规则，但语义不同：classify 是「这条记忆属于哪类」，
    这里是「这次提问指向哪类」。分开写是因为以后它们大概率会长得不一样
    （提问可能带语气词，记忆不会）。
    """
    return classify(query)


# ------------------------------------------------------------------ 语义级联
# 规则分不出（default）时，用句向量做语义路由——这是 Routed CL「LLM 路由」
# 的降级版：没有 LLM，就用 embedding 判断「这段话和哪类最像」。
# 每簇一段描述，代表它的内容特征。

CLUSTER_DESCRIPTIONS: Dict[str, str] = {
    "tech": "技术、代码、接口、服务、部署、报错、配置、数据库、脚本、进程、日志",
    "emotion": "情绪、感受、心情、难过、开心、焦虑、累、想念、安慰、抱抱",
    "preference": "喜欢、讨厌、偏好、习惯、爱好、口味、最爱",
    "project": "项目进展、计划、上线、完成、进度、排期、需求、验收",
}

# 语义路由的相似度门槛：够近才归类，否则还是 default——不硬猜。
CLUSTER_SEMANTIC_THRESHOLD = 0.5


def _cosine(a: List[float], b: List[float]) -> float:
    """两向量余弦。embedder 已 L2 归一化，等价于点积。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def classify_semantic(text: str, embedder: object) -> str:
    """语义路由：算文本和每簇描述的距离，取最近（够近才算）。

    embedder 必须是有 is_semantic 的真句向量，否则直接返回 default。
    """
    if not getattr(embedder, "is_semantic", False):
        return DEFAULT_CLUSTER
    try:
        q = embedder.encode(str(text or ""))  # type: ignore[attr-defined]
        best = DEFAULT_CLUSTER
        best_sim = 0.0
        for c, desc in CLUSTER_DESCRIPTIONS.items():
            s = _cosine(q, embedder.encode(desc))  # type: ignore[attr-defined]
            if s > best_sim:
                best_sim = s
                best = c
        return best if best_sim >= CLUSTER_SEMANTIC_THRESHOLD else DEFAULT_CLUSTER
    except Exception:  # noqa: BLE001
        return DEFAULT_CLUSTER


def classify_cascade(text: str, embedder: object = None) -> str:
    """级联路由：规则先粗分（快、免费、覆盖大多数）；分不出再语义精分。

    和「宁可不分类也不分错」一致：两级都判不出来，就老老实实 default，
    退化为无加成的原行为。
    """
    rule = classify(text)
    if rule != DEFAULT_CLUSTER:
        return rule
    if embedder is not None:
        return classify_semantic(text, embedder)
    return DEFAULT_CLUSTER
