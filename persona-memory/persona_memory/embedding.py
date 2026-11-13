# -*- coding: utf-8 -*-
"""文本向量化。

默认提供 `HashingEmbedder`：零依赖、确定性、可离线，用字符 n-gram 做 feature hashing。
它的定位是**接口占位**——保证 Milvus 链路能端到端跑通、可复现、可测试；
生产环境应替换成真实句向量模型（adapter 已是接口，见 `Embedder`）。

为什么不用云端 embedding：原型必须在断网、无 API key 时也能起得来；
否则「记忆服务」会在最需要它的深夜停摆。
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from typing import List, Optional, Protocol, Sequence

DIM_DEFAULT = 512


class Embedder(Protocol):
    dim: int

    def encode(self, text: str) -> List[float]: ...


_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _features(text: str) -> List[str]:
    """字符 n-gram + 词元。中文按字组合，英文按词，数字/符号保留。"""
    t = (text or "").strip().lower()
    if not t:
        return []
    feats: List[str] = []
    # 英文/数字词
    feats.extend(_WORD_RE.findall(t))
    # 中文：单字 + 相邻 bigram + trigram（捕捉「记忆」「人格」这类组合）
    cjk_chars = _CJK_RE.findall(t)
    feats.extend(cjk_chars)
    for n in (2, 3):
        for i in range(len(cjk_chars) - n + 1):
            feats.append("".join(cjk_chars[i : i + n]))
    # 混合文本：整体字符 bigram
    compact = re.sub(r"\s+", "", t)
    for i in range(len(compact) - 1):
        feats.append(compact[i : i + 2])
    return feats


def _hash_pair(token: str, dim: int) -> "tuple[int, float]":
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "little", signed=False)
    idx = v % dim
    sign = 1.0 if (v >> 17) & 1 else -1.0
    return idx, sign


class HashingEmbedder:
    """确定性 feature hashing 向量。同一文本永远得到同一向量。"""

    def __init__(self, dim: int = DIM_DEFAULT) -> None:
        self.dim = dim
        # 字面 hashing：只认字面重叠，不懂语义，不能做语义情绪线索
        self.is_semantic = False

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        feats = _features(text)
        if not feats:
            return vec
        for tok in feats:
            idx, sign = _hash_pair(tok, self.dim)
            vec[idx] += sign
        # L2 归一化（COSINE 下等价于只保留方向）
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


# ------------------------------------------------------------------ 真句向量（可选）
DEFAULT_SEMANTIC_MODEL = "BAAI/bge-small-zh-v1.5"


class FastEmbedEmbedder:
    """真句向量：fastembed + ONNX Runtime，**不依赖 torch**。

    为什么是它：
    - 只要几十 MB 的 onnxruntime，不需要 2GB 的 torch
    - CPU 推理够快（单条 20-50ms），服务端能接受
    - 模型可以从 HF 镜像拉：设 `HF_ENDPOINT=https://hf-mirror.com`

    和 HashingEmbedder 的根本区别：它真的懂「累」和「deadline 压得喘不过气」
    是同一件事，而 hashing 只认字面重叠——这就是之前「问焦虑找不到那条」的根因。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        cache_dir: Optional[str] = None,
        threads: int = 2,
    ) -> None:
        from fastembed import TextEmbedding  # 延迟导入：没装也能跑纯逻辑测试

        self.model_name = model_name
        # 真句向量：能理解「累」和「喘不过气」是同一件事，可做语义情绪线索
        self.is_semantic = True
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)
        probe = list(self._model.embed(["维度探测"]))
        self.dim = len(probe[0]) if probe else DIM_DEFAULT

    def encode(self, text: str) -> List[float]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        items = [t if (t and str(t).strip()) else " " for t in texts]
        return [[float(x) for x in vec.tolist()] for vec in self._model.embed(items)]


def create_embedder(kind: str = "hash", **kwargs: object):
    """工厂：默认 hash（零依赖、永远可用）；fastembed 不可用时自动退回 hash。

    记忆服务最不能接受的就是「因为模型没装好所以起不来」——
    所以这里永远有退路，只是退路的质量低一点。
    """
    kind = (kind or "hash").lower()
    if kind in ("fastembed", "semantic", "bge"):
        try:
            return FastEmbedEmbedder(**kwargs)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            print(f"[embedder] fastembed 不可用，退回 hash：{e}", file=sys.stderr, flush=True)
    return HashingEmbedder()
