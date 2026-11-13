# -*- coding: utf-8 -*-
"""服务入口。

    python -m persona_memory.main --port 8770

默认连本机 mnemosyne（127.0.0.1:8765）与 Milvus（127.0.0.1:19530）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .api import create_app
from .embedding import DEFAULT_SEMANTIC_MODEL, HashingEmbedder, create_embedder
from .milvus_store import MilvusVectorStore
from .mnemosyne_client import MnemosyneClient
from .service import PersonaMemoryService


def _make_embedder():
    """挑向量器：默认试 fastembed（真语义），不可用就退回 hash（永远可用）。

    两者维度碰巧都是 512，但**语义空间不同，绝不能混用一个 collection**，
    所以 collection 名跟着向量器走（hash -> v1，semantic -> v1s）。

    环境变量：
        PERSONA_EMBEDDER      auto（默认）| hash | fastembed
        PERSONA_EMBED_MODEL   默认 BAAI/bge-small-zh-v1.5
        PERSONA_MODEL_DIR     模型缓存目录（默认用 fastembed 自己的缓存）
    """
    kind = (os.environ.get("PERSONA_EMBEDDER") or "auto").lower()
    if kind in ("hash", "hashing"):
        return HashingEmbedder(), "hash"
    try:
        model = os.environ.get("PERSONA_EMBED_MODEL") or DEFAULT_SEMANTIC_MODEL
        emb = create_embedder(
            "fastembed",
            model_name=model,
            cache_dir=os.environ.get("PERSONA_MODEL_DIR") or None,
        )
        print(f"[main] 向量器：fastembed / {model}（dim={emb.dim}）", file=sys.stderr, flush=True)
        return emb, "fastembed"
    except Exception as e:  # noqa: BLE001
        print(f"[main] fastembed 不可用，退回 hash：{e}", file=sys.stderr, flush=True)
        return HashingEmbedder(), "hash"


def collection_for(kind: str) -> str:
    """向量集合名跟着向量器走：换了模型就必须重建，不能混。"""
    override = os.environ.get("MILVUS_COLLECTION")
    if override:
        return override
    return "persona_memory_v1" if kind == "hash" else "persona_memory_semantic_v1"


def collection_from_env() -> str:
    """不加载模型，只看环境变量推断集合名——运维脚本用这个，省 9 秒加载时间。"""
    override = os.environ.get("MILVUS_COLLECTION")
    if override:
        return override
    kind = (os.environ.get("PERSONA_EMBEDDER") or "auto").lower()
    return "persona_memory_v1" if kind in ("hash", "hashing") else "persona_memory_semantic_v1"


def build_service() -> PersonaMemoryService:
    mn = MnemosyneClient(
        base_url=os.environ.get("MNEMOSYNE_URL", "http://127.0.0.1:8765"),
        token=os.environ.get("MNEMOSYNE_TOKEN", "dev-token"),
    )
    # 第三级降级用的权威源文件：Milvus 和 mnemosyne 都挂时直接读它做字面扫描。
    # 默认空字符串 = 不启用；只有真的存在这个路径才该配。
    literal_db = os.environ.get("MNEMOSYNE_DB", "")
    embedder, kind = _make_embedder()
    vectors = MilvusVectorStore(
        uri=os.environ.get("MILVUS_URI", "http://127.0.0.1:19530"),
        collection=collection_for(kind),
        embedder=embedder,
    )
    data_dir = Path(
        os.environ.get("PERSONA_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    # 内存向量索引（2026-11-12）：Milvus 的替补。
    #   PERSONA_VECTOR_INDEX   npz 路径；默认 data/vector_index.npz
    #                          设成 off/none/空 = 关掉（只剩 Milvus + 字面兜底）
    # 第一次检索时懒加载：读得到 npz 就直接用，读不到就在后台重建（1-2 分钟）。
    index_path = os.environ.get("PERSONA_VECTOR_INDEX")
    if index_path is None:
        index_path = str(data_dir / "vector_index.npz")
    if str(index_path).strip().lower() in ("", "off", "none", "0", "disabled"):
        index_path = ""
    return PersonaMemoryService(
        mnemosyne=mn,
        vectors=vectors,
        persona_path=str(data_dir / "persona_state.json"),
        literal_db_path=literal_db or None,
        memory_index_path=index_path or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    import uvicorn

    app = create_app(build_service())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
