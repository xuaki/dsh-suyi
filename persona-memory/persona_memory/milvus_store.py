# -*- coding: utf-8 -*-
"""Milvus 向量层：补上 mnemosyne 的 `vector_index: enable: false`。

实测发现（Milvus 3.0-20261030 + pymilvus 3.0.2）：
- 简化的 `create_collection(dimension=..., metric_type=...)` 会**静默**建出
  搜不到的集合（search 返回空、row_count 为 0）；
- 必须用 `create_schema()` + `prepare_index_params()` 显式建索引才正常。
本模块只走显式路径，并在建集合后强制 flush + load。

设计约束：
- Milvus 只是**索引**，不是权威源。权威源是 mnemosyne 的 SQLite。
- 写入失败不阻塞主流程，记入 outbox（待重建），检索时以 SQLite 结果为准。
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .embedding import Embedder, HashingEmbedder

COLLECTION_DEFAULT = "persona_memory_v1"


@dataclass
class VectorHit:
    memory_id: str
    content: str
    # Milvus 在 COSINE 度量下返回的是**相似度**（越大越像；相同文本为 1.0），
    # 不是距离。当成距离用会让所有融合打分反向——这个坑踩过一次，记在这里。
    score: float
    metadata: Dict[str, Any]


def _truncate_bytes(text: str, max_bytes: int = 4000) -> str:
    """按 UTF-8 **字节**截断。

    Milvus 的 VARCHAR max_length 是字节数，而中文一个字占三字节——
    按字符截断到 4000，中文长文本会变成 12000 字节，直接溢出 4096
    （2026-11-10 重建索引时实锤：一条 4227 字的记忆死活进不去）。

    注意：向量是用**完整文本**算的，这里截断的只是随向量一起存的缓存字段；
    真要看全文走 expand，从 SQLite 读。
    """
    raw = str(text or "")
    b = raw.encode("utf-8")
    if len(b) <= max_bytes:
        return raw
    return b[:max_bytes].decode("utf-8", errors="ignore")


def _looks_like_local_path(uri: str) -> bool:
    """这个 uri 是 milvus-lite 的本地文件，还是远程服务地址。

    本地文件（`data/milvus_lite.db`、`D:\\...\\x.db`）必须走位置参数；
    远程（`http://127.0.0.1:19530`）走 `uri=` 关键字。
    """
    s = str(uri or "").strip()
    if not s:
        return False
    if s.lower().startswith(("http://", "https://", "tcp://", "unix:")):
        return False
    return s.endswith(".db") or ("\\" in s) or ("/" in s)


class MilvusVectorStore:
    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        collection: str = COLLECTION_DEFAULT,
        embedder: Optional[Embedder] = None,
        token: str = "",
    ) -> None:
        # 默认地址走环境变量：Milvus 跑在 WSL 里时地址会变（localhost 转发常断），
        # 由 bootstrap 探测后经 MILVUS_URI 传进来（2026-11-11 加）。
        self.uri = uri or os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
        self.collection = collection
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.token = token
        self._client = None
        self._inserted = 0

    # ---------------- 连接与建表 ----------------
    def _get(self):
        if self._client is None:
            from pymilvus import MilvusClient  # 延迟导入，纯逻辑测试不需要它

            # milvus-lite 的本地文件**必须用位置参数**传。
            # 新版 pymilvus 对 `uri="D:\...\x.db"` 会报
            #   Illegal uri: [D:\...\x.db], expected form 'http[s]://...'
            # 而 `MilvusClient("D:\...\x.db")` 是认的（2026-11-12 踩到）。
            if _looks_like_local_path(self.uri):
                import sys as _sys

                print(
                    f"[vector] 用 milvus-lite 本地文件：{self.uri!r}",
                    file=_sys.stderr,
                    flush=True,
                )
                if self.token:
                    self._client = MilvusClient(self.uri, token=self.token)
                else:
                    self._client = MilvusClient(self.uri)
            else:
                import sys as _sys

                print(
                    f"[vector] 用远程 Milvus：{self.uri!r}",
                    file=_sys.stderr,
                    flush=True,
                )
                kwargs: Dict[str, Any] = {"uri": self.uri}
                if self.token:
                    kwargs["token"] = self.token
                self._client = MilvusClient(**kwargs)
        return self._client

    def connect(self) -> bool:
        get = getattr(self._get(), "list_collections", None)
        if get is None:  # pragma: no cover
            return False
        get()
        return True

    def ensure_collection(self) -> None:
        from pymilvus import DataType

        client = self._get()
        if self.collection in client.list_collections():
            client.load_collection(self.collection)
            return

        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embedder.dim)
        schema.add_field("memory_id", DataType.VARCHAR, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=4096)
        schema.add_field("mtype", DataType.VARCHAR, max_length=32)
        schema.add_field("status", DataType.VARCHAR, max_length=32)
        schema.add_field("occurred_at", DataType.DOUBLE)
        schema.add_field("source_kind", DataType.VARCHAR, max_length=48)
        schema.add_field("salience", DataType.DOUBLE)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )
        client.create_collection(
            collection_name=self.collection, schema=schema, index_params=index_params
        )
        client.load_collection(self.collection)

    # ---------------- 写 ----------------
    def upsert(
        self,
        *,
        memory_id: str,
        content: str,
        mtype: str = "semantic",
        status: str = "candidate",
        occurred_at: Optional[float] = None,
        source_kind: str = "dialog",
        salience: float = 0.5,
        flush: bool = False,
    ) -> int:
        """插入一条向量。

        默认**不** flush：Milvus 一次 flush 是秒级，逐条 flush 会让批量写入退化到不可用。
        调用方在批量写完（或检索前）显式调用 `flush()`。
        """
        client = self._get()
        vec = self.embedder.encode(content)
        client.insert(
            collection_name=self.collection,
            data=[
                {
                    "vector": vec,
                    "memory_id": memory_id,
                    "content": _truncate_bytes(content, 4000),
                    "mtype": mtype,
                    "status": status,
                    "occurred_at": float(occurred_at or time.time()),
                    "source_kind": source_kind,
                    "salience": float(salience),
                }
            ],
        )
        self._inserted += 1
        if flush:
            client.flush(self.collection)
        # Milvus 没有主键级 upsert 语义（auto_id）；用 delete+insert 语义由调用方保证
        return 1

    def flush(self) -> None:
        """让已写入的数据对检索可见。批量写入后调用一次即可。"""
        self._get().flush(self.collection)

    def delete(self, memory_id: str, *, flush: bool = False) -> None:
        client = self._get()
        client.delete(collection_name=self.collection, filter=f'memory_id == "{memory_id}"')
        if flush:
            client.flush(self.collection)

    def _count(self) -> int:
        try:
            return int(self._get().get_collection_stats(self.collection).get("row_count", 0))
        except Exception:
            return 0

    # ---------------- 读 ----------------
    def search(self, query: str, *, limit: int = 10) -> List[VectorHit]:
        client = self._get()
        qv = self.embedder.encode(query)
        if not any(qv):
            return []
        res = client.search(
            collection_name=self.collection,
            data=[qv],
            limit=limit,
            output_fields=["memory_id", "content", "mtype", "status", "occurred_at", "source_kind", "salience"],
        )
        hits: List[VectorHit] = []
        for row in res:
            for h in row:
                ent = h.get("entity", {}) or {}
                hits.append(
                    VectorHit(
                        memory_id=str(ent.get("memory_id", "")),
                        content=str(ent.get("content", "")),
                        score=float(h.get("distance", 0.0)),
                        metadata={
                            "mtype": ent.get("mtype"),
                            "status": ent.get("status"),
                            "occurred_at": ent.get("occurred_at"),
                            "source_kind": ent.get("source_kind"),
                            "salience": ent.get("salience"),
                        },
                    )
                )
        return hits

    def stats(self) -> Dict[str, Any]:
        return {"collection": self.collection, "rows": self._count(), "dim": self.embedder.dim}
