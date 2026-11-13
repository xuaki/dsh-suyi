# -*- coding: utf-8 -*-
"""内存向量索引：把「语义召回」从 Milvus 手里收回来。

为什么不用 Milvus 了（2026-11-12）：
它第三次掉链子——内部 mixcoord 组件挂掉，报 `node not found`，建不起来也
重启不了（docker 没权限）。而这套记忆的量级是「几百条，未来几千条」：
643 × 512 维在内存里算余弦是**毫秒级**，根本不需要一个独立的向量数据库、
一个容器、一个要单独运维的进程。

所以这一层就是 numpy：
  - 建：把所有记忆 embed 一遍，存成 (n, 512) 的 float32 矩阵；
  - 查：query embed 一次，`matrix @ q` 一次矩阵乘，取 top-k；
  - 存：`data/vector_index.npz`（ids + matrix + 正文 + 状态 + meta），
        重启直接加载，**不重新 embed**（重建一次要一两分钟，只在语料变了才做）。

设计约束（和 milvus_store.py 一样）：
  - 索引不是权威源。权威源永远是 mnemosyne 的 SQLite。
  - 任何失败都必须被吞掉并降级：记忆服务不能因为「向量建不起来」就起不来。
  - 懒加载：进程启动时**不**阻塞。第一次 search() 时：能读 npz 就直接读，
    读不到（首次部署 / 语料换代 / 文件损坏）就在后台线程里重建，
    这一轮先返回空——下一次召回就有语义了。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# 一批 embed 多少条：fastembed 的 ONNX 推理在 32 条一批时吞吐最好，
# 再大只是多吃内存，不会更快。
BUILD_BATCH = 32
# 每编码这么多条打一次进度：重建 600+ 条要一两分钟，没有任何输出会让人
# 以为服务卡死了（这一条是给运维看的，不是给日志系统看的）。
PROGRESS_EVERY = 128
# upsert/remove 之后的落盘节流：每写一条记忆就重写一次 npz 是没必要的磁盘抖动。
SAVE_THROTTLE_SECONDS = 300.0


def _log(msg: str) -> None:
    print(f"[vector-memory] {msg}", file=sys.stderr, flush=True)


def _as_epoch(value: Any) -> float:
    """把 occurred_at 归一成 epoch 秒：可能是 float，也可能是 ISO 字符串。"""
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        dt = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except Exception:  # noqa: BLE001 —— 时间读不出来就当「不知道」，不要拦住建库
        return 0.0


class MemoryVectorIndex:
    """内存向量索引：643 条记忆 × 512 维，纯 numpy 余弦相似度。

    `path`      持久化文件（npz）。为空串 = 纯内存，不落盘。
    `embedder`  任何有 encode / encode_batch 的东西（见 embedding.py）。
    `loader`    没找到 npz 时用它拿全量语料：`() -> [{memory_id, content, status}]`。
                不传就只能读盘，读不到就永远为空（索引是增强，不是前提）。
    """

    def __init__(
        self,
        path: str,
        embedder: Any,
        *,
        loader: Optional[Callable[[], Sequence[Dict[str, Any]]]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.path = str(path or "")
        self.embedder = embedder
        self._loader = loader
        self._log = log or _log

        self._lock = threading.RLock()
        self._ids: List[str] = []
        self._contents: List[str] = []
        self._statuses: List[str] = []
        # 每条记忆的发生时间（epoch 秒）。融合打分里有「时间衰减」，
        # 没有这个字段，语义命中的老记忆就会被当成「刚发生」而多拿分。
        self._times: List[float] = []
        # (n, dim) float32，每行 L2 归一化过——余弦因此退化成点积。
        self._matrix: Optional[np.ndarray] = None
        self._row_of: Dict[str, int] = {}

        self._loaded = False
        self._building = False
        self._dirty = False
        self._last_save = 0.0
        self._built_at: Optional[float] = None

    # ------------------------------------------------------------ 元信息
    def _model_key(self) -> str:
        """向量空间的身份证：换了模型 / 维度，旧的矩阵就必须作废。

        两个 embedder 的维度碰巧都是 512，但语义空间完全不同——混用会让
        「相似度」变成随机数，而且不会报错。所以宁可重建。
        """
        name = getattr(self.embedder, "model_name", None) or type(self.embedder).__name__
        return f"{name}:{int(getattr(self.embedder, 'dim', 0) or 0)}"

    def _meta(self) -> Dict[str, Any]:
        return {"model": self._model_key(), "built_at": self._built_at, "version": 1}

    # ------------------------------------------------------------ 建
    def build(self, rows: Sequence[Dict[str, Any]]) -> None:
        """从全量语料重建索引（同步、会打进度）。

        rows: [{memory_id, content, status?}]。重复 id 只留第一条；
        空正文、空 id 直接跳过——索引里放这种东西只会制造噪声。
        """
        t0 = time.time()
        pairs: List[tuple] = []
        seen = set()
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            mid = str(r.get("memory_id") or r.get("id") or "").strip()
            content = str(r.get("content") or "")
            if not mid or mid in seen or not content.strip():
                continue
            seen.add(mid)
            pairs.append(
                (mid, content, str(r.get("status") or ""), _as_epoch(r.get("occurred_at")))
            )

        if not pairs:
            self._log("语料是空的，索引保持为空（语义召回会降级）")
            with self._lock:
                self._ids, self._contents, self._statuses, self._times = [], [], [], []
                self._matrix = None
                self._row_of = {}
                self._loaded = True
            return

        self._log(f"开始建索引：{len(pairs)} 条，模型 {self._model_key()}")
        vecs: List[List[float]] = []
        for i in range(0, len(pairs), BUILD_BATCH):
            batch = [c for _, c, _, _ in pairs[i : i + BUILD_BATCH]]
            try:
                vecs.extend(self.embedder.encode_batch(batch))
            except AttributeError:  # 只实现了 encode 的 embedder
                vecs.extend(self.embedder.encode(t) for t in batch)
            done = min(len(pairs), i + BUILD_BATCH)
            if done % PROGRESS_EVERY < BUILD_BATCH or done == len(pairs):
                self._log(f"  编码进度 {done}/{len(pairs)}（{time.time() - t0:.1f}s）")

        mat = np.asarray(vecs, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] != len(pairs):
            raise RuntimeError(f"向量维度不对：{getattr(mat, 'shape', None)} vs {len(pairs)} 条")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        # 一次性换上去：检索线程要么看到旧的完整索引，要么看到新的完整索引，
        # 不会看到「矩阵换了、ids 还没换」的中间态。
        with self._lock:
            self._ids = [p[0] for p in pairs]
            self._contents = [p[1] for p in pairs]
            self._statuses = [p[2] for p in pairs]
            self._times = [p[3] for p in pairs]
            self._matrix = mat
            self._row_of = {mid: i for i, mid in enumerate(self._ids)}
            self._loaded = True
            self._dirty = False
            self._built_at = time.time()
        self.save()
        self._log(
            f"建好了：{mat.shape[0]} 条 × {mat.shape[1]} 维，耗时 {time.time() - t0:.1f}s"
        )

    # ------------------------------------------------------------ 存 / 读
    def save(self) -> None:
        """落盘（原子：先写 .tmp.npz 再 replace）。失败只打日志，不抛。"""
        if not self.path:
            return
        with self._lock:
            if self._matrix is None or not self._ids:
                return
            ids = np.asarray(self._ids)
            # 正文/状态用 unicode 数组而不是 object 数组：npz 里不掺 pickle，
            # 加载时不必 allow_pickle（别为了省几 MB 给未来埋个反序列化口子）。
            contents = np.asarray(self._contents)
            statuses = np.asarray(self._statuses)
            times = np.asarray(self._times, dtype=np.float64)
            matrix = self._matrix
            meta = np.asarray(json.dumps(self._meta(), ensure_ascii=False))
            path = self.path
        tmp = ""
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = path + ".tmp.npz"
            np.savez(
                tmp,
                ids=ids,
                matrix=matrix,
                contents=contents,
                statuses=statuses,
                times=times,
                meta=meta,
            )
            os.replace(tmp, path)
            with self._lock:
                self._dirty = False
                self._last_save = time.time()
        except Exception as e:  # noqa: BLE001 —— 落盘失败不该挡住召回
            self._log(f"落盘失败（索引仍可用，只是重启要重建）：{e!r}")
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:  # noqa: BLE001
                    pass

    def _load(self) -> bool:
        """从 npz 读回索引。读不到 / 不兼容 → 返回 False（调用方去重建）。"""
        if not self.path or not os.path.exists(self.path):
            return False
        try:
            with np.load(self.path, allow_pickle=False) as z:
                ids = [str(x) for x in np.asarray(z["ids"]).tolist()]
                mat = np.asarray(z["matrix"], dtype=np.float32)
                contents = (
                    [str(x) for x in np.asarray(z["contents"]).tolist()]
                    if "contents" in z
                    else [""] * len(ids)
                )
                statuses = (
                    [str(x) for x in np.asarray(z["statuses"]).tolist()]
                    if "statuses" in z
                    else [""] * len(ids)
                )
                times = (
                    [float(x) for x in np.asarray(z["times"]).tolist()]
                    if "times" in z
                    else [0.0] * len(ids)
                )
                meta = json.loads(str(np.asarray(z["meta"]).tolist())) if "meta" in z else {}
        except Exception as e:  # noqa: BLE001 —— 文件坏了就重建，不要炸
            self._log(f"读 {self.path} 失败，准备重建：{e!r}")
            return False

        if not ids or mat.ndim != 2 or mat.shape[0] != len(ids):
            self._log(f"索引文件形状不对（{len(ids)} ids / {getattr(mat, 'shape', None)}），准备重建")
            return False
        # 换了向量器：旧矩阵和现在的语义空间不兼容。返回 False 让调用方重建，
        # 而不是把两种模型的空间混在一起算相似度。
        if int(mat.shape[1]) != int(getattr(self.embedder, "dim", mat.shape[1])):
            self._log(f"维度变了（{mat.shape[1]} → {self.embedder.dim}），重建")
            return False
        old_model = str(meta.get("model") or "")
        if old_model and old_model != self._model_key():
            self._log(f"向量器变了（{old_model} → {self._model_key()}），重建")
            return False
        if len(contents) != len(ids):
            contents = [""] * len(ids)
        if len(statuses) != len(ids):
            statuses = [""] * len(ids)
        if len(times) != len(ids):
            times = [0.0] * len(ids)

        with self._lock:
            self._ids = ids
            self._contents = contents
            self._statuses = statuses
            self._times = times
            self._matrix = mat
            self._row_of = {mid: i for i, mid in enumerate(ids)}
            self._loaded = True
            self._built_at = meta.get("built_at")
        return True

    # ------------------------------------------------------------ 懒加载
    def _ensure_loaded(self) -> None:
        """没索引就准备一份：能读盘就直接读；读不到就后台重建（不阻塞这一轮）。"""
        if self._loaded:
            return
        with self._lock:
            if self._loaded or self._building:
                return
            if self._load():
                self._log(f"从 {self.path} 加载了 {len(self._ids)} 条（没有重新 embed）")
                return
            self._building = True
        # 重建放到后台：600+ 条 embed 要一两分钟，挡在 HTTP 请求里等于超时。
        t = threading.Thread(target=self._rebuild_worker, name="vector-index-build", daemon=True)
        t.start()

    def _rebuild_worker(self) -> None:
        try:
            rows: List[Dict[str, Any]] = []
            if self._loader is not None:
                try:
                    rows = list(self._loader() or [])
                except Exception as e:  # noqa: BLE001
                    self._log(f"取语料失败，语义召回降级：{e!r}")
                    rows = []
            if not rows:
                with self._lock:
                    self._loaded = True
                self._log("没有语料可用，索引为空")
                return
            self.build(rows)
        except Exception as e:  # noqa: BLE001 —— 索引挂了只是降级，不能让服务倒
            self._log(f"后台建索引失败，语义召回降级：{e!r}")
            with self._lock:
                self._loaded = True
                self._matrix = None
        finally:
            with self._lock:
                self._building = False

    # ------------------------------------------------------------ 查
    def search(self, query: str, limit: int = 10, *, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """向量召回：返回 [{memory_id, content, score, status, occurred_at}]。

        score 是余弦相似度（越大越像）。occurred_at 是 epoch 秒；0 = 不知道，
        调用方据此跳过「时间衰减」，而不是把老记忆当成刚发生。

        第一次调用会触发懒加载：能从 npz 读就直接读（毫秒级），读不到就在
        后台重建——这种情况下**这一次**返回空，下一次才有语义结果。
        """
        q = str(query or "").strip()
        if not q:
            return []
        self._ensure_loaded()
        with self._lock:
            mat = self._matrix
            if mat is None or not self._ids:
                return []
            ids = list(self._ids)
            contents = list(self._contents)
            statuses = list(self._statuses)
            times = list(self._times)

        try:
            qv = np.asarray(self.embedder.encode(q), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            self._log(f"query 编码失败：{e!r}")
            return []
        if qv.ndim != 1 or qv.shape[0] != mat.shape[1] or not bool(np.any(qv)):
            return []
        n = float(np.linalg.norm(qv))
        if n > 0:
            qv = qv / n

        sims = mat @ qv
        k = min(max(1, int(limit or 1)), len(ids))
        if k < len(ids):
            top = np.argpartition(-sims, k - 1)[:k]
        else:
            top = np.arange(len(ids))
        top = top[np.argsort(-sims[top])]

        out: List[Dict[str, Any]] = []
        for i in top:
            score = float(sims[int(i)])
            if score < float(min_score):
                continue
            out.append(
                {
                    "memory_id": ids[int(i)],
                    "content": contents[int(i)],
                    "score": round(score, 4),
                    "status": statuses[int(i)],
                    "occurred_at": times[int(i)] or None,
                }
            )
        return out

    # ------------------------------------------------------------ 增 / 删
    def upsert(
        self,
        memory_id: str,
        content: str,
        *,
        status: str = "",
        occurred_at: Optional[float] = None,
    ) -> None:
        """加一条 / 换一条。**索引还没建就什么都不做**。

        为什么不在这里触发建库：建库要一两分钟，而 observe 是热路径。
        而且 build() 是从权威源全量读的，先写进来的这一条会被一起读到，
        丢不了——所以「没建就跳过」是安全且便宜的。
        """
        mid = str(memory_id or "").strip()
        text = str(content or "")
        if not mid or not text.strip() or not self._loaded:
            return
        try:
            vec = np.asarray(self.embedder.encode(text), dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
        except Exception as e:  # noqa: BLE001
            self._log(f"upsert 编码失败（跳过 {mid}）：{e!r}")
            return
        with self._lock:
            if self._matrix is None:
                return
            if vec.shape[0] != self._matrix.shape[1]:
                return
            row = self._row_of.get(mid)
            if row is None:
                self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])
                self._ids.append(mid)
                self._contents.append(text)
                self._statuses.append(str(status or ""))
                self._times.append(float(occurred_at or 0.0))
                self._row_of[mid] = len(self._ids) - 1
            else:
                self._matrix[row] = vec
                self._contents[row] = text
                if status:
                    self._statuses[row] = str(status)
                if occurred_at:
                    self._times[row] = float(occurred_at)
            self._dirty = True
        self._maybe_save()

    def remove(self, memory_id: str) -> None:
        """删一条（被纠正的记忆不该再从索引里被召回）。没建索引就跳过。"""
        mid = str(memory_id or "").strip()
        if not mid or not self._loaded:
            return
        with self._lock:
            row = self._row_of.get(mid)
            if row is None or self._matrix is None:
                return
            keep = np.ones(self._matrix.shape[0], dtype=bool)
            keep[row] = False
            self._matrix = self._matrix[keep]
            del self._ids[row]
            del self._contents[row]
            del self._statuses[row]
            del self._times[row]
            self._row_of = {m: i for i, m in enumerate(self._ids)}
            self._dirty = True
        self._maybe_save()

    def _maybe_save(self) -> None:
        if not self._dirty:
            return
        if time.time() - self._last_save < SAVE_THROTTLE_SECONDS:
            return
        self.save()

    # ------------------------------------------------------------ 状态
    def stats(self) -> Dict[str, Any]:
        """诊断用。**不会**触发建库（stats 不该有副作用）。"""
        with self._lock:
            rows = len(self._ids)
            dim = int(self._matrix.shape[1]) if self._matrix is not None else int(
                getattr(self.embedder, "dim", 0) or 0
            )
            building = self._building
            loaded = self._loaded
        return {
            "path": self.path,
            "rows": rows,
            "dim": dim,
            "loaded": loaded,
            "building": building,
            "model": self._model_key(),
            "built_at": self._built_at,
        }



