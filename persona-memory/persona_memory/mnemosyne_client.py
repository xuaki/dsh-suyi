# -*- coding: utf-8 -*-
"""dsh-mnemosyne REST 客户端。

只调用它的 HTTP API，不改它的代码（AGPL 项目，调用不构成衍生；
融合层保持自己的许可证与部署边界）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class MnemosyneError(RuntimeError):
    def __init__(self, status: int, body: Any = None) -> None:
        super().__init__(f"mnemosyne http {status}: {body}")
        self.status = status
        self.body = body


class MnemosyneClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        token: str = "dev-token",
        *,
        timeout: float = 10.0,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.api_prefix = api_prefix.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )

    # ---------------- 传输 ----------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"

    def _req(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:500]
            raise MnemosyneError(resp.status_code, body)
        if not resp.content:
            return None
        return resp.json()

    # ---------------- 探活 ----------------
    def healthz(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}/healthz", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        return self._req("GET", "/health")

    # ---------------- 观察 ----------------
    def capture_observation(
        self,
        *,
        content: str,
        source_kind: str = "dialog",
        occurred_at: Optional[str] = None,
        producer: str = "persona-memory",
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source_kind": source_kind,
            "content": content,
            "producer": producer,
            "metadata": metadata or {},
        }
        if occurred_at:
            payload["occurred_at"] = occurred_at
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        return self._req("POST", "/observations", json=payload)

    # ---------------- 记忆 ----------------
    def propose_memory(
        self,
        *,
        content: str,
        type: str = "semantic",
        source_ids: Optional[List[str]] = None,
        status: str = "candidate",
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": type,
            "content": content,
            "source_ids": source_ids or [],
            "status": status,
            "metadata": metadata or {},
        }
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        return self._req("POST", "/memories", json=payload)

    def update_memory(self, memory_id: str, **fields: Any) -> Dict[str, Any]:
        return self._req("PATCH", f"/memories/{memory_id}", json=fields)

    def get_memory(self, memory_id: str, *, include_evidence: bool = False) -> Dict[str, Any]:
        return self._req(
            "GET", f"/memories/{memory_id}", params={"include_evidence": str(include_evidence).lower()}
        )

    def forget_memory(self, memory_id: str, *, hard: bool = False) -> Dict[str, Any]:
        return self._req("DELETE", f"/memories/{memory_id}", params={"hard": str(hard).lower()})

    def search(
        self,
        *,
        query: str = "",
        types: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
        limit: int = 10,
        include_evidence: bool = False,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"query": query, "limit": limit, "include_evidence": include_evidence}
        if types:
            payload["types"] = types
        if status:
            payload["status"] = status
        out = self._req("POST", "/memories/search", json=payload)
        return list((out or {}).get("items", []))

    def build_context(
        self, *, query: str, token_budget: Optional[int] = None, limit: int = 12
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "limit": limit}
        if token_budget is not None:
            payload["token_budget"] = token_budget
        return self._req("POST", "/contexts/build", json=payload)

    def feedback(self, *, memory_id: str, useful: Optional[bool] = None, incorrect: Optional[bool] = None, note: str = "") -> Dict[str, Any]:
        return self._req(
            "POST",
            "/feedback",
            json={"memory_id": memory_id, "useful": useful, "incorrect": incorrect, "note": note},
        )
