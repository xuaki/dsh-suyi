# -*- coding: utf-8 -*-
"""REST 接口：给 DSH / 桌宠 / 任何本机客户端用。

一个原则：人格和记忆都是**可观测的**。所有决策都要能解释，
所以 /retrieve 返回来源，/persona 返回三层，/explain 能说清一条记忆为什么在。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .content_guard import reason_message
from .correction_harvest import detect_correction
from .failures import FailureBook, summarize
from .gaps import GapStore, _is_agent_relay, harvest_profile_gaps
from .interest_harvest import extract_implicit_topics
from .interests import InterestBook
from .presence import Presence
from .profile import build_profile_digest
from .service import PersonaMemoryService
from .ui import PAGE


# 构建戳：服务代码一改就换一个。DSH 的 bootstrap 会比对它，
# 不一致就自己重启服务——改 Python 不用再手动杀进程（2026-11-07 加）。
SERVICE_BUILD = "2026-11-14-harvest-inputs2"


def _write_text_atomic(path: str, text: str, *, only_if_changed: bool = True) -> bool:
    """原子写一段文本。返回是否真的写了。

    先写 .tmp 再 os.replace：半截文件比没有文件更糟（profile.md 会被手改/被读）。
    """
    try:
        if only_if_changed and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    if f.read() == text:
                        return False
            except Exception:  # noqa: BLE001 —— 读不出来就当需要重写
                pass
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
        return True
    except Exception:  # noqa: BLE001 —— 落盘失败不该让 /profile 看起来坏了
        return False



class ObserveIn(BaseModel):
    text: str
    source_kind: str = "dialog"
    tier: str = "attention"
    strength: float = 0.6
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveIn(BaseModel):
    query: str
    limit: int = 10
    token_budget: Optional[int] = None
    # 默认 False：只返回摘要 + 引用（Inline & Spill）。
    # 要原文时显式传 true，或拿 memory_id 调 /expand。
    detail: bool = False
    # >0 时只回这个时间窗内的记忆（「我们昨天聊了什么」靠它）
    since_hours: float = 0.0
    # 记忆路由：这次问的是哪类问题（tech/emotion/preference/project）。
    # 不传就由服务端按 query 自动判断；判断不出就不加成。
    cluster: Optional[str] = None


class ExpandIn(BaseModel):
    memory_id: str
    depth: int = 1


class FeedbackIn(BaseModel):
    memory_id: str
    note: str = ""


class EmotionIn(BaseModel):
    releases: Dict[str, float] = Field(default_factory=dict)


class TraitEvidenceIn(BaseModel):
    """经验层证据：一条独立证据，不是一次改人格。"""

    trait: str
    delta: float = 0.2
    polarity: int = 1
    evidence_id: Optional[str] = None
    note: str = ""
    # 支撑这条证据的那条记忆。有它，证据能点回原句，记忆被纠正时也能扣回去。
    memory_id: Optional[str] = None


class PromoteIn(BaseModel):
    limit: int = 50


class GapIn(BaseModel):
    """种一条未解执念。"""

    text: str
    strength: float = 1.0
    source: str = "manual"
    note: str = ""


class GapCloseIn(BaseModel):
    id: str
    closed_by: str = ""
    note: str = ""


class InterestIn(BaseModel):
    """手动补一条「他喜欢什么」。"""

    topic: str
    example: str = ""


class InterestSignalIn(BaseModel):
    """一条弱证据：他顺口提到某样东西。不直接算确认。"""

    topic: str
    polarity: int = 1
    tier: str = "passive"
    quote: str = ""
    confidence: float = 0.5


class FailureIn(BaseModel):
    """记一条「我说错/做错的事」——我自己的翻车账本。"""

    text: str
    reason: str = ""
    corrected_to: str = ""
    category: str = "其它"


# 纠正时的「理由」有时候只是机器给的一个占位符，不是人话。
_CORRECTION_PLACEHOLDERS = {"", "user_correction", "user-correction", "correction"}


def _correction_reason(note: str) -> str:
    """把机器占位符翻成一句人话；他真给了理由就用他的原话。"""
    n = str(note or "").strip()
    if n.lower() in _CORRECTION_PLACEHOLDERS:
        return "他说我记错了"
    return n


def _is_housekeeping_note(note: str) -> bool:
    """测试清理也走 /feedback。别让「test_api_cleanup」这种账混进翻车记录里。"""
    n = str(note or "").strip().lower()
    return "cleanup" in n or n.startswith("test")


def _dig_text(row: Any) -> str:
    """从记忆行里挖出正文摘要。形状不稳定，按常见字段逐层找。"""
    if not isinstance(row, dict):
        return ""
    for key in ("content", "text", "summary"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return summarize(v)
    inner = row.get("memory")
    if isinstance(inner, dict):
        return _dig_text(inner)
    return ""


def _memory_text(service: PersonaMemoryService, memory_id: str) -> str:
    """取被纠正那条记忆的摘要。拿不到就返回空——不编内容。"""
    try:
        return _dig_text(service.mnemosyne.get_memory(memory_id))
    except Exception:  # noqa: BLE001
        return ""


def create_app(service: PersonaMemoryService) -> FastAPI:
    app = FastAPI(title="persona-memory", version="0.1.0")
    app.state.service = service

    # 缺口池：未解执念。和人格状态放同一个 data 目录，服务重启也不丢。
    gap_path = None
    if service.persona_path:
        gap_path = os.path.join(os.path.dirname(service.persona_path), "gaps.json")
    gap_store = GapStore(gap_path)

    # 在场感：上一次有互动是什么时候。只记时间，不记别的。
    presence_path = None
    if service.persona_path:
        presence_path = os.path.join(os.path.dirname(service.persona_path), "presence.json")
    presence = Presence(presence_path)

    # 兴趣画像：记住「你喜欢什么」。只从明确句式里学，不猜。
    interest_path = None
    if service.persona_path:
        interest_path = os.path.join(os.path.dirname(service.persona_path), "interests.json")
    interest_book = InterestBook(interest_path)

    # 翻车账本：我自己说错/做错的事。他纠正一次，教训得留下一次。
    failure_path = None
    if service.persona_path:
        failure_path = os.path.join(os.path.dirname(service.persona_path), "failures.json")
    failure_book = FailureBook(failure_path)

    # 常驻小画像：一块 ≤400 字的「他是什么样的」，落盘成人类可读的 md，可手改。
    profile_path = None
    if service.persona_path:
        profile_path = os.path.join(os.path.dirname(service.persona_path), "profile.md")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        try:
            mn = service.mnemosyne.health()
        except Exception as e:  # noqa: BLE001
            mn = {"ok": False, "error": str(e)}
        return {
            "ok": bool(mn.get("ok")),
            "service": "persona-memory",
            "version": "0.1.0",
            "build": SERVICE_BUILD,
            "mnemosyne": mn,
            "vector_up": service._vector_up,
            "vector": service.vectors.stats() if (service._vector_up and service.vectors) else None,
            # 内存向量索引（Milvus 的替补）：loaded=False / building=True 表示
            # 还在懒加载，这一轮召回没有语义腿，别把它误读成「坏了」。
            "memory_index": (
                service.vector_memory.stats() if service.vector_memory is not None else None
            ),
            "vector_engine": "milvus" if service._milvus_usable() else (
                "memory_index" if service.vector_memory is not None else "none"
            ),
        }

    @app.post("/observe")
    def observe(body: ObserveIn) -> Dict[str, Any]:
        presence.touch()
        text = body.text or ""
        # 只从「他说的」里学兴趣；我自己的回复说「我最喜欢…」不算他的偏好
        role = str((body.metadata or {}).get("role") or "")
        # 转述的不是他：子智能体的报告经 send_message 回来时落成 role="user"，
        # 里面的句子是**转述**，不是他此刻在跟我说话（gaps 那边同一条闸，
        # 2026-11-14 就因为没这道闸把一份报告收成了执念）。
        spoke_to_me = role != "assistant" and not _is_agent_relay(text)
        if spoke_to_me:
            # 语义归并：真句向量时把「吉他/吉它」合成一个话题，避免 count 分散
            emb = getattr(service.vectors, "embedder", None) if service.vectors else None
            explicit = interest_book.observe(body.text, embedder=emb) or []
            # 这次真正学到的话题：明确的 + 下面隐式抽的（响应里能看见）
            learned = list(explicit)
            # 隐式兴趣（2026-11-14）：481 条真实消息里「我喜欢X」这类明确句式
            # 命中 0 条——只认教科书句式等于永远收不到。所以补一条「顺口一提」：
            # 「我最近又去滑雪了」抽一条 passive 线索。它不是确认，
            # 单条永远不会把话题推成「他喜欢X」（见 interests.interest_verdict）。
            try:
                for topic in extract_implicit_topics(body.text):
                    if topic in explicit:
                        continue  # 这句已经按明确句式记过 engage，别重复记一条
                    interest_book.observe_signal(
                        topic,
                        tier="passive",
                        quote=text[:80],
                        confidence=0.4,
                    )
                    if topic not in learned:
                        learned.append(topic)
            except Exception:  # noqa: BLE001 —— 收割失败不该挡住记忆写入
                pass
        # 纠正识别（2026-11-14）：`su_forget` 18 天 0 次调用，失败账本永远是 0。
        # 这里先只做判断（纯函数、无副作用）；真正落账要等准入结论出来。
        correction = detect_correction(text) if spoke_to_me else None
        # 执念收割：自动收「没弄明白的事」。三道闸在 gaps.is_gap_worthy 里。
        # 收割失败不该挡住记忆写入——所以单独 try。
        try:
            gap_store.harvest(body.text or "", role=role)
        except Exception:  # noqa: BLE001
            pass
        r = service.observe(
            body.text,
            source_kind=body.source_kind,
            tier=body.tier,
            strength=body.strength,
            metadata=body.metadata,
        )
        # 纠正落账（准入判定之后）。
        # 账本和记忆入库是两回事：准入不放行的短句（「记错了」只有三个字）
        # 照样是他的一次纠正，不能因为没入库就当没发生；但被**硬拒**的
        # （乱码 / 提示注入）不记——那不是一次真实的对话纠正，记了就是假账。
        if correction and correction.get("strength") == "strong":
            if not str(r.reason).startswith("rejected:"):
                try:
                    failure_book.add(
                        f"（他说我在纠正）{text[:100]}",
                        reason=correction["reason"],
                        category="记忆",
                    )
                except Exception:  # noqa: BLE001 —— 账本坏了不该挡住对话
                    pass
        return {
            "admitted": r.admitted,
            "reason": r.reason,
            # 机器码在上面，人话在下面：被安检拦下的要说清「为什么没记」
            "message": (
                reason_message(r.reason)
                or ("记下了" if r.admitted else "这条没记：暂不满足入库条件")
            ),
            "observation_id": r.observation_id,
            "memory_id": r.memory_id,
            "indexed": r.indexed,
            "persona_dynamic": r.persona_dynamic,
            "scores": r.scores,
            # 这两条是给「为什么账本里多了一条」用的：自动采集做了什么，可观测。
            "learned_interests": learned if spoke_to_me else [],
            "correction": correction,
        }

    @app.post("/retrieve")
    def retrieve(body: RetrieveIn) -> Dict[str, Any]:
        return service.retrieve(
            body.query,
            limit=body.limit,
            token_budget=body.token_budget,
            detail=body.detail,
            since_hours=body.since_hours,
            cluster=body.cluster,
        )

    @app.post("/expand")
    def expand(body: ExpandIn) -> Dict[str, Any]:
        """链式召回：从一条记忆出发，把它的正文和邻居摘要拉回来。"""
        return service.expand(body.memory_id, depth=body.depth)

    @app.get("/proactive")
    def proactive(limit: int = 1) -> Dict[str, Any]:
        """主动提起：挑一件此刻值得想起来的旧事（会标记冷却期，别频繁调）。"""
        return service.proactive_recall(limit=limit)

    @app.get("/graph")
    def graph(limit: int = 50) -> Dict[str, Any]:
        """记忆网络：最近一批记忆的节点 + 引用边，给看板画图用。"""
        return service.graph(limit=limit)

    @app.get("/topics")
    def topics(window: int = 60, k: int = 5) -> Dict[str, Any]:
        """话题聚类：最近的记忆按语义聚成几类。"""
        return service.topics(window=window, k=k)

    @app.get("/selfcheck")
    def selfcheck() -> Dict[str, Any]:
        """轻量自检：索引一致性 / 元数据 / 链接覆盖 / 重复。su_health 会调它。"""
        return service.selfcheck()

    @app.post("/persona/soothe")
    def soothe() -> Dict[str, Any]:
        """安抚自己：想起那些「被爱着」的时刻（会释放催产素/内啡肽）。"""
        return service.soothe(force=True)

    @app.get("/persona")
    def persona() -> Dict[str, Any]:
        return service.persona.snapshot()

    @app.post("/persona/emotion")
    def emotion(body: EmotionIn) -> Dict[str, Any]:
        service.persona.release_emotion(body.releases)
        service.save_persona()
        return service.persona.snapshot()["dynamic"]

    @app.post("/persona/trait")
    def trait_evidence(body: TraitEvidenceIn) -> Dict[str, Any]:
        """给经验层（经验塑造）投一条证据。

        ≥3 条独立证据才开始动值；每次位移 ≤0.05；目标先向基础锚收缩。
        人格不会因为一次聊天就变——这正是设计要的。
        """
        return service.record_trait_evidence(
            body.trait,
            delta=body.delta,
            polarity=body.polarity,
            evidence_id=body.evidence_id,
            note=body.note,
            memory_id=body.memory_id or "",
        )

    @app.get("/trait/evidence")
    def trait_evidence(trait: str, limit: int = 10) -> Dict[str, Any]:
        """列出某个特质的证据，并把能查到的记忆摘要一起带出来。

        「我为什么好奇变强了」点得开：每条证据给出 memory_id 和原句摘要；
        记忆被清理或服务暂时不通时那条摘要留空，不影响证据本身。
        """
        return service.trait_evidence(trait, limit=limit)

    # ---------------- 缺口池：未解执念 ----------------
    @app.get("/interests")
    def interests(limit: int = 20) -> Dict[str, Any]:
        """他喜欢什么。已确认的排前面；没确认的也返回（带原话，给「待观察」用）。"""
        return interest_book.snapshot(limit=limit)

    @app.post("/interest")
    def add_interest(body: InterestIn) -> Dict[str, Any]:
        """手动补一条：模型判断「他刚才是在说喜欢这个」时用。"""
        try:
            t = interest_book.add(body.topic, example=body.example)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, "topic": t.to_dict()}

    @app.post("/interest/signal")
    def add_interest_signal(body: InterestSignalIn) -> Dict[str, Any]:
        """记一条弱证据：他顺口提到某样东西。

        这不是确认——单条永远不会把话题推成「他喜欢X」。它只是把线索攒起来，
        跨天反复出现之后，状态机才会自己把它判成 confirmed（见 interests.py）。
        """
        try:
            t = interest_book.observe_signal(
                body.topic,
                polarity=body.polarity,
                tier=body.tier,
                quote=body.quote,
                confidence=body.confidence,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e)) from e
        # 画像缺口：写完弱信号后顺手扫一遍，把「只有 1 条弱信号、且久没再出现」的
        # 话题种成 profile gap。挂在 /interest/signal 而不是 /observe 的理由：
        # 弱信号现在有两条来源（这里是模型主动丢的，/observe 里是隐式收割的），
        # 但「种成执念」仍然只在**主动**丢证据时做——/observe 每句话都种一遍，
        # 池子会被「他到底喜不喜欢公司」这种噪声灌满，执念就不值钱了。
        # 种缺口失败不该挡 signal 写入——所以单独 try。
        try:
            harvest_profile_gaps(interest_book, gap_store)
        except Exception:  # noqa: BLE001
            pass
        return {
            "ok": True,
            "topic": t.to_dict(),
            "note": "留了个线索。还不算他的偏好——攒够了它自己会变。",
        }

    # ---------------- 常驻小画像 ----------------
    @app.get("/profile")
    def get_profile(refresh: bool = True) -> Dict[str, Any]:
        """一块 ≤400 字的常驻画像：已确认兴趣 / 最绷的执念 / 最近翻过的车。

        同时落盘 data/profile.md（原子写）。`refresh=false` 时直接读盘返回，
        手改过的内容会原样回来——不过下一次 refresh 会重新生成覆盖它。
        """
        if not refresh and profile_path and os.path.exists(profile_path):
            try:
                with open(profile_path, encoding="utf-8") as f:
                    return {
                        "digest": f.read().strip(),
                        "generated_at": datetime.fromtimestamp(
                            os.path.getmtime(profile_path), timezone.utc
                        ).isoformat(timespec="seconds"),
                    }
            except Exception:  # noqa: BLE001 —— 读不出来就重新生成
                pass
        digest = build_profile_digest(interest_book, gap_store, failure_book)
        if profile_path:
            _write_text_atomic(profile_path, digest + "\n" if digest else "")
        return {
            "digest": digest,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ---------------- 翻车账本：我自己翻过的车 ----------------
    @app.get("/failures")
    def failures(limit: int = 5, max_days: float = 7) -> Dict[str, Any]:
        """最近翻过的车。只给最近 `max_days` 天、最多 `limit` 条。"""
        items = failure_book.recent(limit=limit, max_days=max_days)
        return {"items": items, "count": len(items), "total": len(failure_book.failures)}

    @app.post("/failure")
    def add_failure(body: FailureIn) -> Dict[str, Any]:
        """手动补一条翻车。同一句话重复记时累加次数，不会堆两条。"""
        try:
            f = failure_book.add(
                body.text,
                reason=body.reason,
                corrected_to=body.corrected_to,
                category=body.category,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, "failure": f.to_dict()}

    @app.get("/presence")
    def get_presence() -> Dict[str, Any]:
        """上次有互动是多久以前。只给间隔，不给时间戳——报读数是机器的说法。"""
        return presence.snapshot()

    @app.get("/gaps")
    def gaps(status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """列未解执念。张力是惰性算的——服务停机期间它照样在绷。"""
        return gap_store.list(status=status, limit=limit)

    @app.post("/gap")
    def add_gap(body: GapIn) -> Dict[str, Any]:
        """种一条「没弄明白的事」。同一条 open 的执念不会种两遍。"""
        try:
            g = gap_store.add(body.text, strength=body.strength, source=body.source, note=body.note)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, "gap": g.to_dict()}

    @app.post("/gap/close")
    def close_gap(body: GapCloseIn) -> Dict[str, Any]:
        """闭合一条执念。闭合后张力归零。"""
        g = gap_store.close(body.id, closed_by=body.closed_by, note=body.note)
        if not g:
            raise HTTPException(status_code=404, detail="没有这条执念")
        return {"ok": True, "gap": g.to_dict()}

    @app.post("/feedback")
    def feedback(body: FeedbackIn) -> Dict[str, Any]:
        # 纠正前先拿到原句：一旦 invalidated，就不一定还能读到了。
        snippet = _memory_text(service, body.memory_id)
        try:
            out = service.correct(body.memory_id, note=body.note)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e)) from e
        # 纠正只让那条记忆失效，教训本身必须留下——否则下一轮照犯。
        # 记账单独 try：账本坏了不该让「纠正」看起来失败。
        if not _is_housekeeping_note(body.note):
            try:
                f = failure_book.add(
                    snippet or f"一条被纠正的记忆（{body.memory_id}）",
                    reason=_correction_reason(body.note),
                    category="记忆",
                )
                if isinstance(out, dict):
                    out = {**out, "failure": f.to_dict()}
            except Exception:  # noqa: BLE001
                pass
        return out

    @app.post("/promote")
    def promote(body: PromoteIn) -> Dict[str, Any]:
        return service.promote_candidates(limit=body.limit)

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        return service.snapshot()

    # ---------------- 看板 ----------------
    @app.get("/ui", response_class=HTMLResponse)
    def board() -> str:
        """溯忆看板：一页看完记得什么、现在什么心情。"""
        return PAGE

    @app.get("/memories")
    def memories(limit: int = 300) -> Dict[str, Any]:
        """列记忆（给看板用）。权威源是 mnemosyne，这里只做转发。"""
        try:
            rows = service.mnemosyne.search(query="", limit=limit)
        except Exception as e:  # noqa: BLE001
            return {"items": [], "count": 0, "error": str(e)}
        return {"items": rows, "count": len(rows)}

    return app
