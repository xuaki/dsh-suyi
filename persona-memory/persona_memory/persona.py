# -*- coding: utf-8 -*-
"""人格三层：基础人格 / 经验塑造 / 动态状态。

设计依据（来自 dsh-mnemosyne service/README.md §二.2 与 STATE_MACHINE）：

| 层 | 内容 | 更新节奏 |
| 基础人格 | 性格倾向、表达风格、价值边界 | 相对稳定，不因一两轮对话改写 |
| 经验塑造 | 兴趣、熟悉度、行为习惯 | 随经历缓慢更新，保留依据 |
| 动态状态 | 当前情绪、注意力、疲劳 | 短期波动，可恢复 |

要避免的两件事：
- 人格固化（只会换说法重复人设）—— 上一轮讨论里被否决的「过拟合」做法
- 人格漂移（一段对话突然改长期立场）—— 无门槛更新的后果

实现策略：高锚定、低更新速率。
经验层每次更新只走一小步（alpha），且需要至少 N 条独立证据；
基础层只允许显式命令修改（用户重置/授权），日常对话永不触碰。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .molecules import EndocrineSystem


# 经验层单次最大位移：防止一次强烈事件改写长期倾向
SHAPED_UPDATE_ALPHA = 0.08
# 晋升到「经验层可写」所需的最少独立证据数
SHAPED_MIN_EVIDENCE = 3
# 病历保留多少条。原先只留 10 条，但那意味着第 11 条之前的证据一旦被纠正，
# 就查不到当初的 delta、扣不回去（2026-11-12 加 reconcile 时发现）。
# 晋升门要求 20 条证据，所以病历至少要装得下 20 条；50 是留余量。
EVIDENCE_NOTE_LIMIT = 50
# 基础层：只允许显式修改，日常路径一律拒绝
BASE_WRITE_REQUIRES_EXPLICIT = True

# ── 基础层慢通道（Nested Learning 的层间传递，最小实现）───────────
# 论文的 CMS 里，低频层是从高频层的压缩中学习的。我们基础层默认只读
# （防漂移），但「永远不动」会变成僵死。所以留一条极窄的门：
# 证据够多、跨得够久、冷却够，才准挪一步，而且一次只挪 0.01。
BASE_PROMOTE_MIN_EVIDENCE = 20      # 至少这么多条独立证据
BASE_PROMOTE_MIN_SPAN_DAYS = 90.0   # 证据时间跨度至少这么久
BASE_PROMOTE_MAX_STEP = 0.01        # 单次位移硬上限
BASE_PROMOTE_COOLDOWN_DAYS = 30.0   # 两次基础层更新至少隔这么久

# ── 疲劳：交互消耗的精力，不是情绪计数器 ──────────────────────────
# 以前疲劳只跟「情绪词」挂钩（每命中一次 +0.05）：纯技术对话一整天零疲劳，
# 反倒一句「晚安 mua」让人累——这不对。而且撞顶 1.0 之后，按 2 小时半衰期
# 回落，两小时后还有 0.5，状态里还写着「很疲倦」。
#
# 所以：① 疲劳改由「说了多少」驱动（见 service.observe / PersonaState.add_fatigue）；
#       ② 离开够久就当「睡了一觉」，直接清零。
# 为什么是 6 小时：它不是一个衰减参数，而是一个比喻——人不会因为「休息了
# 6 小时」而精确地被恢复 90%，但会睡一觉，醒来就是 0。半衰期模型对短时间
# 小憩是对的（喘口气，疲劳少一点），对一个通宵就荒唐了。
FATIGUE_RESET_SECONDS = 6 * 3600.0


@dataclass
class TraitCell:
    """经验层的一个特质格：值 + 证据 + 时间。"""

    value: float = 0.5
    support: float = 0.0
    oppose: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    # 证据的「病历」：每条留一句话和时间。
    # 只有 id 的话，我只能看到 0.7731 这个数，说不出它是被什么推上来的——
    # Nested Learning 的 CMS 里每层都有自己的 context flow，这就是那东西的最小版。
    # 2026-11-12 起每条还带 memory_id / delta / polarity：
    #   - memory_id 让「我为什么好奇变强了」能点回原句；
    #   - delta + polarity 是扣账依据——记忆被纠正时要按当初加的量扣回去。
    evidence_notes: List[Dict[str, Any]] = field(default_factory=list)
    # 被纠正后失效的证据 id：从 evidence_ids 移到这里，保持可追溯（不删账）。
    invalidated_evidence: List[str] = field(default_factory=list)
    # 第一条证据的时间。单独存，不从 evidence_notes 推——
    # 病历只留最近 10 条，拿它算跨度会让「时间够久」这道门永远打不开
    # （2026-11-11 单测抓到的 bug）。
    first_evidence_at: float = 0.0
    updated_at: float = 0.0
    last_verified_at: float = 0.0
    version: int = 1

    def net(self, oppose_discount: float = 0.7) -> float:
        return self.support - oppose_discount * self.oppose

    def to_dict(self) -> Dict[str, object]:
        return {
            "value": round(self.value, 4),
            "support": round(self.support, 4),
            "oppose": round(self.oppose, 4),
            "evidence_count": len(self.evidence_ids),
            "invalidated_count": len(self.invalidated_evidence),
            "version": self.version,
            # 只给最近三条：病历是给「想起来」用的，不是给「翻档案」用的
            "evidence_notes": [
                {
                    "note": str(e.get("note", ""))[:120],
                    "at": e.get("at"),
                    # 带上 memory_id，前端/模型才能顺着它调 expand 看原句
                    "memory_id": e.get("memory_id"),
                    "invalidated": bool(e.get("invalidated")),
                }
                for e in self.evidence_notes[-3:]
            ],
        }


@dataclass
class DynamicState:
    """动态状态层：分子 + 时间戳。"""

    endocrine: EndocrineSystem = field(default_factory=EndocrineSystem)
    last_tick_at: float = field(default_factory=time.time)
    fatigue: float = 0.0  # 0..1，随连续交互上升，随时间回落

    def tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        dt = max(0.0, now - self.last_tick_at)
        self.endocrine.tick(dt)
        if dt >= FATIGUE_RESET_SECONDS:
            # 隔了这么久，就当睡了一觉：直接清零。
            # 半衰期式的回落有个尴尬的尾巴——撞过 1.0 之后两小时还有 0.5，
            # 人明明睡醒了，状态里却还写着「很疲倦」。
            self.fatigue = 0.0
        else:
            # 没到「睡一觉」的尺度，就只是喘了口气：2 小时半衰期慢慢回落。
            self.fatigue *= pow(0.5, dt / 7200.0)
        self.last_tick_at = now


@dataclass
class PersonaState:
    """三层人格状态。基础层是只读锚，经验层慢变，动态层快变。"""

    base: Dict[str, Dict[str, object]] = field(default_factory=dict)
    shaped: Dict[str, TraitCell] = field(default_factory=dict)
    dynamic: DynamicState = field(default_factory=DynamicState)
    last_shaped_write_at: float = 0.0
    # 只读回查：给 evidence_notes 里的 memory_id 取一句记忆摘要。
    # 由 service 在构造时注入（PersonaState 不该知道 mnemosyne 怎么连）；
    # 不写进 to_state_dict——这是运行时的能力，不是人格状态。
    memory_lookup: Optional[Callable[[str], str]] = field(
        default=None, repr=False, compare=False
    )

    # ---------- 基础层 ----------
    @classmethod
    def from_base_profile(cls, profile: Dict[str, Dict[str, object]]) -> "PersonaState":
        """profile 形如 {"warmth": {"value": 0.85, "note": "..."}, ...}。

        基础层是人格锚：这里写入后，日常接口不允许改。
        """
        state = cls(base={k: dict(v) for k, v in profile.items()})
        # 经验层从基础层初始化，但从此独立演化
        for name, spec in profile.items():
            state.shaped[name] = TraitCell(
                value=float(spec.get("value", 0.5)),
                updated_at=time.time(),
                last_verified_at=time.time(),
            )
        return state

    def set_base(
        self, trait: str, value: float, *, explicit_user_command: bool = False
    ) -> bool:
        """改基础层。日常路径调用会被拒绝——这是防「人格漂移」的硬闸。"""
        if BASE_WRITE_REQUIRES_EXPLICIT and not explicit_user_command:
            return False
        self.base[trait] = dict(self.base.get(trait, {}), value=float(value))
        return True

    def base_value(self, trait: str, default: float = 0.5) -> float:
        spec = self.base.get(trait)
        if not spec:
            return default
        try:
            return float(spec.get("value", default))
        except (TypeError, ValueError):
            return default

    def _promotion_candidate(
        self, trait: str, now: Optional[float] = None
    ) -> Optional[Dict[str, object]]:
        """纯判断：如果现在该晋升，会怎么动。**不改任何状态**。

        四道门，全过才返回候选：
          1. 证据数 ≥ BASE_PROMOTE_MIN_EVIDENCE（20 条）
          2. 证据时间跨度 ≥ BASE_PROMOTE_MIN_SPAN_DAYS（90 天）
          3. 距上次基础层更新 ≥ BASE_PROMOTE_COOLDOWN_DAYS（30 天）
          4. 经验值确实偏离基础值（不然白动）
        单次最多挪 BASE_PROMOTE_MAX_STEP（0.01）。
        """
        now = time.time() if now is None else now
        cell = self.shaped.get(trait)
        if cell is None or len(cell.evidence_ids) < BASE_PROMOTE_MIN_EVIDENCE:
            return None
        stamps = [float(e.get("at") or 0.0) for e in cell.evidence_notes if e.get("at")]
        if not stamps:
            return None
        # 跨度用 first_evidence_at 算，不用 min(stamps)：
        # stamps 只是最近 10 条，拿它算会让门永远打不开。
        first = float(cell.first_evidence_at or min(stamps))
        span_days = (now - first) / 86400.0
        if span_days < BASE_PROMOTE_MIN_SPAN_DAYS:
            return None
        spec = dict(self.base.get(trait, {}))
        last = float(spec.get("last_promoted_at", 0.0) or 0.0)
        if last and (now - last) < BASE_PROMOTE_COOLDOWN_DAYS * 86400.0:
            return None
        cur = self.base_value(trait)
        target = float(cell.value)
        if abs(target - cur) < 1e-6:
            return None
        step = max(-BASE_PROMOTE_MAX_STEP, min(BASE_PROMOTE_MAX_STEP, target - cur))
        return {
            "trait": trait,
            "base_before": round(cur, 4),
            "base_after": round(max(0.0, min(1.0, cur + step)), 4),
            "step": round(step, 4),
            "evidence_count": len(cell.evidence_ids),
            "span_days": round(span_days, 1),
        }

    def maybe_promote_to_base(self, trait: str, now: Optional[float] = None) -> Optional[Dict[str, object]]:
        """真改版：够格就把基础层挪一步。

        为什么这么严：基础层一松，人格就会漂；但永远不松，人格就成了
        一张永远不更新的照片。慢，是它唯一的保护。
        """
        now = time.time() if now is None else now
        cand = self._promotion_candidate(trait, now)
        if cand is None:
            return None
        spec = dict(self.base.get(trait, {}))
        spec["value"] = cand["base_after"]
        spec["last_promoted_at"] = now
        spec["promoted_from_shaped"] = round(float(self.shaped[trait].value), 4)
        spec["promoted_evidence_count"] = cand["evidence_count"]
        spec["promoted_span_days"] = cand["span_days"]
        self.base[trait] = spec
        return {"applied": True, **cand}

    def shadow_report(self, now: Optional[float] = None) -> Dict[str, Dict[str, object]]:
        """影子模式：对所有**有证据**的特质算一份「晋升画像」——只算不改。

        即使没过门槛，也记录「离门槛还有多远」；过了门槛就附上虚拟改动
        （virtual_promotion）。这样跑几个月，能离线验证 20 证据 / 90 天这组
        门槛是「稳」还是「死」，而不冒任何人格漂移的风险。
        """
        now = time.time() if now is None else now
        report: Dict[str, Dict[str, object]] = {}
        for trait, cell in self.shaped.items():
            n = len(cell.evidence_ids)
            if n == 0:
                continue
            stamps = [float(e.get("at") or 0.0) for e in cell.evidence_notes if e.get("at")]
            first = float(cell.first_evidence_at or (min(stamps) if stamps else now))
            span_days = (now - first) / 86400.0
            report[trait] = {
                "shaped_value": round(float(cell.value), 4),
                "base_value": round(self.base_value(trait), 4),
                "evidence_count": n,
                "span_days": round(span_days, 1),
                "evidence_left": max(0, BASE_PROMOTE_MIN_EVIDENCE - n),
                "span_days_left": round(max(0.0, BASE_PROMOTE_MIN_SPAN_DAYS - span_days), 1),
                "virtual_promotion": self._promotion_candidate(trait, now),
            }
        return report

    # ---------- 经验层 ----------
    def observe_trait(
        self,
        trait: str,
        *,
        delta: float,
        evidence_id: str,
        polarity: int = 1,
        note: str = "",
        memory_id: str = "",
        now: Optional[float] = None,
    ) -> Dict[str, object]:
        """给某个特质加一条证据。返回是否真的落地（及原因）。

        规则（对应 STATE_MACHINE 的证据门槛）：
        - 证据先累积，不立刻改 value；
        - 证据数 >= SHAPED_MIN_EVIDENCE 才算「够格」；
        - 每次改值最多走 alpha，且被基础层锚住（不会漂到反面）。

        note 是这条证据的「病历」——一句话说清它是什么事。
        没有它，经验层就只剩一个数，说不出自己是怎么来的。

        memory_id 指向支撑这句话的那条记忆：以后能点开看原句，
        记忆被纠正时也能顺着它把这条证据扣回去（reconcile_trait_evidence）。
        """
        now = time.time() if now is None else now
        cell = self.shaped.setdefault(trait, TraitCell(value=self.base_value(trait)))
        if evidence_id in cell.evidence_ids:
            return {"applied": False, "reason": "duplicate_evidence"}
        cell.evidence_ids.append(evidence_id)
        if not cell.first_evidence_at:
            cell.first_evidence_at = now
        # 每条证据都记一笔（不再只在有 note 时记）：delta/polarity 是扣账依据，
        # 缺了它们，记忆被纠正时就算知道该扣也扣不动。
        cell.evidence_notes.append(
            {
                "id": evidence_id,
                "note": str(note).strip()[:200],
                "at": now,
                "memory_id": memory_id or None,
                "delta": abs(float(delta)),
                "polarity": 1 if polarity >= 0 else -1,
            }
        )
        # 病历只留最近 EVIDENCE_NOTE_LIMIT 条：够 reconcile 用，又不会无限膨胀
        cell.evidence_notes = cell.evidence_notes[-EVIDENCE_NOTE_LIMIT:]

        if polarity >= 0:
            cell.support += abs(delta)
        else:
            cell.oppose += abs(delta)
        cell.last_verified_at = now

        if len(cell.evidence_ids) < SHAPED_MIN_EVIDENCE:
            cell.updated_at = now
            return {
                "applied": False,
                "reason": "evidence_below_threshold",
                "evidence_count": len(cell.evidence_ids),
                "needed": SHAPED_MIN_EVIDENCE,
            }

        target = _clamp01(cell.value + polarity * abs(delta))
        anchor = self.base_value(trait, cell.value)
        # 高锚定：目标值先向基础锚收缩一半，再以 alpha 步长靠近
        anchored_target = (target + anchor) / 2.0
        step = (anchored_target - cell.value) * SHAPED_UPDATE_ALPHA
        # 单次位移硬上限，防尖峰
        step = max(-0.05, min(0.05, step))

        cell.value = _clamp01(cell.value + step)
        cell.version += 1
        cell.updated_at = now
        self.last_shaped_write_at = now
        return {
            "applied": True,
            "reason": "shaped_update",
            "value": round(cell.value, 4),
            "step": round(step, 4),
            "evidence_count": len(cell.evidence_ids),
        }

    def trait_evidence(self, trait: str, *, limit: int = 10) -> Dict[str, Any]:
        """列出某个特质的证据；有 memory_id 的顺手把记忆摘要带出来（取不到就留空，别报错）。

        这是「我为什么好奇变强了」点得开的那一半：
        evidence_notes 里存的是当时的判断，memory_id 回查给出的是原句所在的记忆。
        摘要查不到不算失败——记忆可能被清理/服务暂时不通，证据本身仍然有效。
        """
        limit = max(1, min(100, int(limit or 10)))
        cell = self.shaped.get(trait)
        if cell is None:
            return {"trait": trait, "found": False, "items": [], "count": 0}
        items: List[Dict[str, Any]] = []
        # 最新的在前：想知道「最近是什么把我推成这样的」
        for e in list(reversed(cell.evidence_notes))[:limit]:
            mid = e.get("memory_id")
            summary = ""
            if mid and self.memory_lookup is not None:
                try:
                    summary = str(self.memory_lookup(str(mid)) or "")
                except Exception:  # noqa: BLE001 —— 回查失败只留空，不影响证据本身
                    summary = ""
            items.append(
                {
                    "id": e.get("id"),
                    "note": str(e.get("note", "")),
                    "at": e.get("at"),
                    "memory_id": mid,
                    "memory_summary": summary,
                    "delta": e.get("delta"),
                    "polarity": e.get("polarity"),
                    "invalidated": bool(e.get("invalidated")),
                }
            )
        return {
            "trait": trait,
            "found": True,
            "value": round(float(cell.value), 4),
            "support": round(float(cell.support), 4),
            "oppose": round(float(cell.oppose), 4),
            "version": cell.version,
            "evidence_count": len(cell.evidence_ids),
            "invalidated_count": len(cell.invalidated_evidence),
            "items": items,
            "count": len(items),
        }

    def reconcile_trait_evidence(self, memory_id: str) -> Dict[str, Any]:
        """一条记忆被纠正了：把引用它的证据标失效，并从 support/oppose 里扣掉。

        返回被影响的 trait 列表和扣掉的量。

        为什么不回滚 value：值是「历史累积到这里的结果」，扣账只该让它
        「不再有证据往前推」，不该假装那段时间没发生过。证据数掉到 3 以下
        也一样——门槛只挡新证据，不审旧账。

        找不到 delta 的老证据（2026-11-12 之前记的）没法精确扣：
        记进 unknown_delta 让它可见，而不是猜一个数扣下去。
        """
        now = time.time()
        mid = str(memory_id or "").strip()
        out: Dict[str, Any] = {
            "memory_id": mid,
            "affected": [],
            "deducted": {},
            "invalidated_evidence": [],
            "unknown_delta": [],
        }
        if not mid:
            return out
        touched = False
        for trait, cell in self.shaped.items():
            hits = [
                e
                for e in cell.evidence_notes
                if str(e.get("memory_id") or "") == mid and not e.get("invalidated")
            ]
            if not hits:
                continue
            touched = True
            amount = 0.0
            for e in hits:
                e["invalidated"] = True
                e["invalidated_at"] = now
                ev_id = str(e.get("id") or "")
                delta = e.get("delta")
                polarity = e.get("polarity")
                if isinstance(delta, (int, float)) and polarity is not None:
                    d = abs(float(delta))
                    if int(polarity) >= 0:
                        cell.support = max(0.0, cell.support - d)
                    else:
                        cell.oppose = max(0.0, cell.oppose - d)
                    amount += d
                elif ev_id:
                    out["unknown_delta"].append(ev_id)
                if ev_id:
                    out["invalidated_evidence"].append(ev_id)
                    # 从「还活着的证据」里移走，但留在 invalidated_evidence 里可追溯
                    if ev_id in cell.evidence_ids:
                        cell.evidence_ids.remove(ev_id)
                    if ev_id not in cell.invalidated_evidence:
                        cell.invalidated_evidence.append(ev_id)
            out["affected"].append(trait)
            out["deducted"][trait] = round(amount, 4)
            cell.version += 1
            cell.updated_at = now
        if touched:
            self.last_shaped_write_at = now
        return out

    # ---------- 动态层 ----------
    def release_emotion(
        self,
        releases: Dict[str, float],
        *,
        relaxes: Optional[Dict[str, float]] = None,
        fatigue_delta: float = 0.0,
    ) -> None:
        """情绪事件只改分子。疲劳不从这里来——见 add_fatigue。

        从前 fatigue_delta 默认 0.05，等于「每命中一个情绪词就累一点」：
        纯技术对话一整天零疲劳，一句「晚安 mua」反而累。情绪和精力是两件事。
        """
        self.dynamic.tick()
        self.dynamic.endocrine.release_many(releases)
        if relaxes:
            self.dynamic.endocrine.relax_many(relaxes)
        if fatigue_delta:
            self.dynamic.fatigue = _clamp01(self.dynamic.fatigue + float(fatigue_delta))

    def add_fatigue(self, delta: float) -> None:
        """交互带来的疲劳：和情绪无关——聊天本身就会累，不是只有说出「累」字才累。"""
        self.dynamic.tick()
        self.dynamic.fatigue = _clamp01(self.dynamic.fatigue + max(0.0, float(delta)))

    def effects(self) -> Dict[str, object]:
        self.dynamic.tick()
        return self.dynamic.endocrine.effects()

    # ---------- 持久化 ----------
    def to_state_dict(self) -> Dict[str, object]:
        """完整状态（与 snapshot 不同：这是可恢复的原始值，不是展示用）。"""
        self.dynamic.tick()
        return {
            "base": self.base,
            "shaped": {
                k: {
                    "value": v.value,
                    "support": v.support,
                    "oppose": v.oppose,
                    "evidence_ids": list(v.evidence_ids),
                    "evidence_notes": list(v.evidence_notes),
                    "invalidated_evidence": list(v.invalidated_evidence),
                    "first_evidence_at": v.first_evidence_at,
                    "updated_at": v.updated_at,
                    "last_verified_at": v.last_verified_at,
                    "version": v.version,
                }
                for k, v in self.shaped.items()
            },
            "dynamic": {
                "endocrine": self.dynamic.endocrine.to_list(),
                "last_tick_at": self.dynamic.last_tick_at,
                "fatigue": self.dynamic.fatigue,
            },
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, object]) -> "PersonaState":
        """从磁盘恢复。停机期间的时间照常流逝（睡一觉，体温回基线）。"""
        state = cls.from_base_profile(dict(data.get("base", {}) or {}))  # type: ignore[arg-type]
        for name, cell in dict(data.get("shaped", {}) or {}).items():  # type: ignore[arg-type]
            # 老版本的行没有 memory_id / delta / polarity：用 .get 补 None，
            # 读到就兼容，读不到也不报错（reconcile 会把它们记进 unknown_delta）。
            notes: List[Dict[str, Any]] = []
            for e in list(cell.get("evidence_notes", []) or []):
                if not isinstance(e, dict):
                    continue
                notes.append({**e, "memory_id": e.get("memory_id")})
            state.shaped[name] = TraitCell(
                value=float(cell.get("value", 0.5)),
                support=float(cell.get("support", 0.0)),
                oppose=float(cell.get("oppose", 0.0)),
                evidence_ids=list(cell.get("evidence_ids", [])),
                evidence_notes=notes,
                invalidated_evidence=list(cell.get("invalidated_evidence", []) or []),
                first_evidence_at=float(cell.get("first_evidence_at", 0.0)),
                updated_at=float(cell.get("updated_at", 0.0)),
                last_verified_at=float(cell.get("last_verified_at", 0.0)),
                version=int(cell.get("version", 1)),
            )
        dyn = dict(data.get("dynamic", {}) or {})  # type: ignore[arg-type]
        state.dynamic = DynamicState(
            endocrine=EndocrineSystem(dyn.get("endocrine")),  # type: ignore[arg-type]
            last_tick_at=float(dyn.get("last_tick_at", time.time())),
            fatigue=float(dyn.get("fatigue", 0.0)),
        )
        state.dynamic.tick()
        return state

    # ---------- 对外快照 ----------
    def snapshot(self) -> Dict[str, object]:
        self.dynamic.tick()
        return {
            "base": self.base,
            "shaped": {k: v.to_dict() for k, v in self.shaped.items()},
            "dynamic": {
                "fatigue": round(self.dynamic.fatigue, 4),
                "effects": {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in self.effects().items()
                    if k != "emotion"
                },
                "emotion": {
                    k: round(v, 4)
                    for k, v in self.effects()["emotion"].items()  # type: ignore[index]
                },
                "molecules": self.dynamic.endocrine.snapshot(),
            },
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
