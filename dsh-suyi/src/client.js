/**
 * 溯忆 · Anamnesis —— 与本地 persona-memory 服务通信的 HTTP 客户端。
 * 不引入任何第三方依赖：Node 18+ 自带 fetch。
 */

const DEFAULT_BASE = 'http://127.0.0.1:8772';

export class SuyiError extends Error {
  constructor(method, path, status, body) {
    super(`suyi http ${status} ${method} ${path}: ${typeof body === 'string' ? body : JSON.stringify(body)}`);
    this.status = status;
    this.body = body;
  }
}

export class SuyiClient {
  constructor({ baseUrl = DEFAULT_BASE, timeout = 12000 } = {}) {
    this.baseUrl = String(baseUrl || DEFAULT_BASE).replace(/\/+$/, '');
    this.timeout = timeout;
  }

  async _req(method, path, body) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeout);
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: ctrl.signal,
      });
      const text = await res.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = text;
      }
      if (!res.ok) throw new SuyiError(method, path, res.status, data);
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  health() {
    return this._req('GET', '/health');
  }

  observe({ text, sourceKind = 'dialog', tier = 'attention', strength = 0.6, metadata }) {
    return this._req('POST', '/observe', {
      text,
      source_kind: sourceKind,
      tier,
      strength,
      metadata: metadata || {},
    });
  }

  retrieve({ query, limit = 8, tokenBudget, detail = false, sinceHours = 0, cluster = '' }) {
    const body = { query, limit, detail };
    if (tokenBudget) body.token_budget = tokenBudget;
    if (sinceHours) body.since_hours = sinceHours;
    if (cluster) body.cluster = cluster;
    return this._req('POST', '/retrieve', body);
  }

  expand({ memoryId, depth = 1 }) {
    return this._req('POST', '/expand', { memory_id: memoryId, depth });
  }

  /** 主动提起：挑一件此刻值得想起来的旧事（会消耗该条的冷却期）。 */
  proactive({ limit = 1 } = {}) {
    return this._req('GET', `/proactive?limit=${encodeURIComponent(limit)}`);
  }

  /** 轻量自检：索引 / 元数据 / 链接覆盖 / 重复。 */
  selfcheck() {
    return this._req('GET', '/selfcheck');
  }

  persona() {
    return this._req('GET', '/persona');
  }

  /** 在场感：上次有互动是多久以前（只有间隔，没有时间戳）。 */
  presence() {
    return this._req('GET', '/presence');
  }

  /** 他喜欢什么（只收反复出现、且不是太久以前的事）。 */
  interests({ limit = 20 } = {}) {
    return this._req('GET', `/interests?limit=${encodeURIComponent(limit)}`);
  }

  addInterest({ topic, example = '' }) {
    return this._req('POST', '/interest', { topic, example });
  }

  /**
   * 记一条弱证据：他顺口提到某样东西。
   * 这不是确认——单条永远不会把话题推成「他喜欢X」；跨天攒够了服务端自己会判。
   */
  interestSignal({ topic, polarity = 1, tier = 'passive', quote = '', confidence = 0.5 }) {
    return this._req('POST', '/interest/signal', { topic, polarity, tier, quote, confidence });
  }

  /** 常驻小画像（≤400 字）：已确认兴趣 / 最绷的执念 / 最近翻过的车。 */
  profile() {
    return this._req('GET', '/profile');
  }

  /** 我最近翻过的车：最近 maxDays 天内、最多 limit 条（服务端已截断正文）。 */
  failures({ limit = 5, maxDays = 7 } = {}) {
    return this._req(
      'GET',
      `/failures?limit=${encodeURIComponent(limit)}&max_days=${encodeURIComponent(maxDays)}`,
    );
  }

  addFailure({ text, reason = '', correctedTo = '', category = '其它' }) {
    return this._req('POST', '/failure', {
      text,
      reason,
      corrected_to: correctedTo,
      category,
    });
  }

  /** 缺口池：未解执念。张力是服务端惰性算出来的（停机期间也照样在绷）。 */
  gaps({ status, limit = 50 } = {}) {
    const q = new URLSearchParams();
    if (status) q.set('status', status);
    q.set('limit', String(limit));
    return this._req('GET', `/gaps?${q.toString()}`);
  }

  addGap({ text, strength = 1, source = 'manual', note = '' }) {
    return this._req('POST', '/gap', { text, strength, source, note });
  }

  closeGap({ id, closedBy = '', note = '' }) {
    return this._req('POST', '/gap/close', { id, closed_by: closedBy, note });
  }

  emotion({ releases }) {
    return this._req('POST', '/persona/emotion', { releases: releases || {} });
  }

  /**
   * 给「经验塑造」层投一条证据。
   * 服务端自己守门槛：≥3 条独立证据才动值、单次位移 ≤0.05、目标先向基础锚收缩。
   * 同一 evidence_id 重复投会被拒（duplicate_evidence），不会重复记账。
   * memoryId 指向支撑这条证据的记忆：以后能点开看原句，记忆被纠正时也会扣回去。
   */
  traitEvidence({ trait, delta = 0.2, polarity = 1, evidenceId, note = '', memoryId = '' }) {
    return this._req('POST', '/persona/trait', {
      trait,
      delta,
      polarity,
      evidence_id: evidenceId,
      note,
      memory_id: memoryId || null,
    });
  }

  correct({ memoryId, note = '' }) {
    return this._req('POST', '/feedback', { memory_id: memoryId, note });
  }

  promote({ limit = 50 } = {}) {
    return this._req('POST', '/promote', { limit });
  }

  /** 列最近的记忆（给「开场前情」用）。 */
  memories({ limit = 40 } = {}) {
    return this._req('GET', `/memories?limit=${encodeURIComponent(limit)}`);
  }
}
