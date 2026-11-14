// 工作台交接（Workbench Handoff）
//
// 背景：会话一长，我"脑子里"正在做的那件事会模糊——不是记忆丢了（溯忆都记着），
// 是**工作台被清空了**：变量叫啥、改到第几行、为什么选 A 不选 B，这些只活在上下文里。
//
// 设计参考了 DSH 生态里几个成熟插件，只取核心：
//   · dsh-handoff-session：压力投影 + 分级阈值 + 七节交接文档
//   · dsh-auto-handoff：简报的九节结构（目标/约束/已完成/进度/改的文件/决策/问题/未完成/下一步）
//   · dsh-context-rollover：笔记+转录旁路，缺细节再 grep
//
// **故意砍掉的**（为了不和 DSH 内部耦合）：
//   · 自动创建新会话（要调 sessionController，那是内部 API）
//   · 子智能体迁移（逻辑重，我们子代理用得不多）
//
// 阈值为什么这么定：**不按"上下文占用"**。
// `request/context` 只给 contextWindow（1M），拿不到实时占用；而且数据上
// 1M 窗口用 4% 的时候就已经在退化了（见 `_inspect/degeneration-report.md`：
// 127 轮的会话退化最重）。所以按**注意力衰减**定：watch 40 轮 / recommended 70 / urgent 100。

import fs from 'node:fs';
import path from 'node:path';

export const LEVELS = ['low', 'watch', 'recommended', 'urgent'];

export const DEFAULT_THRESHOLDS = {
  // 轮数：数据里 127 轮退化最重，所以 watch 40 就开始提醒
  turns: [40, 70, 100],
  // 步数：一步 = 一次模型请求，长任务里涨得比轮数快
  steps: [200, 400, 700],
  // 累计正文估算 token 的字符数（中文约 1 字 ≈ 1 token，粗算 chars/2.5）
  chars: [150000, 300000, 500000],
};

export function createPressureState() {
  return {
    turn: 0,
    step: 0,
    chars: 0,
    // 交接基线：写完后把当前值记成基线，压力从 0 重新算——
    // 否则"交接完还在被催"（turn 是会话级的，不会因为交接而变小）。
    baselineTurn: 0,
    baselineStep: 0,
    baselineChars: 0,
    // 最近一次 request/context 里看到的模型与窗口，仅用于展示
    model: '',
    contextWindow: 0,
    startedAt: Date.now(),
  };
}

/** 从一条会话事件里更新压力状态（纯函数，就地改 state，返回 state）。 */
export function trackEvent(state, event) {
  if (!state || !event) return state;
  const type = event.type;
  const data = event.data || {};
  if (type === 'turn/start') {
    const t = Number(data.turn);
    if (Number.isFinite(t) && t > state.turn) state.turn = t;
  } else if (type === 'step/start') {
    // step 是单调递增的，直接取
    const s = Number(data.step);
    if (Number.isFinite(s) && s > state.step) state.step = s;
    const t = Number(data.turn);
    if (Number.isFinite(t) && t > state.turn) state.turn = t;
  } else if (type === 'request/context') {
    if (data.model) state.model = String(data.model);
    if (Number.isFinite(Number(data.contextWindow))) state.contextWindow = Number(data.contextWindow);
  } else if (type === 'assistant/message') {
    // 粗估上下文增长：assistant 正文 + 工具调用参数
    const text = messageText(data.message);
    state.chars += text.length;
    for (const tc of data.message?.toolCalls || data.toolCalls || []) {
      state.chars += String(tc?.arguments || '').length;
    }
  } else if (type === 'tool/result') {
    state.chars += String(messageText(data.message)).length;
  }
  return state;
}

function messageText(message) {
  if (!message) return '';
  const c = message.content;
  if (typeof c === 'string') return c;
  if (!Array.isArray(c)) return '';
  let out = '';
  for (const part of c) {
    if (!part) continue;
    if (typeof part === 'string') out += part;
    else if (typeof part.text === 'string') out += part.text;
    else if (part.type === 'tool_call' || part.type === 'tool_use') out += JSON.stringify(part);
  }
  return out;
}

/** 把「多少轮 / 多少步 / 多少字」折算成 0~3 的档位（-1 表示还没到第一档）。 */
function levelIndex(value, thresholds) {
  if (!Array.isArray(thresholds) || thresholds.length === 0) return -1;
  let idx = -1;
  for (let i = 0; i < thresholds.length; i++) {
    if (value >= thresholds[i]) idx = i;
  }
  return idx;
}

/**
 * 当前压力档位。取三个维度里最高的那个——任何一项超标都算超标。
 * 返回 { index, level, reasons: [...] }
 */
export function pressureOf(state, thresholds = DEFAULT_THRESHOLDS) {
  if (!state) return { index: -1, level: 'low', reasons: [] };
  // 减去交接基线：写完交接文档后，压力从零重新累积
  const turn = Math.max(0, (Number(state.turn) || 0) - (Number(state.baselineTurn) || 0));
  const step = Math.max(0, (Number(state.step) || 0) - (Number(state.baselineStep) || 0));
  const chars = Math.max(0, (Number(state.chars) || 0) - (Number(state.baselineChars) || 0));
  const dims = [
    ['轮数', turn, thresholds.turns],
    ['步数', step, thresholds.steps],
    ['累计字数', chars, thresholds.chars],
  ];
  let best = -1;
  const reasons = [];
  for (const [name, value, thr] of dims) {
    const idx = levelIndex(value, thr);
    if (idx >= 0) reasons.push(`${name} ${value}（阈值 ${thr.join('/')}）`);
    if (idx > best) best = idx;
  }
  return {
    index: best,
    level: best < 0 ? 'low' : LEVELS[Math.min(best + 1, LEVELS.length - 1)],
    reasons,
  };
}

/** 给模型看的一段提示。low 档返回空——别在健康时占 token。 */
export function renderPressure(state, thresholds = DEFAULT_THRESHOLDS) {
  const p = pressureOf(state, thresholds);
  if (p.index < 0) return '';
  const tips = {
    watch: '留意一下：如果你发现自己在原地打转，可以考虑把当前进度写成交接文档，然后换个新会话接着干。',
    recommended: '建议交接了。现在多半是个干活的好时机——写一份交接文档，让新会话接着做。',
    urgent: '该交接了。再往下聊，我会开始记不清细节。先写交接文档，别硬撑。',
  };
  const turn = Math.max(0, (Number(state.turn) || 0) - (Number(state.baselineTurn) || 0));
  const step = Math.max(0, (Number(state.step) || 0) - (Number(state.baselineStep) || 0));
  const chars = Math.max(0, (Number(state.chars) || 0) - (Number(state.baselineChars) || 0));
  const lines = [
    `[工作台] ${p.level} · 当前 ${turn} 轮 / ${step} 步 / 约 ${Math.round(chars / 2.5 / 1000)}K 字`,
  ];
  if (p.reasons.length) lines.push(`触发：${p.reasons.join('、')}`);
  const tip = tips[p.level];
  if (tip) lines.push(tip);
  lines.push('交接文档用 su_handoff 写；新会话开场会自动读到它。');
  lines.push('');
  return lines.join('\n');
}

// ---------------- 交接文档 ----------------

export const HANDOFF_SECTIONS = [
  ['goal', '目标', '这件事最终要达成什么。'],
  ['status', '现状', '此刻做到哪一步了。'],
  ['done', '已完成', '已经落地、验证过的东西。'],
  ['todo', '待办', '还没做的，按优先级列。'],
  ['decisions', '关键决策', '为什么选了 A 没选 B——这是新会话最需要知道的。'],
  ['files', '关键文件', '改过或必须读的文件路径。'],
  ['verify', '验证方式', '怎么确认做对了。'],
];

/**
 * 生成交接文档正文。所有字段都可选，空的整节省略。
 * @param {object} input  { goal, status, done, todo, decisions, files, verify, extra }
 * @param {object} meta   { turn, step, chars, model, level, sessionId, createdAt }
 */
export function buildHandoffDoc(input = {}, meta = {}) {
  const lines = [];
  const when = new Date(meta.createdAt || Date.now()).toISOString().replace('T', ' ').slice(0, 19);
  lines.push(`# 工作台交接 · ${when}`);
  lines.push('');
  lines.push('> 这份文档是给**下一个会话的我**看的。它只有摘要，细节请按需去查。');
  lines.push('');
  if (meta.sessionId) lines.push(`- 来源会话：${meta.sessionId}`);
  if (meta.level) lines.push(`- 交接时压力：${meta.level}`);
  if (Number.isFinite(meta.turn)) lines.push(`- 轮次 / 步数：${meta.turn} / ${meta.step}`);
  if (meta.model) lines.push(`- 模型：${meta.model}`);
  lines.push('');

  for (const [key, title, hint] of HANDOFF_SECTIONS) {
    const value = normalizeSection(input[key]);
    if (!value) continue;
    lines.push(`## ${title}`);
    if (!value.includes('\n')) {
      lines.push(value);
    } else {
      lines.push(value);
    }
    lines.push('');
  }

  const extra = normalizeSection(input.extra);
  if (extra) {
    lines.push('## 其它');
    lines.push(extra);
    lines.push('');
  }
  return lines.join('\n');
}

function normalizeSection(value) {
  if (value == null) return '';
  if (Array.isArray(value)) {
    const items = value.map((v) => String(v || '').trim()).filter(Boolean);
    if (!items.length) return '';
    return items.map((v) => (v.startsWith('- ') || /^\d+\./.test(v) ? v : `- ${v}`)).join('\n');
  }
  return String(value).trim();
}

/** 写一份交接文档，返回文件路径。 */
export function saveHandoff(dir, doc, { sessionId = '', at = Date.now() } = {}) {
  fs.mkdirSync(dir, { recursive: true });
  const stamp = new Date(at).toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const safeSid = String(sessionId || 'session').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 16) || 'session';
  const file = path.join(dir, `handoff-${stamp}-${safeSid}.md`);
  fs.writeFileSync(file, doc, 'utf8');
  return file;
}

/** 读目录里最新一份交接文档（按文件名排序，带时间戳所以可靠）。 */
export function readLatestHandoff(dir) {
  try {
    if (!fs.existsSync(dir)) return null;
    const files = fs
      .readdirSync(dir)
      .filter((f) => f.startsWith('handoff-') && f.endsWith('.md') && !f.endsWith('.transcript.md'))
      .sort();
    if (!files.length) return null;
    const latest = files[files.length - 1];
    const full = path.join(dir, latest);
    const stat = fs.statSync(full);
    // 只认最近 7 天的；更早的交接对"接着干"没意义了
    if (Date.now() - stat.mtimeMs > 7 * 24 * 3600 * 1000) return null;
    return { file: full, name: latest, content: fs.readFileSync(full, 'utf8'), mtime: stat.mtimeMs };
  } catch {
    return null;
  }
}

/** 新会话开场用：把交接文档包成一段注入文本。 */
export function renderHandoffIntro(handoff, { maxChars = 4000 } = {}) {
  if (!handoff || !handoff.content) return '';
  let body = handoff.content.trim();
  if (body.length > maxChars) {
    body = body.slice(0, maxChars) + '\n\n…（后面截断了，需要细节就自己 read 这个文件）';
  }
  const ageMin = Math.max(0, Math.round((Date.now() - handoff.mtime) / 60000));
  const ago = ageMin < 60 ? `${ageMin} 分钟前` : `${Math.round(ageMin / 60)} 小时前`;
  return [
    `[工作台交接] ${ago}你写下过这份交接，现在接着做：`,
    '',
    body,
    '',
    `（原文在 ${handoff.name}，要看细节自己 read。）`,
    '',
  ].join('\n');
}
