/**
 * dsh-suyi 插件测试。
 *
 * 捕获逻辑用 mock fetch 拦截，不污染真实记忆库；
 * 测试时关掉 auto_start，避免测试进程去拉起真实服务。
 * 最后用真实服务验一次连通。
 */
import assert from 'node:assert';

const mod = await import(new URL('../src/index.js', import.meta.url).href);
const boot = await import(new URL('../src/bootstrap.js', import.meta.url).href);

let pass = 0;
let fail = 0;
function check(name, fn) {
  try {
    fn();
    pass += 1;
    console.log(`  PASS ${name}`);
  } catch (e) {
    fail += 1;
    console.log(`  FAIL ${name}: ${e.message}`);
  }
}
async function checkAsync(name, fn) {
  try {
    await fn();
    pass += 1;
    console.log(`  PASS ${name}`);
  } catch (e) {
    fail += 1;
    console.log(`  FAIL ${name}: ${e.message}`);
  }
}

console.log('[module]');
check('exports name', () => assert.equal(mod.name, 'dsh-suyi'));
check('injects tools+systemPrompt', () => assert.deepEqual(mod.inject, ['tools', 'systemPrompt']));
check('exports apply', () => assert.equal(typeof mod.apply, 'function'));

// ---------------- 注册 ----------------
console.log('[registration]');
const registered = [];
const handlers = {};
const toolsService = {
  register: (t) => {
    registered.push(t);
    return () => {};
  },
};
const promptCalls = [];
const systemPromptService = {
  variable: (name, fn) => {
    promptCalls.push({ kind: 'variable', name, fn });
    return () => {};
  },
  section: (s) => {
    promptCalls.push({ kind: 'section', ...s });
    return () => {};
  },
};
const ctx = {
  get: (k) => {
    if (k === 'tools') return toolsService;
    if (k === 'systemPrompt') return systemPromptService;
    if (k === 'config') return { 'dsh-suyi': { auto_start: false } }; // 测试不拉起真服务
    return {};
  },
  effect: (fn) => fn(),
  on: (evt, cb) => {
    handlers[evt] = cb;
    return () => {};
  },
};

const dispose = mod.apply(ctx);

// 工具清单会随功能增长，断言「关键工具都在」+「数量合理」，
// 别写死具体数字——写死了每加一个功能就得改测试（2026-11-14 之前就卡在 6）。
check('registers a reasonable number of tools', () => assert.ok(registered.length >= 10, `实际 ${registered.length}`));
check('core tools are present', () => {
  const names = registered.map((t) => t.name);
  for (const required of ['su_recall', 'su_remember', 'su_forget', 'su_health', 'su_persona']) {
    assert.ok(names.includes(required), `缺少 ${required}`);
  }
});
check('growth tools are present', () => {
  const names = registered.map((t) => t.name);
  for (const required of ['su_evidence', 'su_gap', 'su_interest', 'su_interest_note']) {
    assert.ok(names.includes(required), `缺少 ${required}`);
  }
});
check('no duplicate tool names', () => {
  const names = registered.map((t) => t.name);
  assert.equal(new Set(names).size, names.length);
});
check('every tool has description', () => assert.ok(registered.every((t) => (t.description || '').length > 5)));
check('every tool has execute', () => assert.ok(registered.every((t) => typeof t.execute === 'function')));
check('every tool has output render', () => assert.ok(registered.every((t) => typeof t.output?.render === 'function')));
check('parameters are JSON Schema objects', () => assert.ok(registered.every((t) => t.parameters?.type === 'object')));
check('hooks session/event', () => assert.equal(typeof handlers['session/event'], 'function'));
check('hooks session/flush', () => assert.equal(typeof handlers['session/flush'], 'function'));
check('registers state variable', () =>
  assert.ok(promptCalls.some((c) => c.kind === 'variable' && c.name === 'suyi_state')));
check('registers state section', () =>
  assert.ok(
    promptCalls.some((c) => c.kind === 'section' && c.name === 'suyi-state' && c.text === '{{suyi_state}}'),
  ));
check('registers intro variable', () =>
  assert.ok(promptCalls.some((c) => c.kind === 'variable' && c.name === 'suyi_recent')));

// ---------------- 节奏监控 ----------------
// 数据考古的结论：思考退化的强因子是「连续失败的重试」，不是注入量。
// 所以这里验：连续失败到阈值才出现提示、成功一次就清零。
console.log('[pace]');
const paceVar = promptCalls.find((c) => c.kind === 'variable' && c.name === 'suyi_pace');
check('registers pace variable', () => assert.ok(paceVar));
check('registers pace section', () =>
  assert.ok(promptCalls.some((c) => c.kind === 'section' && c.name === 'suyi-pace' && c.order === 10270)));

const fakeHeader = { version: 4, id: 'pace-test', createdAt: '', isSeeded: false, delegationDepth: 0 };
const fakeSession = { id: 'pace-test', header: fakeHeader };
const fakeCtx = { scope: { session: { header: fakeHeader } } };
const renderPaceNow = () => paceVar.fn(fakeCtx);
const fireTool = (isError) =>
  handlers['session/event'](fakeSession, {
    type: 'tool/result',
    data: { message: { isError, content: [{ type: 'text', text: 'x' }] } },
  });

check('pace silent before any failure', () => assert.equal(renderPaceNow(), ''));
fireTool(true);
check('pace still silent after 1 failure (threshold=2)', () => assert.equal(renderPaceNow(), ''));
fireTool(true);
check('pace appears after 2 consecutive failures', () => assert.ok(renderPaceNow().includes('连续失败 2 次')));
fireTool(true);
check('pace counts the 3rd failure', () => assert.ok(renderPaceNow().includes('连续失败 3 次')));
fireTool(false);
check('pace clears after a success', () => assert.equal(renderPaceNow(), ''));
check('other sessions unaffected', () => {
  const other = { scope: { session: { header: { ...fakeHeader, id: 'other-session' } } } };
  fireTool(true);
  fireTool(true);
  return assert.ok(!paceVar.fn(other).includes('连续失败'));
});
check('registers intro section', () =>
  assert.ok(
    promptCalls.some((c) => c.kind === 'section' && c.name === 'suyi-recent' && c.text === '{{suyi_recent}}'),
  ));

// ---------------- 主动提起 ----------------
console.log('[proactive render]');
check('recall renders summary + reason', () => {
  const s = mod.renderRecall([{ summary: '用户说想去看海', reason: '2 天前你说的' }]);
  assert.ok(s.includes('用户说想去看海'), s);
  assert.ok(s.includes('2 天前你说的'), s);
  assert.ok(s.includes('不必硬提'), s);
});
check('recall is empty without items', () => assert.equal(mod.renderRecall([]), ''));
check('recall skips empty summaries', () => assert.equal(mod.renderRecall([{ summary: '' }]), ''));
check('registers recall variable', () =>
  assert.ok(promptCalls.some((c) => c.kind === 'variable' && c.name === 'suyi_recall')));
check('registers recall section', () =>
  assert.ok(
    promptCalls.some((c) => c.kind === 'section' && c.name === 'suyi-recall' && c.text === '{{suyi_recall}}'),
  ));

// ---------------- 状态文案 ----------------
console.log('[mood render]');
check('calm state is silent', () => {
  const calm = {
    dynamic: { fatigue: 0.0, effects: { arousal: 0.08, valence: 0.1 }, emotion: { anxiety: 0.5, sadness: 0.5 } },
  };
  assert.equal(mod.renderMood(calm), '');
});
check('tired adds fatigue cue', () => {
  const t = { dynamic: { fatigue: 0.7, effects: { arousal: 0.1, valence: 0.1 }, emotion: {} } };
  assert.ok(mod.renderMood(t).includes('疲倦'), mod.renderMood(t));
});
check('anxious adds arousal cue', () => {
  const a = { dynamic: { fatigue: 0.1, effects: { arousal: 0.7, valence: -0.3 }, emotion: { anxiety: 1.0 } } };
  const s = mod.renderMood(a);
  assert.ok(s.includes('焦虑'), s);
  assert.ok(s.includes('心跳偏快'), s);
});
check('mood text leaks no numbers', () => {
  const a = { dynamic: { fatigue: 0.7, effects: { arousal: 0.7, valence: -0.3 }, emotion: { anxiety: 1.0 } } };
  assert.ok(!/\d\.\d/.test(mod.renderMood(a)), mod.renderMood(a));
});

// ---------------- 开场前情 ----------------
console.log('[intro render]');
check('intro picks active only', () => {
  const s = mod.renderIntro([
    { status: 'active', content: '第一件事' },
    { status: 'candidate', content: '候选不该出现' },
    { status: 'active', content: '对话片段：摘要不该出现' },
    { status: 'active', content: '第二件事' },
  ]);
  assert.ok(s.includes('第一件事'));
  assert.ok(s.includes('第二件事'));
  assert.ok(!s.includes('候选不该出现'));
  assert.ok(!s.includes('摘要不该出现'));
  assert.ok(s.includes('我们之前聊过这些'));
});
check('intro empty when nothing', () => assert.equal(mod.renderIntro([]), ''));
check('intro caps each line', () => {
  const long = '啊'.repeat(200);
  const s = mod.renderIntro([{ status: 'active', content: long }]);
  assert.ok(s.length < 120, String(s.length));
});

// ---------------- 自动捕获（mock fetch） ----------------
console.log('[auto capture]');
const calls = [];
const realFetch = globalThis.fetch;
globalThis.fetch = async (url, opts) => {
  calls.push({ url: String(url), body: opts?.body ? JSON.parse(opts.body) : null });
  return new Response(JSON.stringify({ admitted: true, memory_id: 'mem_x' }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
};

const session = { id: 'sess_test_1' };
const emit = (evt) => handlers['session/event'](session, evt);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// turn/end 现在还会推一次 /promote，所以只数 /observe
const observes = () => calls.filter((c) => c.url.includes('/observe'));

emit({ type: 'user/message', seq: 1, time: Date.now(), data: { role: 'user', content: '用户说这周想去看海' } });
await sleep(120);
check('captures user message', () => assert.equal(observes().length, 1));
check('user tier engage', () => assert.equal(observes()[0].body.tier, 'engage'));

calls.length = 0;
emit({ type: 'user/message', seq: 2, time: Date.now(), data: { role: 'user', content: '嗯' } });
await sleep(80);
check('skips too-short message', () => assert.equal(observes().length, 0));

calls.length = 0;
emit({
  type: 'assistant/message',
  seq: 3,
  time: Date.now(),
  data: { message: { role: 'assistant', content: '好呀，这周我们就去看海' } },
});
await sleep(60);
check('does not capture assistant until turn end', () => assert.equal(observes().length, 0));
emit({ type: 'turn/end', seq: 4, time: Date.now(), data: {} });
await sleep(120);
check('captures final assistant at turn end', () => assert.equal(observes().length, 1));
check('assistant not self-statement stays attention', () => assert.equal(observes()[0].body.tier, 'attention'));
check('turn end pushes promotion', () => assert.ok(calls.some((c) => c.url.includes('/promote'))));

calls.length = 0;
emit({
  type: 'assistant/message',
  seq: 5,
  time: Date.now(),
  data: {
    message: {
      role: 'assistant',
      content: [
        { type: 'reasoning', text: '这是隐藏推理，不该被记住' },
        { type: 'text', text: '我记下了，明天我们一起去看海。' },
      ],
    },
  },
});
emit({ type: 'turn/end', seq: 6, time: Date.now(), data: {} });
await sleep(120);
check('never stores hidden reasoning', () =>
  assert.ok(!String(observes()[0]?.body?.text || '').includes('隐藏推理')));
check('stores visible part', () =>
  assert.ok(String(observes()[0]?.body?.text || '').includes('这是可见回答') || String(observes()[0]?.body?.text || '').includes('我记下了')));
check('self statement upgrades to engage', () => assert.equal(observes()[0].body.tier, 'engage'));

globalThis.fetch = realFetch;
check('disposer callable', () => assert.equal(typeof dispose, 'function'));

// ---------------- 工具错误不炸 ----------------
console.log('[tool execute]');
globalThis.fetch = async () => new Response('{"ok":false}', { status: 500 });
const res = await registered.find((t) => t.name === 'su_health').execute({}, {});
check('tool error returned not thrown', () => assert.ok(res && typeof res === 'object' && 'error' in res));
globalThis.fetch = realFetch;

// ---------------- 服务守护 ----------------
console.log('[bootstrap]');
check('builds 2 services', () => assert.equal(boot.buildServices({}).length, 2));
const svcDefs = boot.buildServices({});
check('mnemosyne def', () => {
  const m = svcDefs.find((s) => s.name === 'mnemosyne');
  assert.ok(m.args.includes('-m') && m.args.includes('mnemosyne.main'));
  assert.ok(String(m.health).includes('8765'));
});
check('persona def', () => {
  const p = svcDefs.find((s) => s.name === 'persona-memory');
  assert.ok(p.args.includes('persona_memory.main'));
  assert.ok(String(p.health).includes('8772'));
});
check('paths point at workspace', () => {
  const d = boot.defaultPaths();
  assert.ok(d.python.includes('.venv'));
  assert.ok(d.personaRoot.endsWith('persona-memory'));
});
check('config can override paths', () => {
  const defs = boot.buildServices({ python: 'C:/x/python.exe', persona_root: 'C:/y' });
  assert.equal(defs[1].command, 'C:/x/python.exe');
  assert.equal(defs[1].cwd, 'C:/y');
});
await checkAsync('isHealthy false for dead port', async () => {
  assert.equal(await boot.isHealthy('http://127.0.0.1:59999/health', 600), false);
});

// ---------------- Fast Path ----------------
console.log('[fast-path]');
const fp = await import(new URL('../src/fast-path.js', import.meta.url).href);

check('ack: 嗯', () => assert.equal(fp.isAckLike('嗯'), true));
check('ack: 好的好的', () => assert.equal(fp.isAckLike('好的好的'), true));
check('ack: 收到！', () => assert.equal(fp.isAckLike('收到！'), true));
check('ack: 真内容不算语气词', () => assert.equal(fp.isAckLike('今天好累啊，想早点睡'), false));
check('echo fragment detected', () => assert.equal(fp.isEchoFragment('对话片段：您好'), true));
check('normal text is not echo', () => assert.equal(fp.isEchoFragment('普通消息'), false));

check('deduper: first is not duplicate', () => {
  const d = fp.createDeduper();
  assert.equal(d('我想去看海'), false);
});
check('deduper: same text (whitespace-insensitive) is duplicate', () => {
  const d = fp.createDeduper();
  d('我想去看海');
  assert.equal(d('我想 去看海'), true);
});
check('deduper: zero window never duplicates', () => {
  const d = fp.createDeduper({ windowMs: 0 });
  d('a');
  assert.equal(d('a'), false);
});

check('shouldSkip: ack', () => assert.equal(fp.shouldSkipCapture('好的', null), 'ack'));
check('shouldSkip: echo', () => assert.equal(fp.shouldSkipCapture('对话片段：x', null), 'echo_fragment'));
check('shouldSkip: duplicate', () => {
  const d = fp.createDeduper();
  d('同一句话');
  assert.equal(fp.shouldSkipCapture('同一句话', d), 'duplicate');
});
check('shouldSkip: real content passes', () => {
  const d = fp.createDeduper();
  assert.equal(fp.shouldSkipCapture('用户说这周想去看海，他最近太累了', d), '');
});

// ---------------- 真实服务连通 ----------------
console.log('[live service]');
const { SuyiClient } = await import(new URL('../src/client.js', import.meta.url).href);
const client = new SuyiClient({ baseUrl: process.env.SUYI_TEST_BASE || 'http://127.0.0.1:8772' });
try {
  const h = await client.health();
  check('live health ok', () => assert.equal(h.ok, true));
  const p = await client.persona();
  check('live persona 3 layers', () => assert.deepEqual(Object.keys(p).sort(), ['base', 'dynamic', 'shaped']));
  const r = await client.retrieve({ query: '看海', limit: 3 });
  check('live recall has provenance', () => assert.ok(r.items.every((i) => i.source)));
  await checkAsync('live bootstrap reports already-running', async () => {
    const results = await boot.ensureAll({});
    assert.ok(
      results.every((x) => x.state === 'already-running'),
      JSON.stringify(results),
    );
  });
} catch (e) {
  console.log(`  SKIP live service (${e.message})`);
}

console.log(`\n${fail ? `RESULT: ${fail} FAILED` : 'RESULT: ALL PASS'} (${pass} passed)`);
process.exit(fail ? 1 : 0);
