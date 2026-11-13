# -*- coding: utf-8 -*-
"""溯忆看板：一页看完「它记住了什么、现在是什么心情」。

不是后台管理界面，是「掀开被子看一眼」的地方：
- 左：此刻的我（人格三层 + 15 种分子的浓度 + 情绪 12 维）
- 右：记忆（状态徽章、来源、时间，一眼看到哪些还只是候选）
- 顶：一行统计，告诉你它长了多少

单文件、无外部依赖，5 秒自动刷新。
"""

from __future__ import annotations

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>溯忆</title>
<style>
  :root{
    --bg:#14100f; --card:#1e1917; --card2:#251f1c; --line:#332b27;
    --text:#ece3dc; --muted:#9c8e86; --dim:#6f625b;
    --warm:#ff9a6c; --warm2:#ffc46b; --pink:#ff8fa3;
    --ok:#7fd1a0; --wait:#e8c268; --dead:#6f625b;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:
      radial-gradient(900px 500px at 15% -10%, #2a1d18 0%, transparent 60%),
      radial-gradient(700px 400px at 110% 10%, #241a20 0%, transparent 55%),
      var(--bg);
    color:var(--text);
    font:15px/1.65 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1180px;margin:0 auto;padding:28px 22px 60px}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
  h1{font-size:24px;margin:0;letter-spacing:.5px;font-weight:650}
  h1 small{font-size:12.5px;color:var(--dim);font-weight:400;letter-spacing:1px;margin-left:10px}
  .sub{color:var(--muted);font-size:13px}
  .stats{display:flex;gap:18px;margin:16px 0 24px;flex-wrap:wrap}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px 16px;min-width:110px}
  .stat b{display:block;font-size:22px;line-height:1.2;color:var(--warm2);font-variant-numeric:tabular-nums}
  .stat span{font-size:12px;color:var(--muted)}
  .grid{display:grid;grid-template-columns:340px 1fr;gap:22px;align-items:start}
  @media (max-width:880px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
  .card + .card{margin-top:18px}
  .card h2{font-size:14px;margin:0 0 14px;color:var(--warm);font-weight:600;letter-spacing:1px}
  .mood{font-size:13px;color:var(--muted);margin-bottom:14px;line-height:1.9}
  .mood b{color:var(--text);font-weight:600}
  .mol{display:flex;flex-direction:column;gap:7px}
  .mrow{display:grid;grid-template-columns:64px 1fr 40px;gap:8px;align-items:center;font-size:12px}
  .mname{color:var(--muted);text-align:right;white-space:nowrap}
  .mbar{height:7px;border-radius:6px;background:#2c241f;overflow:hidden}
  .mbar i{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,var(--warm),var(--warm2));transition:width .6s ease}
  .mval{color:var(--dim);text-align:right;font-variant-numeric:tabular-nums}
  .layers{display:flex;flex-direction:column;gap:10px}
  .layer{border:1px solid var(--line);border-radius:12px;padding:10px 12px;background:var(--card2)}
  .layer .lt{font-size:12px;color:var(--dim);letter-spacing:1px;margin-bottom:6px}
  .layer .vals{display:flex;flex-wrap:wrap;gap:6px}
  .chip{font-size:12px;padding:3px 9px;border-radius:99px;background:#2c241f;color:var(--muted)}
  .chip b{color:var(--warm2);font-weight:600;font-variant-numeric:tabular-nums}
  .mems{display:flex;flex-direction:column;gap:12px}
  .mem{background:var(--card2);border:1px solid var(--line);border-left:3px solid var(--dim);border-radius:12px;padding:13px 15px}
  .mem.active{border-left-color:var(--ok)}
  .mem.candidate{border-left-color:var(--wait)}
  .mem.invalidated,.mem.rejected,.mem.superseded,.mem.decayed{border-left-color:var(--dead);opacity:.55}
  .mtext{font-size:14.5px;color:var(--text);word-break:break-word}
  .mmeta{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:11.5px;color:var(--dim)}
  .badge{font-size:11px;padding:1px 8px;border-radius:99px;background:#2c241f}
  .badge.active{background:rgba(127,209,160,.14);color:var(--ok)}
  .badge.candidate{background:rgba(232,194,104,.14);color:var(--wait)}
  .empty{color:var(--dim);font-size:13px;padding:16px 0;text-align:center}
  footer{margin-top:28px;color:var(--dim);font-size:12px;text-align:center}
  .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-right:6px;vertical-align:middle}
  .dot.bad{background:#e06c6c}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:10px}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  .legend i{display:inline-block;width:9px;height:9px;border-radius:50%}
  .legend i.dash{border-radius:0;width:14px;height:0;border-top:1px dashed #4f6f8f;background:transparent}
  .graph{min-height:150px}
  .graph svg{width:100%;height:auto;display:block}
  .topics{display:flex;flex-direction:column;gap:12px}
  .topic{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
  .thead{display:flex;align-items:center;gap:8px;margin-bottom:8px}
  .tdot{width:9px;height:9px;border-radius:50%;display:inline-block}
  .tname{font-size:14px;color:var(--text);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .tcount{font-size:11.5px;color:var(--dim)}
  .tmem{font-size:12.5px;color:var(--muted);padding:4px 0;border-top:1px dashed var(--line);line-height:1.5}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>溯忆<small>记得，是为了重新认出你</small></h1>
    <span class="sub" id="health"><span class="dot"></span>连接中…</span>
  </header>

  <div class="stats" id="stats"></div>

  <div class="grid">
    <div>
      <div class="card">
        <h2>此刻的我</h2>
        <div class="mood" id="mood">—</div>
        <div class="mol" id="molecules"></div>
      </div>
      <div class="card">
        <h2>人格三层</h2>
        <div class="layers" id="layers"></div>
      </div>
    </div>

    <div class="card">
      <h2>记忆网络</h2>
      <div class="legend">
        <span><i style="background:#ff9a6c"></i>你说的</span>
        <span><i style="background:#7fb0d8"></i>我回的</span>
        <span><i style="background:#8d7f76"></i>未标记</span>
        <span><i style="background:#7fd1a0"></i>长期记忆</span>
        <span><i class="dash"></i>时间相邻</span>
        <span><i style="background:#7a5240"></i>内容相关</span>
      </div>
      <div class="graph" id="graph"><div class="empty">读取中…</div></div>
    </div>

    <div class="card">
      <h2>最近的话题</h2>
      <div class="topics" id="topics"><div class="empty">读取中…</div></div>
    </div>

    <div class="card">
      <h2>记忆</h2>
      <div class="mems" id="mems"><div class="empty">读取中…</div></div>
    </div>
  </div>

  <footer>每 5 秒自动刷新 · <span id="ts">—</span></footer>
</div>

<script>
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fix = (n, d = 3) => (typeof n === 'number' ? n.toFixed(d) : '—');

// 界面上的名词一律说中文；英文只留在 CSS class 和接口字段里。
const STATUS_ZH = {
  active: '长期', candidate: '候选', invalidated: '已失效', superseded: '已取代',
  decayed: '已衰减', rejected: '已拒绝', archived: '已归档', needs_review: '待复核',
  buffered: '缓冲中', episodized: '已成情节', observed: '已观察',
};
const TYPE_ZH = {
  // working 是心理学里的 working memory（短时记忆），不是「工作/办公」。
  // 以前译成「工作」，看到「摸摸头」标着「工作」会以为分类错了——其实是译名错了。
  episodic: '情景', semantic: '语义', working: '短时', procedural: '技能', interest: '兴趣',
};
const EMOTION_ZH = {
  valence: '效价', arousal: '唤醒', joy: '喜悦', sadness: '悲伤', anger: '愤怒',
  fear: '恐惧', disgust: '厌恶', surprise: '惊讶', calm: '平静', anxiety: '焦虑',
  trust: '信任', love: '爱',
};
const TRAIT_ZH = {
  warmth: '温柔', directness: '直率', curiosity: '好奇', playfulness: '玩心', boundary: '边界',
};

async function load() {
  try {
    const [h, p, m, g, t] = await Promise.all([
      fetch('/health').then(r => r.json()),
      fetch('/persona').then(r => r.json()),
      fetch('/memories?limit=300').then(r => r.json()),
      fetch('/graph').then(r => r.json()).catch(() => ({ nodes: [], edges: [] })),
      fetch('/topics').then(r => r.json()).catch(() => ({ topics: [] })),
    ]);
    renderHealth(h);
    renderPersona(p);
    renderMemories(m);
    renderGraph(g);
    renderTopics(t);
    $('#ts').textContent = new Date().toLocaleTimeString('zh-CN');
  } catch (e) {
    $('#health').innerHTML = '<span class="dot bad"></span>连不上溯忆服务';
  }
}

function renderHealth(h) {
  const ok = h && h.ok;
  $('#health').innerHTML = `<span class="dot${ok ? '' : ' bad'}"></span>${ok ? '在线' : '离线'}` +
    (h.vector_up ? ' · 向量层正常' : ' · 向量层降级');
  const mn = h.mnemosyne || {};
  const items = [
    ['记忆', mn.memories ?? '—'],
    ['观察', mn.observations ?? '—'],
    ['向量', (h.vector && h.vector.rows) ?? '—'],
  ];
  $('#stats').innerHTML = items.map(([k, v]) =>
    `<div class="stat"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join('');
}

function renderPersona(p) {
  const d = (p && p.dynamic) || {};
  const eff = d.effects || {};
  const em = d.emotion || {};
  const topEmo = Object.entries(em)
    .filter(([k]) => k !== 'valence' && k !== 'arousal')
    .sort((a, b) => Math.abs(b[1] - 0.5) - Math.abs(a[1] - 0.5))
    .slice(0, 3);
  // 情绪值已经以「0.5 = 平时的自己」为中性。服务端 2026-11-11 把放大倍数
  // 从 2.0 调回 1.0（1:1 映射），所以下面乘 100 就真的是「原始偏离百分点」。
  // 旧版这里乘 200 且服务端还放大 2 倍，等于把原始偏离放大 400 倍：
  // 原始偏离 0.25 就顶到 +100「满格恐惧」。现在只映射到 ±50，
  // 并把刻度写在标题里——温和的波动不该被读成极端。
  const dev = (v) => {
    const d = Math.round((v - 0.5) * 100);
    return d > 0 ? `+${d}` : String(d);
  };
  $('#mood').innerHTML = `
    唤醒 <b>${fix(eff.arousal, 2)}</b> · 效价 <b>${fix(eff.valence, 2)}</b> · 疲劳 <b>${fix(d.fatigue, 2)}</b><br>
    情绪偏离（0 = 平时的你 · ±50 = 顶格 · 只列前 3）：${topEmo.map(([k, v]) => `<b>${esc(EMOTION_ZH[k] || k)}</b> ${dev(v)}`).join(' · ') || '—'}`;

  const mols = d.molecules || {};
  const sorted = Object.entries(mols).sort((a, b) => b[1].level - a[1].level).slice(0, 8);
  $('#molecules').innerHTML = sorted.map(([, v]) => `
    <div class="mrow">
      <div class="mname">${esc(v.name)}</div>
      <div class="mbar"><i style="width:${Math.round((v.level || 0) * 100)}%"></i></div>
      <div class="mval">${fix(v.level, 2)}</div>
    </div>`).join('');

  const base = (p && p.base) || {};
  const shaped = (p && p.shaped) || {};
  const chips = (obj, readOnly) => Object.entries(obj).map(([k, v]) => {
    const val = readOnly ? (v && v.value) : (v && v.value);
    return `<span class="chip">${esc(TRAIT_ZH[k] || k)} <b>${fix(val, 2)}</b></span>`;
  }).join('');
  $('#layers').innerHTML = `
    <div class="layer"><div class="lt">基础人格 · 只读锚</div><div class="vals">${chips(base, true) || '<span class="chip">—</span>'}</div></div>
    <div class="layer"><div class="lt">经验塑造 · 证据驱动，慢变</div><div class="vals">${chips(shaped) || '<span class="chip">—</span>'}</div></div>
    <div class="layer"><div class="lt">动态状态 · 分钟级衰减</div><div class="vals">
      <span class="chip">唤醒 <b>${fix(eff.arousal, 2)}</b></span>
      <span class="chip">疲劳 <b>${fix(d.fatigue, 2)}</b></span>
      <span class="chip">记忆显著性 <b>${fix(eff.memory_salience, 2)}</b></span>
    </div></div>`;
}

// 记忆网络：按时间从左到右铺开，边是引用关系。
// 不用力导向库——时间轴本身就能看出「越往右越新」，
// 曲线边能看出「这些记忆是连着的」，点的大小是邻居数。
function renderGraph(g) {
  const box = $('#graph');
  const nodes = (g && g.nodes) || [];
  if (!nodes.length) { box.innerHTML = '<div class="empty">还没有可画的记忆</div>'; return; }
  const W = 1000, H = 320, PAD = 44;
  const sorted = [...nodes].sort((a, b) => String(a.created_at || '').localeCompare(String(b.created_at || '')));
  const pos = {};
  sorted.forEach((n, i) => {
    const t = sorted.length > 1 ? i / (sorted.length - 1) : 0.5;
    pos[n.id] = {
      x: PAD + t * (W - PAD * 2),
      y: H / 2 + Math.sin(t * Math.PI * 3.2) * (H / 2 - 66),
    };
  });
  const edges = ((g && g.edges) || []).map((e) => {
    const a = pos[e.from], b = pos[e.to];
    if (!a || !b) return '';
    const dash = e.rel === 'temporal_neighbor';
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 - 24;
    return `<path d="M${a.x.toFixed(1)} ${a.y.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${b.x.toFixed(1)} ${b.y.toFixed(1)}"` +
      ` stroke="${dash ? '#4f6f8f' : '#7a5240'}" stroke-width="1.1" fill="none"` +
      `${dash ? ' stroke-dasharray="3 4"' : ''} opacity=".6"><title>${esc(dash ? '时间相邻' : '内容相关')}</title></path>`;
  }).join('');
  const dots = sorted.map((n) => {
    const p = pos[n.id];
    const r = Math.min(8, 3.5 + (n.links_count || 0) * 1.1);
    const fill = n.role === 'user' ? '#ff9a6c' : (n.role === 'assistant' ? '#7fb0d8' : '#8d7f76');
    const stroke = n.status === 'active' ? '#7fd1a0' : 'rgba(0,0,0,0)';
    return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r.toFixed(1)}"` +
      ` fill="${fill}" stroke="${stroke}" stroke-width="1.6" opacity=".92"><title>${esc(n.summary)}</title></circle>`;
  }).join('');
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${edges}${dots}</svg>`;
}

// 最近的话题：语义聚类的结果。这事在换真句向量之前做不出来（hashing 聚出来是噪声）。
function renderTopics(t) {
  const box = $('#topics');
  const list = (t && t.topics) || [];
  if (!list.length) {
    box.innerHTML = '<div class="empty">' + esc((t && (t.note || t.error)) || '还没有足够的话题') + '</div>';
    return;
  }
  const colors = ['#ff9a6c', '#7fb0d8', '#7fd1a0', '#c79cff', '#e8c268', '#ff8fa3'];
  box.innerHTML = list.map((tp, i) => {
    const c = colors[i % colors.length];
    const mems = (tp.members || []).slice(0, 3)
      .map((m) => `<div class="tmem">· ${esc(m.summary)}</div>`).join('');
    return `<div class="topic">
      <div class="thead"><span class="tdot" style="background:${c}"></span>
        <span class="tname">${esc(tp.name)}</span><span class="tcount">${tp.count} 条</span></div>
      ${mems}
    </div>`;
  }).join('');
}

function renderMemories(payload) {
  const items = (payload && payload.items) || [];
  if (!items.length) {
    $('#mems').innerHTML = '<div class="empty">还没有记忆。说句话给它听。</div>';
    return;
  }
  const order = { active: 0, candidate: 1, superseded: 2, invalidated: 3 };
  items.sort((a, b) => {
    const oa = order[a.status] ?? 9, ob = order[b.status] ?? 9;
    if (oa !== ob) return oa - ob;
    return String(b.created_at || '').localeCompare(String(a.created_at || ''));
  });
  $('#mems').innerHTML = items.map((m) => `
    <div class="mem ${esc(m.status)}">
      <div class="mtext">${esc(m.content)}</div>
      <div class="mmeta">
        <span class="badge ${esc(m.status)}">${esc(STATUS_ZH[m.status] || m.status)}</span>
        <span>${esc(TYPE_ZH[m.type] || m.type || '')}</span>
        <span>${esc((m.created_at || '').replace('T', ' ').slice(0, 16))}</span>
        <span>${esc(String(m.id || '').slice(-6))}</span>
      </div>
    </div>`).join('');
}

load();
setInterval(load, 5000);
</script>
</body>
</html>
"""
