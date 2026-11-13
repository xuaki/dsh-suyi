// 预览：新会话开场时，会自动带上「我们之前聊过这些」。
const b = 'http://127.0.0.1:8770';

const EM = new Set(['对话片段：']);

function renderIntro(items) {
  const rows = (items || [])
    .filter(
      (x) =>
        String(x.status || '').toLowerCase() === 'active' &&
        !String(x.content || '').startsWith('对话片段：'),
    )
    .slice(0, 5)
    .map((x) => `- ${String(x.content || '').replace(/\s+/g, ' ').slice(0, 36)}`);
  if (!rows.length) return '';
  return '[我们之前聊过这些]（里面可能混着当前这几句，挑没见过的看）\n' + rows.join('\n') + '\n\n';
}

const r = await (await fetch(`${b}/memories?limit=40`)).json();
const out = renderIntro(r?.items);
console.log(out ? '--- 新会话开场会带上的内容 ---' : '--- 没有可用前情 ---');
if (out) console.log(out.trim());
console.log('\n（候选记忆不会进来，只有已经晋升的「长期」才会。）');
