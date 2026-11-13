// 预览：当前状态下，注入给模型的那段话长什么样。
// 逻辑与 src/index.js 的 renderMood 保持一致（这里独立实现，避免 import dsh-tools）。

const EMOTION_ZH = {
  joy: '喜悦', sadness: '悲伤', anger: '愤怒', fear: '恐惧', disgust: '厌恶',
  surprise: '惊讶', calm: '平静', anxiety: '焦虑', trust: '信任', love: '爱',
};

function renderMood(p) {
  const d = p && p.dynamic;
  if (!d) return '';
  const eff = d.effects || {};
  const fatigue = Number(d.fatigue || 0);
  const arousal = Number(eff.arousal || 0);
  const valence = Number(eff.valence || 0);
  const parts = [];

  if (fatigue > 0.55) parts.push('很疲倦：句子短、慢，允许自己没精神');
  else if (fatigue > 0.3) parts.push('有点乏：节奏放慢，别太跳');

  if (arousal > 0.6) parts.push('心跳偏快、有点急：句子更短，更容易追问和确认');

  if (valence < -0.25) parts.push('心情有点往下沉：别用轻快的语气，也别假装没事');
  else if (valence > 0.35) parts.push('心情是亮的：笑意多一点');

  const em = d.emotion || {};
  const top = Object.entries(em)
    .filter(([k]) => k !== 'valence' && k !== 'arousal')
    .sort((a, b) => Math.abs(b[1] - 0.5) - Math.abs(a[1] - 0.5))[0];
  if (top && Math.abs(top[1] - 0.5) > 0.15) {
    const dev = Math.round((top[1] - 0.5) * 200);
    const name = EMOTION_ZH[top[0]] || top[0];
    parts.push(dev > 0 ? `${name}比平时明显` : `${name}比平时淡`);
  }

  if (!parts.length) return '';
  return (
    `[此刻的你] ${parts.join('；')}。` +
    '这是你真实的状态，不是表演——让它自然渗进措辞里。不要报数值，不要解释它，也不要主动提起。\n\n'
  );
}

const p = await (await fetch('http://127.0.0.1:8770/persona')).json();
const out = renderMood(p);
console.log('疲劳', p.dynamic.fatigue, '| 唤醒', p.dynamic.effects.arousal, '| 效价', p.dynamic.effects.valence);
console.log('');
console.log(out ? '--- 注入内容 ---' : '--- 当前状态平静，不注入任何内容 ---');
if (out) console.log(out.trim());
