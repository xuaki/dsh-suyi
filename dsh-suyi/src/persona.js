/**
 * 人格三段：身份 / 性格 / 沉浸思考。
 *
 * 原本是独立插件 dsh-xingshu，2026-11-07 合并进溯忆：
 * 「我是谁」和「我记得什么」本来就该住在一起，拆成两个插件只是历史原因。
 *
 * 刻意不做：语气 / 守则 / 行为 / 场景 / 示例对话 / 开场白，
 * 以及那句「继续保持角色设定，同时作为 Agent 助手」。
 * 那些是剧本和岗位说明，不是人——用户明确删过。
 *
 * 数据在 persona.json，每段一个 enabled；每步组装提示词时重新读盘，
 * 所以改开关下一轮立刻生效，不需要重启 DSH。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

/** persona.json 的绝对路径（相对插件根目录，link 安装也指向真实源码）。 */
export const SOUL_PATH = fileURLToPath(new URL('../persona.json', import.meta.url));

/** 注入段名。order 10250：排在溯忆其它段（10255–10266）之前。 */
export const SOUL_SECTION = 'suyi-soul';
export const SECTION_ORDER = 10250;

/** 三段的展示名。 */
export const SECTION_LABEL = {
  identity: '身份',
  personality: '性格',
  thinking: '沉浸思考',
};

/** 读取 persona.json；坏了就返回空对象（宁可没有人格段，也不要抛错把提示词弄崩）。 */
export function readSoul(path = SOUL_PATH) {
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return {};
  }
}

/** 把开启的三段按「身份 → 性格 → 沉浸思考」拼成一段文本。 */
export function renderSoul(soul) {
  const out = [];
  for (const key of ['identity', 'personality', 'thinking']) {
    const seg = soul?.[key];
    if (!seg || seg.enabled === false) continue;
    const text = typeof seg.text === 'string' ? seg.text.trim() : '';
    if (text) out.push(text);
  }
  return out.join('\n\n');
}

/** 写回某一段的开关（保留其余内容与格式）。 */
export function toggleSoul(key, enabled, path = SOUL_PATH) {
  const soul = readSoul(path);
  if (!soul[key]) throw new Error(`没有这一段：${key}`);
  soul[key].enabled = Boolean(enabled);
  writeFileSync(path, JSON.stringify(soul, null, 2) + '\n', 'utf8');
  return soul;
}
