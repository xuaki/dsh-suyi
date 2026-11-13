/**
 * 溯忆 · Fast Path：把「确定不值得记」的判断放在本地。
 *
 * 借鉴 Super-agent 的 Fast / Slow Path 设计：走服务端那条路灵活但贵，
 * 而已经明确且稳定的高频判断，应该固化成确定性代码。
 *
 * 这里做的判断全部同步、零网络，目的是省掉一次 HTTP 往返 + 一次准入判定：
 *   - 纯语气词（「嗯」「好的」「收到」）不记
 *   - 旧客户端留下的「对话片段：」回显不记
 *   - 同一句话 5 分钟内不重复记（一轮里 user / assistant 常各说一遍）
 *
 * 注意：这里只负责「明显不值得记的别浪费一次往返」，
 * 判断「值不值得留下来」仍然是 persona-memory 的慢路径（五维准入 + 证据链）。
 */

const ACK_RE = /^(嗯+|好+|行|可以|ok|okay|收到|知道了|明白了|了解|谢谢|谢啦|哈哈+|嘿嘿+|👌|👍)[嗯好的行了收到！!。，,、？?\s]*$/i;

/** 纯语气词 / 纯肯定：没有任何可回忆的内容。 */
export function isAckLike(text) {
  if (!text) return true;
  return ACK_RE.test(String(text).trim());
}

/** 旧客户端回显格式（形如「对话片段：…」）：不是用户说的话，也不是我的回答。 */
export function isEchoFragment(text) {
  return String(text || '').trimStart().startsWith('对话片段：');
}

/**
 * 短时去重：同一段文本在 windowMs 内只记一次。
 * 用归一化（去空白 + 截断）做键，避免「格式差异」骗过去。
 */
export function createDeduper({ windowMs = 5 * 60 * 1000, maxEntries = 200 } = {}) {
  const seen = new Map();
  return function isDuplicate(text) {
    const now = Date.now();
    if (seen.size > maxEntries) {
      for (const [k, ts] of seen) {
        if (now - ts > windowMs) seen.delete(k);
      }
    }
    const norm = String(text || '').replace(/\s+/g, '').slice(0, 160);
    if (!norm) return true;
    const last = seen.get(norm);
    seen.set(norm, now);
    return last !== undefined && now - last < windowMs;
  };
}

/**
 * 组合判断：返回跳过原因（空字符串表示「值得进慢路径」）。
 * 调用方拿到原因后可以打日志，方便以后收窄/放宽规则。
 */
export function shouldSkipCapture(text, isDuplicate) {
  if (!text) return 'empty';
  if (isEchoFragment(text)) return 'echo_fragment';
  if (isAckLike(text)) return 'ack';
  if (typeof isDuplicate === 'function' && isDuplicate(text)) return 'duplicate';
  return '';
}
