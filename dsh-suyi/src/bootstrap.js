/**
 * 溯忆 · 服务守护：让 DSH 一启动，记忆服务自己醒过来。
 *
 * 之前的问题：服务是我用 DSH 的后台任务拉起来的，DSH 一重启，
 * 它们就跟着一起被带走——每次都要手动重新启动，这不该是用户的活。
 *
 * 这里的做法：
 *   1. 先探健康，已经在跑就不重复启动；
 *   2. 需要启动时用 detached + unref 拉起，**服务脱离 DSH 进程树**，
 *      之后 DSH 关闭/重开都不影响它；
 *   3. 把服务日志写到 data/<name>.log，出问题能查。
 *
 * 路径优先级：配置 > 默认推断（插件目录的上级就是工作区）。
 */

import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, openSync } from 'node:fs';
import { join, resolve } from 'node:path';

const PLUGIN_DIR = resolve(import.meta.dirname, '..');
const WORKSPACE = resolve(PLUGIN_DIR, '..');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * 探测 Milvus 的地​址。
 *
 * 2026-11-12 起 Milvus 搬到了 Docker Desktop（见 `_milvus/docker-compose.yml`），
 * 它的端口是**直通** 127.0.0.1 的，不用再探测 WSL 的 IP。
 *
 * 原来为什么探 WSL：Milvus 之前跑在 WSL 的 docker 里，WSL2 的 localhost 端口转发
 * 会断（重启后尤其常见），一断 persona-memory 就连不上 19530。它三次掉链子、
 * 还丢过索引（重建时报 mixcoord 节点丢失），所以搬走了。
 *
 * 连不上也没关系：`retrieve()` 会自动降级到内存向量索引（vector_memory.py）。
 */
function resolveMilvusUri() {
  if (process.env.MILVUS_URI) return process.env.MILVUS_URI;
  return 'http://127.0.0.1:19530';
}

/**
 * 构建戳：必须和服务端 persona_memory/api.py 的 SERVICE_BUILD 一致。
 * 改过 Python 代码就同步改这里——下次 DSH 启动时 bootstrap 会发现版本不对，
 * 自动杀掉旧服务、拉起新的。
 *
 * 为什么需要这个：服务是 detached 拉起的，常常是管理员权限，
 * 普通终端杀不动（2026-11-07 真实撞过这堵墙）。让 DSH 自己动手是最稳的一条路。
 */
export const EXPECTED_BUILD = '2026-11-14-panel-freshness';

/**
 * 默认路径：优先读环境变量，否则按「插件在 default-workspace/dsh-suyi 下」推断。
 *
 * 所有路径都能用环境变量覆盖（见 README「安装」一节）：
 *   SUYI_MNEMOSYNE_ROOT   mnemosyne 服务目录（Python 包所在）
 *   SUYI_MNEMOSYNE_DATA   mnemosyne 数据目录（SQLite 权威源）
 *   SUYI_PERSONA_ROOT     persona-memory 服务目录
 *   SUYI_PERSONA_DATA     persona-memory 数据目录
 *   SUYI_PYTHON           跑 Python 服务的解释器
 */
export function defaultPaths() {
  const env = process.env;
  const mnemosyneRoot =
    env.SUYI_MNEMOSYNE_ROOT || join(WORKSPACE, '_inspect', 'mnemosyne', 'service');
  const personaRoot = env.SUYI_PERSONA_ROOT || join(WORKSPACE, 'persona-memory');
  const isWin = process.platform === 'win32';
  return {
    workspace: WORKSPACE,
    python:
      env.SUYI_PYTHON ||
      join(mnemosyneRoot, '.venv', isWin ? 'Scripts' : 'bin', isWin ? 'python.exe' : 'python'),
    mnemosyneRoot,
    mnemosyneData: env.SUYI_MNEMOSYNE_DATA || join(WORKSPACE, '_inspect', 'mnemosyne-data'),
    personaRoot,
    personaData: env.SUYI_PERSONA_DATA || join(personaRoot, 'data'),
    logDir: join(WORKSPACE, 'dsh-suyi', 'data'),
  };
}

/**
 * 构造服务定义。cfg 里可以覆盖任何路径：
 *   python / mnemosyne_root / mnemosyne_data / persona_root / persona_data
 */
export function buildServices(cfg = {}) {
  const d = { ...defaultPaths(), ...pickDefined(cfg) };
  return [
    {
      name: 'mnemosyne',
      health: cfg.mnemosyne_url
        ? `${String(cfg.mnemosyne_url).replace(/\/+$/, '')}/healthz`
        : 'http://127.0.0.1:8765/healthz',
      command: d.python,
      cwd: d.mnemosyneRoot,
      args: [
        '-m',
        'mnemosyne.main',
        '--host',
        '127.0.0.1',
        '--port',
        '8765',
        '--token',
        String(cfg.mnemosyne_token || 'dev-token'),
        '--data-dir',
        d.mnemosyneData,
      ],
      env: { PYTHONPATH: join(d.mnemosyneRoot, 'src') },
      log: join(d.logDir, 'mnemosyne.log'),
    },
    {
      name: 'persona-memory',
      health: cfg.base_url ? `${String(cfg.base_url).replace(/\/+$/, '')}/health` : 'http://127.0.0.1:8772/health',
      command: d.python,
      cwd: d.personaRoot,
      args: ['-m', 'persona_memory.main', '--port', '8772'],
      env: {
        PYTHONPATH: d.personaRoot,
        // WSL 的 localhost 转发会断，所以每次启动探一次真实地址
        // 2026-11-12：Milvus 改用 **milvus-lite**（pymilvus 的嵌入式后端，本地文件）。
        // 为什么不再用服务端：WSL 里那个三次掉链子还丢过索引；Docker Desktop
        // 又拉不到镜像（registry 全超时）。我们的量级（几百到几万条）用本地文件
        // 完全够，而且不要 docker、不要 WSL、不要端口转发——一把更小的刀。
        // 想换回独立服务：设 MILVUS_URI=http://127.0.0.1:19530 即可。
        MILVUS_URI: process.env.MILVUS_URI || join(d.personaData, 'milvus_lite.db'),
        // 语义向量模型已在本地缓存：走镜像 + 离线模式。
        // 不设这两条，fastembed 会去 huggingface.co 校验模型，网络一慢就把
        // 服务启动卡在重试里——2026-11-03 凌晨真出过一次，溯忆直接连不上。
        HF_ENDPOINT: 'https://hf-mirror.com',
        HF_HUB_OFFLINE: '1',
        // 第三级降级：Milvus + mnemosyne 都挂时，直接读权威源的 SQLite 做字面扫描。
        MNEMOSYNE_DB: join(d.mnemosyneData, 'data', 'memory.db'),
        // 内存向量索引（Milvus 的替补）：几百条记忆在 numpy 里算余弦是毫秒级，
        // 不需要一个会掉链子的独立向量数据库。懒加载 + npz 持久化，重启直接读。
        // 设成 off 可以关掉它（只剩 Milvus + 字面兜底）。
        PERSONA_VECTOR_INDEX: join(d.personaData, 'vector_index.npz'),
      },
      build: EXPECTED_BUILD,
      log: join(d.logDir, 'persona-memory.log'),
    },
  ];
}

function pickDefined(obj) {
  const out = {};
  const map = {
    python: 'python',
    workspace: 'workspace',
    mnemosyne_root: 'mnemosyneRoot',
    mnemosyne_data: 'mnemosyneData',
    persona_root: 'personaRoot',
    persona_data: 'personaData',
    log_dir: 'logDir',
  };
  for (const [k, target] of Object.entries(map)) {
    if (obj[k] !== undefined && obj[k] !== null) out[target] = String(obj[k]);
  }
  return out;
}

export async function isHealthy(url, timeoutMs = 1500) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    return res.ok;
  } catch {
    return false;
  }
}

/** 读健康响应里的内容（要 build 戳，不能只看 ok）。 */
async function healthInfo(url, timeoutMs = 1500) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** 从 health URL 里取端口。 */
function portOf(url) {
  try {
    return new URL(url).port;
  } catch {
    return '';
  }
}

/**
 * 按端口杀掉监听进程。
 * 不用 pid 文件：服务是 detached 拉起的、自己不写 pid；
 * 而「谁占着这个端口」用 netstat 一看就知道。
 */
async function killByPort(port) {
  if (!port) return 0;
  const { execFileSync } = await import('node:child_process');
  const pids = new Set();
  try {
    if (process.platform === 'win32') {
      const out = execFileSync('netstat', ['-ano'], { encoding: 'utf8' });
      for (const line of out.split(/\r?\n/)) {
        if (!line.includes(`:${port}`) || !line.includes('LISTENING')) continue;
        const parts = line.trim().split(/\s+/);
        const pid = parts[parts.length - 1];
        if (pid && pid !== '0') pids.add(pid);
      }
    } else {
      const out = execFileSync('lsof', ['-ti', `tcp:${port}`, '-sTCP:LISTEN'], { encoding: 'utf8' });
      for (const pid of out.split(/\s+/)) if (pid) pids.add(pid);
    }
  } catch {
    return 0;
  }
  let killed = 0;
  for (const pid of pids) {
    try {
      execFileSync(
        process.platform === 'win32' ? 'taskkill' : 'kill',
        process.platform === 'win32' ? ['/F', '/PID', pid] : ['-9', pid],
        { stdio: 'ignore' },
      );
      killed += 1;
    } catch {
      /* 权限不够就放弃——不要把启动流程弄崩 */
    }
  }
  return killed;
}

/** 确保一个服务在线。返回 'already-running' | 'started'。 */
export async function ensureService(
  def,
  { waitRounds = 40, intervalMs = 500, log = null, expectedBuild = null } = {},
) {
  const info = await healthInfo(def.health);
  if (info) {
    if (!expectedBuild || info.build === expectedBuild) return 'already-running';
    // 旧版本还活着：代码是老的、功能对不上。杀掉，让新代码上来。
    log?.(`[suyi] ${def.name} build 不匹配（${info.build ?? '(无)'} ≠ ${expectedBuild}），准备重启`);
    const killed = await killByPort(portOf(def.health));
    log?.(`[suyi] ${def.name} 停掉旧进程 ${killed} 个`);
    await sleep(1200);
  }

  if (!existsSync(def.command)) {
    throw new Error(`${def.name}: python 不存在 -> ${def.command}`);
  }
  if (!existsSync(def.cwd)) {
    throw new Error(`${def.name}: 工作目录不存在 -> ${def.cwd}`);
  }

  try {
    mkdirSync(resolve(def.log, '..'), { recursive: true });
  } catch {
    /* ignore */
  }
  let out = null;
  try {
    out = openSync(def.log, 'a');
  } catch {
    out = null;
  }

  const child = spawn(def.command, def.args, {
    cwd: def.cwd,
    env: { ...process.env, ...(def.env || {}) },
    // detached + unref：服务脱离 DSH 的进程树，DSH 关了它还活着
    detached: true,
    stdio: out === null ? 'ignore' : ['ignore', out, out],
    windowsHide: true,
  });
  child.unref();
  log?.(`[suyi] ${def.name} 启动中 (pid ${child.pid})`);

  for (let i = 0; i < waitRounds; i += 1) {
    await sleep(intervalMs);
    if (await isHealthy(def.health)) {
      log?.(`[suyi] ${def.name} 就绪`);
      return 'started';
    }
  }
  throw new Error(`${def.name}: 启动后 ${(waitRounds * intervalMs) / 1000}s 内未就绪，看 ${def.log}`);
}

/** 确保全部服务在线，返回每个服务的结果。 */
export async function ensureAll(cfg = {}, { log = null } = {}) {
  const results = [];
  for (const def of buildServices(cfg)) {
    try {
      const state = await ensureService(def, { log, expectedBuild: def.build || null });
      results.push({ name: def.name, state });
    } catch (e) {
      results.push({ name: def.name, state: 'failed', error: e.message });
    }
  }
  return results;
}

// 允许手动跑：node src/bootstrap.js  → 只确保服务，不启动 DSH
if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
  const results = await ensureAll({}, { log: (m) => console.log(m) });
  console.log(JSON.stringify(results, null, 2));
  process.exit(results.some((r) => r.state === 'failed') ? 1 : 0);
}
