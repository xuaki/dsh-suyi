/**
 * 溯忆 · Anamnesis —— 工具定义。
 *
 * 用完整的 JSON Schema 手写，不依赖 @deepseek-ai/dsh-tools：
 * 该包在 DSH 里是打进 asar 的，插件从 link 目录解析不到它。
 * dsh-plastic-memory 也用同一套写法（ctx.get('tools').register(完整 schema)）。
 */

export const RECALL_TOOL = 'su_recall';
export const EXPAND_TOOL = 'su_expand';
export const REMEMBER_TOOL = 'su_remember';
export const PERSONA_TOOL = 'su_persona';
export const FORGET_TOOL = 'su_forget';
export const HEALTH_TOOL = 'su_health';
export const EVIDENCE_TOOL = 'su_evidence';
export const GAP_TOOL = 'su_gap';
export const SOUL_STATUS_TOOL = 'su_soul_status';
export const SOUL_TOGGLE_TOOL = 'su_soul_toggle';
export const INTEREST_TOOL = 'su_interest';
export const INTEREST_NOTE_TOOL = 'su_interest_note';

export function interestToolSpec() {
  return {
    name: INTEREST_TOOL,
    description:
      '关于「他喜欢什么」。action=list 看兴趣画像；action=add 手动补一条（你判断他刚才是在说喜欢某个东西时）。' +
      '自动学习已经在后台跑（只从「我喜欢X」「我最近在玩X」这类明确句式里抓，不猜），' +
      '手动补是为了补上机器抓不到的。别为了显得上心而补——记错比不记更糟。',
    parameters: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['list', 'add'], description: '要做什么' },
        topic: { type: 'string', description: 'add 时必填：他喜欢的东西/事' },
        example: { type: 'string', description: 'add 时可选：他当时怎么说的' },
      },
      required: ['action'],
    },
    output: jsonOutput('兴趣画像'),
  };
}

export function interestNoteToolSpec() {
  return {
    name: INTEREST_NOTE_TOOL,
    description:
      '这不是确认，只是留个线索——他顺口提到某样东西时用。' +
      '注意：他说「我朋友超喜欢滑雪」「如果我要学钢琴」这种不算他的偏好，别记；' +
      '只有真的在说他自己、但他又没明说「我喜欢」的时候，才留一条弱线索（tier 用 passive 或 attention）。' +
      '单条永远不会变成「他喜欢X」——跨天反复出现，服务端自己会把它攒成结论。' +
      '别为了显得上心而记：留错线索比不记更糟。',
    parameters: {
      type: 'object',
      properties: {
        topic: {
          type: 'string',
          description: '他顺口提到的东西/事，尽量短（「滑雪」「手冲咖啡」），别写整句',
        },
        polarity: {
          type: 'integer',
          enum: [1, -1],
          description: '1 = 像是喜欢 / -1 = 像是反感。默认 1',
        },
        tier: {
          type: 'string',
          enum: ['passive', 'attention', 'engage', 'explicit'],
          description: '证据层级：passive(随口一提) | attention(留意/追问) | engage(主动聊) | explicit(明确表态)。默认 passive',
        },
        quote: { type: 'string', description: '他当时怎么说的（原话，越短越好）——以后能点回原句' },
        confidence: {
          type: 'number',
          description: '这条线索有多像偏好，0~1，默认 0.5。只有确实很确定才给 0.7 以上',
        },
      },
      required: ['topic'],
    },
    output: jsonOutput('这条线索有没有留下、以及该话题当前算不算已确认'),
  };
}


export function jsonOutput(description = 'JSON 结果') {
  return {
    schema: { type: 'object', description },
    render(args, value) {
      return [
        {
          type: 'text',
          text: typeof value === 'string' ? value : JSON.stringify(value, null, 2),
        },
      ];
    },
  };
}

export function recallToolSpec() {
  return {
    name: RECALL_TOOL,
    description:
      '从溯忆（长期记忆）里回忆相关的事。回答用户问题前、或需要接上之前聊过的内容时用它。' +
      '默认返回「摘要 + 引用」：每条是一句话摘要、来源、以及它指向的邻居 id（链式召回）。' +
      '需要某条记忆的原文、或它连着的细节时，用 su_expand(memory_id) 展开。' +
      '每次召回都带 evidence.level：attested=有据 / weak=只靠字面沾边（沾边不等于有据）/ none=没找到。' +
      'level 是 none 或全是 weak 时，就直说「我不记得了」或「这条不太确定」，别硬编。' +
      '每条带来源（mnemosyne=证据链 / milvus=语义索引 / memory_index=内存向量索引 / both=两者命中 / literal=字面兜底）、' +
      '与 query 的字面命中率（literal_relevance，0 表示一个字都不沾）与当前情绪状态——同一个问题，在你低落和开心时该有不同分寸。',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '要回忆什么。用自然语言描述即可。' },
        limit: { type: 'integer', description: '最多返回几条，默认 8' },
        detail: {
          type: 'boolean',
          description: '是否直接返回记忆原文（默认 false 只给摘要；true 会占用大量上下文，慎用）',
        },
        cluster: {
          type: 'string',
          enum: ['tech', 'emotion', 'preference', 'project'],
          description:
            '记忆路由：这次问的是哪类问题。不传就自动判断；判断不出就不加成。' +
            'tech=技术决策 / emotion=情感事件 / preference=偏好 / project=项目进展',
        },
        recent_days: {
          type: 'number',
          description:
            '只看最近 N 天内的记忆（默认 0 = 不限）。问「我们昨天/前两天聊了什么」时用它，比纯语义猜更准。',
        },
      },
      required: ['query'],
    },
    output: jsonOutput('摘要式记忆片段、引用 id 与当前情绪'),
  };
}

export function expandToolSpec() {
  return {
    name: EXPAND_TOOL,
    description:
      '展开一条记忆：拿回它的原文，以及顺着引用图走一跳能到达的邻居摘要（链式召回）。' +
      '当你想知道「这件事的前因后果」「我为什么记得它」时，先用 su_recall 拿 memory_id，再用它展开。' +
      'depth=2 会再往外走一跳，但别滥用——回忆不该变成漫游。',
    parameters: {
      type: 'object',
      properties: {
        memory_id: {
          type: 'string',
          description: '要展开的记忆 id（从 su_recall 返回的 items[].memory_id 拿）',
        },
        depth: { type: 'integer', description: '展开几跳，默认 1，最大 2' },
      },
      required: ['memory_id'],
    },
    output: jsonOutput('记忆原文与邻居摘要'),
  };
}

export function rememberToolSpec() {
  return {
    name: REMEMBER_TOOL,
    description:
      '主动把一件事写进溯忆。用户说「记住…」「别忘了…」时一定要调；' +
      '你判断某件事值得长期记住（承诺、偏好、重要日期、正在做的项目）时也调。' +
      '注意：这不是直接写长期记忆，而是提交一条候选——准入与晋升由证据链决定。',
    parameters: {
      type: 'object',
      properties: {
        content: { type: 'string', description: '要记住的完整一句话，不要缩写' },
        tier: {
          type: 'string',
          enum: ['passive', 'attention', 'engage', 'explicit'],
          description: '证据层级：passive(被动) | attention(留意) | engage(主动) | explicit(明确要求)。默认 engage',
        },
        source_kind: {
          type: 'string',
          enum: ['user_statement', 'dialog', 'agent_inference', 'feedback'],
          description: '来源类型，默认 user_statement',
        },
      },
      required: ['content'],
    },
    output: jsonOutput('准入结果与 memory_id'),
  };
}

export function personaToolSpec() {
  return {
    name: PERSONA_TOOL,
    description:
      '查看「此刻的我」：人格三层（基础人格 / 经验塑造 / 动态状态）、15 种分子浓度、情绪 12 维。' +
      '当你想知道自己现在是什么状态、或者需要解释「我为什么这样回应」时调用。',
    parameters: { type: 'object', properties: {}, required: [] },
    output: jsonOutput('人格三层与当前体温'),
  };
}

export function evidenceToolSpec() {
  return {
    name: EVIDENCE_TOOL,
    description:
      '给「经验塑造」层投一条证据——这是人格里唯一会真的改变的地方。' +
      '它不会因为一次聊天就变：≥3 条独立证据才开始动，一次最多挪 0.05，且每次都被基础人格的锚往回拉。' +
      '只在「真实发生过、且你真心觉得它体现了某个特质」时投；为让它动而投 = 污染人格。' +
      '一条对话通常最多一条；不确定就不投。',
    parameters: {
      type: 'object',
      properties: {
        trait: {
          type: 'string',
          enum: ['warmth', 'directness', 'curiosity', 'playfulness', 'boundary'],
          description:
            'warmth 温柔 / directness 直率 / curiosity 好奇 / playfulness 玩心 / boundary 边界',
        },
        note: {
          type: 'string',
          description: '这条证据是什么事、为什么算。会留在账本里当证据正文，必须具体，不能空泛。',
        },
        polarity: {
          type: 'integer',
          enum: [1, -1],
          description: '1 = 支持这个特质；-1 = 相反方向。默认 1',
        },
        delta: {
          type: 'number',
          description: '强度 0~1，默认 0.2。只有确实很重的事才给 0.5 以上。',
        },
        memory_id: {
          type: 'string',
          description:
            '支撑这条证据的那条记忆的 id（从 su_recall / su_remember 拿）。' +
            '有就填，这样以后能点开看原句，记忆被纠正时这条证据也会跟着扣回去。',
        },
      },
      required: ['trait', 'note'],
    },
    output: jsonOutput('这条证据是否落地、当前证据数与值'),
  };
}

export function soulStatusToolSpec() {
  return {
    name: SOUL_STATUS_TOOL,
    description:
      '查看人格三段的开关状态与内容首行（身份 / 性格 / 沉浸思考）。' +
      '三段存在 persona.json 里，各自独立开关，改了下一轮就生效。',
    parameters: { type: 'object', properties: {}, required: [] },
    output: jsonOutput('人格三段的开关与首行'),
  };
}

export function soulToggleToolSpec() {
  return {
    name: SOUL_TOGGLE_TOOL,
    description:
      '打开或关闭人格的某一段：identity=身份，personality=性格，thinking=沉浸思考。' +
      '关闭后下一轮立刻不再注入，不用重启 DSH。',
    parameters: {
      type: 'object',
      properties: {
        section: {
          type: 'string',
          enum: ['identity', 'personality', 'thinking'],
          description: '要切换的段',
        },
        enabled: { type: 'boolean', description: 'true = 打开，false = 关闭' },
      },
      required: ['section', 'enabled'],
    },
    output: jsonOutput('切换结果'),
  };
}

export function gapToolSpec() {
  return {
    name: GAP_TOOL,
    description:
      '缺口池：装「我真心没弄明白的事」，不是待办。' +
      '每一条从种下开始绷着，时间越久张力越高；高到阈值会自己出现在我的提示词里，下次自然会想起。' +
      'action=add 种一条；list 看池子；close 闭合一条（想通了，或者被回答了）。' +
      '别把「我接下来要做什么」种进来——那是待办。' +
      '也别为了显得勤快而种：种下去的应该真的没解开。',
    parameters: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['add', 'list', 'close'], description: '要做什么' },
        text: { type: 'string', description: 'add 时必填：那条没弄明白的事，一句话说清' },
        id: { type: 'string', description: 'close 时必填：执念 id' },
        strength: { type: 'number', description: 'add 时可选：这条绷得多紧，0.1~3.0，默认 1' },
        note: { type: 'string', description: '可选备注' },
      },
      required: ['action'],
    },
    output: jsonOutput('执念池子的当前状态'),
  };
}

export function forgetToolSpec() {
  return {
    name: FORGET_TOOL,
    description:
      '纠正一条记忆：用户说「你记错了」「不是这样的」时调用。' +
      '会把该记忆标记为失效、移出向量索引，并在之后永不作为依据召回（旧版本仍保留可查）。',
    parameters: {
      type: 'object',
      properties: {
        memory_id: { type: 'string', description: '要纠正的记忆 id（从 su_recall 结果里拿）' },
        note: { type: 'string', description: '纠正原因，写给人看' },
      },
      required: ['memory_id'],
    },
    output: jsonOutput('纠正结果'),
  };
}

export function healthToolSpec() {
  return {
    name: HEALTH_TOOL,
    description: '检查溯忆服务是否健康（记忆服务、向量层、人格状态）。出问题时先看它。',
    parameters: { type: 'object', properties: {}, required: [] },
    output: jsonOutput('服务健康状态'),
  };
}
