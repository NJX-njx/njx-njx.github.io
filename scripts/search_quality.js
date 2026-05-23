const fs = require('fs');
const path = require('path');

const Fuse = require('../themes/PaperMod/assets/js/fuse.basic.min.js');

const indexPath = process.argv[2] || path.join(process.cwd(), 'public', 'index.json');

const TEST_QUERIES = [
  {
    query: 'agent design tool use memory loop',
    expected: ['agent design']
  },
  {
    query: 'how should an agent decide when to use tools',
    expected: ['agent design', 'tools have contracts']
  },
  {
    query: 'eval benchmark failure cases metrics',
    expected: ['eval']
  },
  {
    query: 'how to evaluate an AI product beyond a demo',
    expected: ['eval', 'ai-travel-planner']
  },
  {
    query: 'agentic RL reward policy online offline evaluation',
    expected: ['agentic-rl']
  },
  {
    query: 'reward design for agents',
    expected: ['agentic-rl']
  },
  {
    query: 'travel planner benchmark OOD long context',
    expected: ['travel planner']
  },
  {
    query: 'retrieval context engineering RAG',
    expected: ['retrieval', 'rag', 'ai-note-assistant']
  },
  {
    query: 'Hugo GitHub Pages deployment',
    expected: ['hugo', 'deployment', 'cicd']
  },
  {
    query: '你做过哪些评测相关项目',
    expected: ['eval', 'evaluation', 'benchmark', 'xpert', 'travel']
  }
];

const PRESETS = {
  current: {
    title: 0.4,
    section: 0.28,
    summary: 0.16,
    content: 0.12,
    tags: 0.04
  },
  sectionForward: {
    title: 0.34,
    section: 0.34,
    summary: 0.14,
    content: 0.12,
    tags: 0.06
  },
  contentForward: {
    title: 0.28,
    section: 0.26,
    summary: 0.14,
    content: 0.26,
    tags: 0.06
  },
  tagsForward: {
    title: 0.34,
    section: 0.28,
    summary: 0.12,
    content: 0.14,
    tags: 0.12
  }
};

function main() {
  const index = readIndex(indexPath);
  let best = null;

  for (const [name, weights] of Object.entries(PRESETS)) {
    const report = evaluatePreset(index, name, weights);
    printReport(report);
    if (!best || report.mrr > best.mrr || (report.mrr === best.mrr && report.hitAt3 > best.hitAt3)) {
      best = report;
    }
  }

  console.log(`\nBest preset: ${best.name}`);
  console.log(`Hit@1=${formatMetric(best.hitAt1)} Hit@3=${formatMetric(best.hitAt3)} MRR=${formatMetric(best.mrr)}`);
}

function readIndex(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '');
  const index = JSON.parse(raw);
  if (!Array.isArray(index)) {
    throw new Error('Search index must be a JSON array');
  }
  return index;
}

function evaluatePreset(index, name, weights) {
  const fuse = new Fuse(index, {
    keys: [
      { name: 'title', weight: weights.title },
      { name: 'section', weight: weights.section },
      { name: 'summary', weight: weights.summary },
      { name: 'content', weight: weights.content },
      { name: 'tags', weight: weights.tags }
    ],
    threshold: 0.35,
    includeScore: true,
    includeMatches: true,
    ignoreLocation: true
  });

  const rows = TEST_QUERIES.map((test) => {
    const results = hybridSearch(fuse, index, test.query, 5);
    const rank = firstRelevantRank(results, test.expected);
    return {
      query: test.query,
      expected: test.expected,
      rank,
      top: results.slice(0, 5).map((result) => ({
        score: result.score,
        title: result.item.title,
        section: result.item.section,
        anchor: result.item.anchor
      }))
    };
  });

  return {
    name,
    weights,
    rows,
    hitAt1: average(rows.map((row) => row.rank === 1 ? 1 : 0)),
    hitAt3: average(rows.map((row) => row.rank > 0 && row.rank <= 3 ? 1 : 0)),
    hitAt5: average(rows.map((row) => row.rank > 0 && row.rank <= 5 ? 1 : 0)),
    mrr: average(rows.map((row) => row.rank > 0 ? 1 / row.rank : 0))
  };
}

function hybridSearch(fuse, index, query, limit) {
  const merged = new Map();
  const searches = [
    { query, weight: 1.2 },
    ...queryTokens(query).map((token) => ({ query: token, weight: Math.max(0.35, tokenImportance(token) * 0.45) }))
  ];

  for (const search of searches) {
    for (const result of fuse.search(search.query, { limit: limit * 4 })) {
      const key = chunkKey(result.item);
      const existing = merged.get(key) || { item: result.item, score: 0 };
      existing.score += Math.max(0, 1 - result.score) * search.weight;
      merged.set(key, existing);
    }
  }

  for (const chunk of index) {
    const lexicalScore = lexicalChunkScore(chunk, query);
    if (lexicalScore <= 0) continue;
    const key = chunkKey(chunk);
    const existing = merged.get(key) || { item: chunk, score: 0 };
    existing.score += lexicalScore;
    merged.set(key, existing);
  }

  return Array.from(merged.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((result) => ({
      item: result.item,
      score: result.score
    }));
}

function chunkKey(chunk) {
  return `${chunk.permalink || ''}#${chunk.anchor || ''}`;
}

function queryTokens(query) {
  const stopWords = new Set([
    'a', 'about', 'ai', 'an', 'and', 'are', 'as', 'at', 'be', 'beyond', 'by',
    'can', 'for', 'from', 'how', 'in', 'into', 'is', 'of', 'on', 'or',
    'should', 'the', 'to', 'what', 'when', 'where', 'which', 'who', 'why',
    'with'
  ]);
  const base = String(query || '')
    .toLowerCase()
    .split(/[\s,.;:!?()[\]{}"'`/\\|]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 1 && !stopWords.has(token));
  const expanded = base.slice();
  const queryText = String(query || '').toLowerCase();
  const synonyms = [
    [/\bevals?\b|evaluat|benchmark|metric|measurement|measure/, ['eval', 'evaluation', 'benchmark', 'metric']],
    [/product|project/, ['product', 'project', 'build', 'built']],
    [/demo/, ['eval', 'product', 'measurement']],
    [/agentic|agent|autonom/, ['agent', 'agentic']],
    [/reward|policy|reinforcement|\brl\b/, ['agentic-rl', 'rl', 'reinforcement', 'reward', 'policy']],
    [/tool|tools|function|api/, ['tool', 'tools', 'tool-use']],
    [/memory|remember|context/, ['memory', 'context']],
    [/retrieval|\brag\b|search|recall/, ['retrieval', 'rag', 'context']],
    [/评测|评估|测评/, ['eval', 'evaluation', 'benchmark', 'metric']],
    [/项目|产品/, ['project', 'product', 'build', 'built']],
    [/智能体|代理/, ['agent', 'agentic']],
    [/强化学习/, ['rl', 'reinforcement', 'reward', 'policy']],
    [/检索|召回/, ['retrieval', 'rag', 'context']],
    [/工具/, ['tool', 'tools']]
  ];
  for (const [pattern, terms] of synonyms) {
    if (pattern.test(queryText)) expanded.push(...terms);
  }
  return Array.from(new Set(expanded));
}

function lexicalChunkScore(chunk, query) {
  const tokens = queryTokens(query);
  if (!tokens.length) return 0;

  const fields = [
    { value: chunk.title, weight: 5 },
    { value: chunk.section, weight: 4 },
    { value: chunk.summary, weight: 2.5 },
    { value: Array.isArray(chunk.tags) ? chunk.tags.join(' ') : '', weight: 2 },
    { value: chunk.content, weight: 1 }
  ];
  const haystack = normalizeSearchText(fields.map((field) => field.value || '').join(' '));
  const phraseScore = queryPhrases(query).reduce((score, phrase) => {
    return haystack.includes(phrase.value) ? score + phrase.weight : score;
  }, 0);

  return tokens.reduce((total, token) => {
    const normalizedToken = normalizeSearchText(token);
    const importance = tokenImportance(token);
    return total + fields.reduce((score, field) => {
      const value = String(field.value || '').toLowerCase();
      const normalizedValue = normalizeSearchText(value);
      if (!tokenMatches(value, normalizedValue, token, normalizedToken)) return score;
      return score + (field.weight * importance);
    }, 0);
  }, phraseScore);
}

function normalizeSearchText(value) {
  return String(value || '').toLowerCase().replace(/[-_]+/g, ' ');
}

function tokenImportance(token) {
  const normalized = normalizeSearchText(token);
  if (['rag', 'retrieval', 'travel', 'planner', 'hugo'].includes(normalized)) return 2.4;
  if (['agentic rl', 'reinforcement', 'reward', 'policy'].includes(normalized)) return 1.8;
  if (['eval', 'evaluation', 'benchmark', 'metric'].includes(normalized)) return 1.7;
  if (['tool use', 'tool', 'tools', 'memory'].includes(normalized)) return 1.35;
  if (['context', 'product', 'project', 'build', 'built', 'agent', 'agentic'].includes(normalized)) return 0.8;
  return 1;
}

function tokenMatches(rawValue, normalizedValue, rawToken, normalizedToken) {
  if (normalizedToken.length <= 3) {
    const pattern = new RegExp(`(^|\\s)${escapeRegExp(normalizedToken)}($|\\s)`);
    return pattern.test(normalizedValue);
  }
  return rawValue.includes(rawToken) || normalizedValue.includes(normalizedToken);
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function queryPhrases(query) {
  const queryText = normalizeSearchText(query);
  const phrases = [
    ['travel planner', 9],
    ['ai note', 8],
    ['agent design', 8],
    ['agentic rl', 8],
    ['context engineering', 6],
    ['long context', 5],
    ['tool use', 5]
  ];
  return phrases
    .filter(([phrase]) => queryText.includes(phrase))
    .map(([phrase, weight]) => ({ value: phrase, weight }));
}

function firstRelevantRank(results, expectedTerms) {
  for (let index = 0; index < results.length; index += 1) {
    const haystack = searchableText(results[index].item);
    if (expectedTerms.some((term) => haystack.includes(term.toLowerCase()))) {
      return index + 1;
    }
  }
  return 0;
}

function searchableText(item) {
  return [
    item.title,
    item.section,
    item.anchor,
    item.summary,
    item.content,
    Array.isArray(item.tags) ? item.tags.join(' ') : ''
  ].join(' ').toLowerCase();
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function printReport(report) {
  console.log(`\n== ${report.name} ==`);
  console.log(`weights=${JSON.stringify(report.weights)}`);
  console.log(`Hit@1=${formatMetric(report.hitAt1)} Hit@3=${formatMetric(report.hitAt3)} Hit@5=${formatMetric(report.hitAt5)} MRR=${formatMetric(report.mrr)}`);
  for (const row of report.rows) {
    const top = row.top[0];
    const rank = row.rank || '-';
    console.log(`  [rank ${rank}] ${row.query}`);
    console.log(`    top: ${top ? `${top.title} / ${top.section} / ${top.anchor}` : 'no result'}`);
    row.top.slice(0, 5).forEach((candidate, candidateIndex) => {
      console.log(`      ${candidateIndex + 1}. ${candidate.score.toFixed(2)} ${candidate.title} / ${candidate.section}`);
    });
  }
}

function formatMetric(value) {
  return value.toFixed(3);
}

main();
