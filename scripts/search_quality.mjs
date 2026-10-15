// Ranking regression test for the site search.
// Usage: node scripts/search_quality.mjs [public-dir]   (run after `hugo`)
// Each query must find its expected post within the top 3 results.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createSearch } from '../assets/js/search-core.js';

const require = createRequire(import.meta.url);
const Fuse = require('../themes/PaperMod/assets/js/fuse.basic.min.js');

const CASES = {
    'index.json': [
        ['熵坍缩', 'dapo-clip-higher-entropy-collapse'],
        ['Clip-Higher 裁剪上界', 'dapo-clip-higher-entropy-collapse'],
        ['截断 超长惩罚', 'dapo-overlong-reward-shaping'],
        ['长度偏差 难度偏差', 'dr-grpo-length-and-difficulty-bias'],
        ['prompt masking', 'forge-sft-anatomy'],
        ['组大小', 'grpo-group-size-tradeoffs'],
        ['序列级重要性采样', 'gspo-sequence-level-importance-sampling'],
        ['递归自我改进', 'rsi-self-improvement-loops'],
        ['训练推理不一致', 'tis-train-inference-mismatch'],
        ['token-mean sample-mean', 'token-mean-vs-sample-mean'],
        ['多轮 奖励归因', 'turn-level-reward-credit-assignment']
    ],
    'en/index.json': [
        ['entropy collapse', 'dapo-clip-higher-entropy-collapse'],
        ['overlong filtering truncated responses', 'dapo-overlong-reward-shaping'],
        ['length bias difficulty bias', 'dr-grpo-length-and-difficulty-bias'],
        ['chat template prompt masking', 'forge-sft-anatomy'],
        ['group size', 'grpo-group-size-tradeoffs'],
        ['sequence-level importance ratio', 'gspo-sequence-level-importance-sampling'],
        ['recursive self-improvement', 'rsi-self-improvement-loops'],
        ['train-inference mismatch', 'tis-train-inference-mismatch'],
        ['loss aggregation token-mean', 'token-mean-vs-sample-mean'],
        ['credit assignment multi-turn', 'turn-level-reward-credit-assignment']
    ]
};

const publicDir = process.argv[2] || 'public';
let failures = 0;

for (const [file, cases] of Object.entries(CASES)) {
    const index = JSON.parse(fs.readFileSync(path.join(publicDir, file), 'utf8'));
    const search = createSearch(index, Fuse);
    const ranks = cases.map(([query, slug]) => {
        const results = search(query, 5);
        const rank = results.findIndex((chunk) => chunk.permalink.includes(`/posts/${slug}/`)) + 1;
        if (rank === 0 || rank > 3) {
            failures += 1;
            const top = results.slice(0, 3).map((chunk) => chunk.permalink.split('/').filter(Boolean).pop());
            console.log(`  MISS ${file} "${query}" -> expected ${slug}, got ${top.join(', ')}`);
        }
        return rank;
    });
    const hit1 = ranks.filter((rank) => rank === 1).length / ranks.length;
    const mrr = ranks.reduce((sum, rank) => sum + (rank ? 1 / rank : 0), 0) / ranks.length;
    console.log(`${file}: ${cases.length} queries, Hit@1=${hit1.toFixed(2)} MRR=${mrr.toFixed(2)}`);
}

if (failures) {
    console.log(`${failures} queries missed the top 3`);
    process.exit(1);
}
