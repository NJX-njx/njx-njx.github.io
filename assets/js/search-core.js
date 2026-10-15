// Ranking for the chunked search index. Shared by the search page and
// scripts/search_quality.mjs, so the test exercises the code readers use.

const FUSE_KEYS = [
    { name: 'title', weight: 0.4 },
    { name: 'section', weight: 0.28 },
    { name: 'summary', weight: 0.16 },
    { name: 'content', weight: 0.12 },
    { name: 'tags', weight: 0.04 }
];

const LEXICAL_FIELDS = [
    { name: 'title', weight: 5 },
    { name: 'section', weight: 4 },
    { name: 'summary', weight: 2.5 },
    { name: 'tags', weight: 2 },
    { name: 'content', weight: 1 }
];

const STOP_WORDS = new Set([
    'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'do', 'does',
    'for', 'from', 'how', 'in', 'into', 'is', 'it', 'of', 'on', 'or', 'should',
    'the', 'to', 'what', 'when', 'where', 'which', 'who', 'why', 'with'
]);

const WHOLE_QUERY_WEIGHT = 1.2;
const TOKEN_QUERY_WEIGHT = 0.45;

export function createSearch(index, Fuse) {
    const fuse = new Fuse(index, {
        keys: FUSE_KEYS,
        threshold: 0.35,
        includeScore: true,
        includeMatches: true,
        ignoreLocation: true
    });

    return function search(query, limit) {
        const tokens = queryTokens(query);
        const terms = withCjkBigrams(tokens);
        const merged = new Map();

        function add(chunk, score, matches) {
            const key = `${chunk.permalink}#${chunk.anchor}`;
            const entry = merged.get(key) || { ...chunk, _matches: [], _score: 0 };
            entry._score += score;
            entry._matches = mergeMatches(entry._matches, matches);
            merged.set(key, entry);
        }

        const fuseQueries = [
            { text: query, weight: WHOLE_QUERY_WEIGHT },
            ...tokens.map((token) => ({ text: token, weight: TOKEN_QUERY_WEIGHT }))
        ];
        fuseQueries.forEach(({ text, weight }) => {
            fuse.search(text, { limit: limit * 4 }).forEach((result) => {
                add(result.item, Math.max(0, 1 - result.score) * weight, result.matches || []);
            });
        });

        index.forEach((chunk) => {
            const score = lexicalScore(chunk, terms);
            if (score > 0) add(chunk, score, []);
        });

        return Array.from(merged.values())
            .sort((a, b) => b._score - a._score)
            .slice(0, limit);
    };
}

export function queryTokens(query) {
    const tokens = String(query)
        .toLowerCase()
        .split(/[\s,.;:!?()[\]{}"'`/\\|，。；：！？（）、“”]+/)
        .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
    return Array.from(new Set(tokens));
}

// Chinese is written without spaces, so a query like 序列级重要性采样 arrives as one
// token. Overlapping bigrams let it match text that orders the words differently.
function withCjkBigrams(tokens) {
    const terms = new Set(tokens);
    tokens.forEach((token) => {
        (token.match(/[\u3400-\u9fff]{3,}/g) || []).forEach((run) => {
            for (let i = 0; i < run.length - 1; i += 1) terms.add(run.slice(i, i + 2));
        });
    });
    return Array.from(terms);
}

function lexicalScore(chunk, tokens) {
    return tokens.reduce((total, token) => {
        const normalizedToken = normalize(token);
        return total + LEXICAL_FIELDS.reduce((score, field) => {
            const raw = fieldText(chunk, field.name).toLowerCase();
            return tokenMatches(raw, normalize(raw), token, normalizedToken) ? score + field.weight : score;
        }, 0);
    }, 0);
}

function fieldText(chunk, name) {
    const value = chunk[name];
    return Array.isArray(value) ? value.join(' ') : String(value || '');
}

function normalize(value) {
    return value.replace(/[-_]+/g, ' ');
}

// Short ASCII tokens such as "rl" or "kl" would otherwise match inside other words.
function tokenMatches(raw, normalized, token, normalizedToken) {
    if (/^[a-z0-9 ]{1,3}$/.test(normalizedToken)) {
        return new RegExp(`(^|[^a-z0-9])${normalizedToken}($|[^a-z0-9])`).test(normalized);
    }
    return raw.includes(token) || normalized.includes(normalizedToken);
}

function mergeMatches(current, incoming) {
    const merged = current.slice();
    incoming.forEach((match) => {
        if (!merged.some((item) => item.key === match.key && item.value === match.value)) merged.push(match);
    });
    return merged;
}
