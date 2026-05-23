(function () {
    let searchIndex = null;
    let fuse = null;
    let abortController = null;
    let chatHistory = [];

    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');
    const assistant = document.getElementById('ai-assistant');
    const assistantEndpoint = assistant?.dataset.endpoint || '';
    const assistantResults = document.getElementById('assistantResults');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');

    async function init() {
        await loadSearchIndex();
        bindSearch();
        bindAssistant();
    }

    async function loadSearchIndex() {
        try {
            const response = await fetch('/index.json');
            searchIndex = await response.json();

            if (typeof Fuse !== 'undefined') {
                fuse = new Fuse(searchIndex, {
                    keys: [
                        { name: 'title', weight: 0.4 },
                        { name: 'section', weight: 0.28 },
                        { name: 'summary', weight: 0.16 },
                        { name: 'content', weight: 0.12 },
                        { name: 'tags', weight: 0.04 }
                    ],
                    threshold: 0.35,
                    includeScore: true,
                    includeMatches: true,
                    ignoreLocation: true
                });
            }
        } catch (error) {
            console.error('Failed to load search index:', error);
        }
    }

    function bindSearch() {
        if (!searchInput || !searchResults) return;

        let searchDebounce = null;
        searchInput.addEventListener('input', function () {
            clearTimeout(searchDebounce);
            searchDebounce = setTimeout(() => performSearch(this.value), 150);
        });

        searchInput.addEventListener('keydown', function (event) {
            if (event.key === 'Escape') {
                searchResults.innerHTML = '';
                searchInput.blur();
            }
        });
    }

    function bindAssistant() {
        if (!assistantResults || !chatInput) return;

        if (sendButton) {
            sendButton.addEventListener('click', () => handleAsk());
        }
        chatInput.addEventListener('input', autoResizeChatInput);
        chatInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleAsk();
            }
        });
        autoResizeChatInput();
    }

    function autoResizeChatInput() {
        if (!chatInput) return;
        chatInput.style.height = 'auto';
        chatInput.style.height = `${Math.max(46, chatInput.scrollHeight)}px`;
    }

    function searchRelevantChunks(query, limit = 5) {
        if (!searchIndex || searchIndex.length === 0) return [];

        let results = [];
        if (fuse) {
            const merged = new Map();
            const searches = [
                { query, weight: 1.2 },
                ...queryTokens(query).map((token) => ({ query: token, weight: Math.max(0.35, tokenImportance(token) * 0.45) }))
            ];

            searches.forEach((search) => {
                fuse.search(search.query, { limit: limit * 4 }).forEach((result) => {
                    const key = chunkKey(result.item);
                    const existing = merged.get(key) || {
                        ...result.item,
                        _matches: [],
                        _score: 0
                    };
                    existing._score += Math.max(0, 1 - result.score) * search.weight;
                    existing._matches = mergeMatches(existing._matches, result.matches || []);
                    merged.set(key, existing);
                });
            });

            searchIndex.forEach((chunk) => {
                const lexicalScore = lexicalChunkScore(chunk, query);
                if (lexicalScore <= 0) return;
                const key = chunkKey(chunk);
                const existing = merged.get(key) || {
                    ...chunk,
                    _matches: [],
                    _score: 0
                };
                existing._score += lexicalScore;
                merged.set(key, existing);
            });

            results = Array.from(merged.values()).sort((a, b) => b._score - a._score);
        } else {
            const keywords = queryTokens(query);
            results = searchIndex.map((chunk) => {
                const score = keywords.reduce((acc, keyword) => {
                    if (!keyword) return acc;
                    return acc + lexicalChunkScore(chunk, keyword);
                }, 0);
                return { ...chunk, _score: score, _matches: [] };
            }).filter((chunk) => chunk._score > 0).sort((a, b) => b._score - a._score);
        }

        return results.slice(0, limit);
    }

    function performSearch(query) {
        if (!searchResults) return;

        query = query.trim();
        if (!query) {
            searchResults.innerHTML = '';
            return;
        }

        const results = searchRelevantChunks(query, 12);
        if (results.length === 0) {
            searchResults.innerHTML = '<li class="no-results">No results found</li>';
            return;
        }

        searchResults.innerHTML = results.map((chunk) => renderSearchResult(chunk, query)).join('');
    }

    async function handleAsk() {
        const question = chatInput.value.trim();
        if (!question) return;

        if (!assistantEndpoint) {
            renderAssistantError('Assistant endpoint is not configured yet.');
            return;
        }

        if (abortController) abortController.abort();
        abortController = new AbortController();

        const sources = searchRelevantChunks(question, 5);
        renderAssistantLoading(question);
        if (sendButton) {
            sendButton.disabled = true;
        }

        try {
            const response = await fetch(assistantEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question,
                    sources: sources.map((chunk) => ({
                        title: sourceTitle(chunk),
                        url: chunkUrl(chunk),
                        excerpt: sourceExcerpt(chunk)
                    })),
                    history: chatHistory.slice(-6)
                }),
                signal: abortController.signal
            });

            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.detail || data.error || `Request failed (${response.status})`);
            }

            const answer = data.answer || 'No answer returned.';
            chatHistory.push({ role: 'user', content: question });
            chatHistory.push({ role: 'assistant', content: answer });
            chatInput.value = '';
            autoResizeChatInput();
            renderAssistantAnswer(question, answer, sources);
        } catch (error) {
            if (error.name !== 'AbortError') {
                renderAssistantError(error.message || 'Assistant request failed.');
            }
        } finally {
            abortController = null;
            if (sendButton) {
                sendButton.disabled = false;
            }
            chatInput.focus();
        }
    }

    function renderAssistantLoading(question) {
        assistantResults.innerHTML = `
            <li class="assistant-result">
                <span class="assistant-question">${escapeHtml(question)}</span>
                <span class="assistant-loading">Thinking...</span>
            </li>
        `;
    }

    function renderAssistantAnswer(question, answer, sources) {
        assistantResults.innerHTML = `
            <li class="assistant-result">
                <span class="assistant-question">${escapeHtml(question)}</span>
                <div class="assistant-answer">${parseMarkdown(answer)}</div>
                ${renderSources(sources)}
            </li>
        `;
    }

    function renderAssistantError(message) {
        assistantResults.innerHTML = `
            <li class="assistant-result assistant-error">
                ${escapeHtml(message)}
            </li>
        `;
    }

    function renderSources(sources) {
        if (!sources.length) return '';

        const links = sources.map((chunk) => {
            return `<a class="source-link" href="${escapeHtml(chunkUrl(chunk))}" target="_blank" rel="noopener noreferrer">${escapeHtml(sourceTitle(chunk))}</a>`;
        }).join('');

        return `<div class="sources"><div class="sources-title">Sources</div>${links}</div>`;
    }

    function renderSearchResult(chunk, query) {
        const title = escapeHtml(chunk.title || 'Untitled');
        const section = chunk.section && chunk.section !== chunk.title
            ? `<span class="section">${escapeHtml(chunk.section)}</span>`
            : '';
        const metaParts = [];
        if (chunk.date) metaParts.push(escapeHtml(chunk.date));
        if (Array.isArray(chunk.tags) && chunk.tags.length) {
            metaParts.push(escapeHtml(chunk.tags.slice(0, 4).join(' / ')));
        }
        const meta = metaParts.length ? `<span class="meta">${metaParts.join(' · ')}</span>` : '';
        const snippet = buildSnippet(chunk, query);

        return `
            <li>
                <a href="${escapeHtml(chunkUrl(chunk))}">
                    <span class="title">${title}</span>
                    ${section}
                    <span class="summary">${snippet}</span>
                    ${meta}
                </a>
            </li>
        `;
    }

    function chunkKey(chunk) {
        return `${chunk.permalink || ''}#${chunk.anchor || ''}`;
    }

    function mergeMatches(current, incoming) {
        const merged = Array.isArray(current) ? current.slice() : [];
        (Array.isArray(incoming) ? incoming : []).forEach((match) => {
            const exists = merged.some((item) => item.key === match.key && item.value === match.value);
            if (!exists) merged.push(match);
        });
        return merged;
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
        synonyms.forEach(([pattern, terms]) => {
            if (pattern.test(queryText)) expanded.push(...terms);
        });
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

    function buildSnippet(chunk, query) {
        const match = bestTextMatch(chunk._matches);
        if (match) {
            const text = matchText(chunk, match.key);
            const range = bestRange(match.indices);
            if (text && range) {
                return highlightSnippet(text, range[0], range[1]);
            }
        }

        const content = chunk.content || chunk.summary || '';
        const queryRange = firstKeywordRange(content, query);
        if (queryRange) {
            return highlightSnippet(content, queryRange[0], queryRange[1]);
        }

        return escapeHtml(compactSnippet(content, 180));
    }

    function bestTextMatch(matches) {
        if (!Array.isArray(matches)) return null;
        const priority = ['content', 'summary', 'section', 'title', 'tags'];
        for (const key of priority) {
            const match = matches.find((item) => item.key === key && item.indices && item.indices.length);
            if (match) return match;
        }
        return null;
    }

    function bestRange(indices) {
        if (!Array.isArray(indices) || indices.length === 0) return null;
        return indices.reduce((best, range) => {
            if (!best) return range;
            return (range[1] - range[0]) > (best[1] - best[0]) ? range : best;
        }, null);
    }

    function matchText(chunk, key) {
        const value = chunk[key];
        if (Array.isArray(value)) return value.join(' ');
        return value || '';
    }

    function highlightSnippet(text, start, end) {
        const windowStart = Math.max(0, start - 70);
        const windowEnd = Math.min(text.length, end + 90);
        const before = text.slice(windowStart, start);
        const hit = text.slice(start, end + 1);
        const after = text.slice(end + 1, windowEnd);
        const prefix = windowStart > 0 ? '...' : '';
        const suffix = windowEnd < text.length ? '...' : '';

        return [
            escapeHtml(prefix + before),
            `<mark>${escapeHtml(hit)}</mark>`,
            escapeHtml(after + suffix)
        ].join('');
    }

    function firstKeywordRange(text, query) {
        if (!text || !query) return null;
        const lowerText = text.toLowerCase();
        const lowerQuery = query.trim().toLowerCase();
        if (!lowerQuery) return null;
        const directIndex = lowerText.indexOf(lowerQuery);
        if (directIndex >= 0) return [directIndex, directIndex + lowerQuery.length - 1];

        const keywords = lowerQuery.split(/\s+/).filter((keyword) => keyword.length > 1);
        for (const keyword of keywords) {
            const index = lowerText.indexOf(keyword);
            if (index >= 0) return [index, index + keyword.length - 1];
        }
        return null;
    }

    function compactSnippet(text, length) {
        const compact = String(text || '').replace(/\s+/g, ' ').trim();
        return compact.length > length ? `${compact.slice(0, length)}...` : compact;
    }

    function chunkUrl(chunk) {
        if (!chunk) return '#';
        return chunk.anchor ? `${chunk.permalink}#${chunk.anchor}` : chunk.permalink;
    }

    function sourceTitle(chunk) {
        if (!chunk) return 'Untitled';
        if (chunk.section && chunk.section !== chunk.title) {
            return `${chunk.title} — ${chunk.section}`;
        }
        return chunk.title || 'Untitled';
    }

    function sourceExcerpt(chunk) {
        const heading = chunk.section ? `${chunk.section}\n\n` : '';
        return `${heading}${chunk.content || chunk.summary || ''}`.substring(0, 1200);
    }

    function parseMarkdown(text) {
        if (!text) return '';

        const codeBlocks = [];
        const inlineCodes = [];

        let processed = text.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
            codeBlocks.push({ lang, code });
            return placeholder;
        });

        processed = processed.replace(/`([^`]+)`/g, (match, code) => {
            const placeholder = `__INLINE_CODE_${inlineCodes.length}__`;
            inlineCodes.push(code);
            return placeholder;
        });

        processed = escapeHtml(processed)
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
                const trimmedUrl = url.trim();
                if (/^(https?:\/\/|mailto:|\/|#)/.test(trimmedUrl) || !/^[a-z]+:/i.test(trimmedUrl)) {
                    const safeUrl = trimmedUrl.replace(/"/g, '&quot;');
                    return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${label}</a>`;
                }
                return `${label} (${trimmedUrl})`;
            })
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');

        inlineCodes.forEach((code, index) => {
            processed = processed.replace(`__INLINE_CODE_${index}__`, `<code>${escapeHtml(code)}</code>`);
        });

        codeBlocks.forEach((block, index) => {
            processed = processed.replace(
                `__CODE_BLOCK_${index}__`,
                `<pre><code class="language-${escapeHtml(block.lang)}">${escapeHtml(block.code)}</code></pre>`
            );
        });

        return processed;
    }

    function escapeHtml(text) {
        if (!text) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
