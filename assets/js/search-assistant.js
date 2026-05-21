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
                    keys: ['title', 'content', 'summary'],
                    threshold: 0.4,
                    includeScore: true,
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

    function searchRelevantPosts(query, limit = 3) {
        if (!searchIndex || searchIndex.length === 0) return [];

        let results = [];
        if (fuse) {
            results = fuse.search(query).map((result) => result.item);
        } else {
            const keywords = query.toLowerCase().split(/\s+/);
            results = searchIndex.map((post) => {
                const text = `${post.title} ${post.content} ${post.summary}`.toLowerCase();
                const score = keywords.reduce((acc, keyword) => acc + (text.includes(keyword) ? 1 : 0), 0);
                return { ...post, score };
            }).filter((post) => post.score > 0).sort((a, b) => b.score - a.score);
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

        const results = searchRelevantPosts(query, 10);
        if (results.length === 0) {
            searchResults.innerHTML = '<li class="no-results">No results found</li>';
            return;
        }

        searchResults.innerHTML = results.map((post) => {
            const title = escapeHtml(post.title);
            const summary = escapeHtml((post.summary || post.content || '').substring(0, 100));
            return `<li><a href="${post.permalink}"><span class="title">${title}</span><span class="summary">${summary}...</span></a></li>`;
        }).join('');
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

        const sources = searchRelevantPosts(question, 3);
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
                    sources: sources.map((post) => ({
                        title: post.title,
                        url: post.permalink,
                        excerpt: (post.content || post.summary || '').substring(0, 1200)
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

        const links = sources.map((post) => {
            return `<a class="source-link" href="${post.permalink}" target="_blank" rel="noopener noreferrer">${escapeHtml(post.title)}</a>`;
        }).join('');

        return `<div class="sources"><div class="sources-title">Sources</div>${links}</div>`;
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
