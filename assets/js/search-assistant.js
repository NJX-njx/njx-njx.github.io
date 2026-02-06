(function () {
    const CONFIG = {
        aistudio: {
            baseUrl: 'https://aistudio.baidu.com/llm/lmapi/v3/chat/completions',
            model: 'deepseek-v3'
        },
        deepseek: {
            baseUrl: 'https://api.deepseek.com/chat/completions',
            model: 'deepseek-chat'
        },
        openai: {
            baseUrl: 'https://api.openai.com/v1/chat/completions',
            model: 'gpt-4o-mini'
        }
    };

    let searchIndex = null;
    let fuse = null;

    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const apiKeyInput = document.getElementById('api-key');
    const apiProviderSelect = document.getElementById('api-provider');
    const saveApiKeyButton = document.getElementById('save-api-key');
    const apiStatus = document.getElementById('api-status');

    if (!chatMessages || !chatInput || !sendButton) {
        return;
    }

    async function init() {
        const savedProvider = localStorage.getItem('ai-provider');
        const savedKey = localStorage.getItem('ai-api-key');
        if (savedProvider) apiProviderSelect.value = savedProvider;
        if (savedKey) {
            apiKeyInput.value = savedKey;
            apiStatus.textContent = '✓ Saved';
            apiStatus.style.color = '#27ae60';
        }

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

        saveApiKeyButton.addEventListener('click', saveApiConfig);
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    function saveApiConfig() {
        const provider = apiProviderSelect.value;
        const key = apiKeyInput.value.trim();

        if (key) {
            localStorage.setItem('ai-provider', provider);
            localStorage.setItem('ai-api-key', key);
            apiStatus.textContent = '✓ Saved';
            apiStatus.style.color = '#27ae60';
        } else {
            localStorage.removeItem('ai-api-key');
            apiStatus.textContent = '✗ No key';
            apiStatus.style.color = '#e74c3c';
        }
    }

    function searchRelevantPosts(query, limit = 3) {
        if (!searchIndex || searchIndex.length === 0) return [];

        if (fuse) {
            const results = fuse.search(query);
            return results.slice(0, limit).map((r) => r.item);
        }

        const keywords = query.toLowerCase().split(/\s+/);
        const scored = searchIndex.map((post) => {
            const text = `${post.title} ${post.content} ${post.summary}`.toLowerCase();
            const score = keywords.reduce((acc, kw) => acc + (text.includes(kw) ? 1 : 0), 0);
            return { ...post, score };
        });

        return scored
            .filter((post) => post.score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    function addMessage(contentNodes, isUser, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.appendChild(contentNodes);

        if (sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';

            const titleDiv = document.createElement('div');
            titleDiv.className = 'sources-title';
            titleDiv.textContent = '📚 Related Articles:';
            sourcesDiv.appendChild(titleDiv);

            sources.forEach((source) => {
                const link = document.createElement('a');
                link.href = source.permalink;
                link.className = 'source-link';
                link.textContent = `→ ${source.title}`;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                sourcesDiv.appendChild(link);
            });

            contentDiv.appendChild(sourcesDiv);
        }

        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return messageDiv;
    }

    function addLoadingMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        messageDiv.id = 'loading-message';
        messageDiv.innerHTML = '<div class="loading"><span></span><span></span><span></span></div>';
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeLoadingMessage() {
        const loading = document.getElementById('loading-message');
        if (loading) loading.remove();
    }

    function setErrorMessage(text) {
        const errorSpan = document.createElement('span');
        errorSpan.className = 'error-message';
        errorSpan.textContent = `❌ Error: ${text}`;
        addMessage(errorSpan, false);
    }

    async function sendMessage() {
        const query = chatInput.value.trim();
        if (!query) return;

        const apiKey = localStorage.getItem('ai-api-key');
        const provider = localStorage.getItem('ai-provider') || 'deepseek';

        if (!apiKey) {
            const wrapper = document.createElement('div');
            const line1 = document.createElement('p');
            line1.textContent = '⚠️ 请先配置 API Key！展开上方的 "Configure API Key" 进行设置。';
            const line2 = document.createElement('p');
            const link = document.createElement('a');
            link.href = 'https://platform.deepseek.com/';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'DeepSeek 官方 API';
            line2.appendChild(document.createTextNode('💡 推荐使用 '));
            line2.appendChild(link);
            line2.appendChild(document.createTextNode('（注册即送额度）'));
            wrapper.appendChild(line1);
            wrapper.appendChild(line2);
            addMessage(wrapper, false);
            return;
        }

        const userText = document.createElement('span');
        userText.textContent = query;
        addMessage(userText, true);
        chatInput.value = '';
        sendButton.disabled = true;

        const relevantPosts = searchRelevantPosts(query);
        let context = '';
        if (relevantPosts.length > 0) {
            context = relevantPosts
                .map(
                    (post) =>
                        `Article: "${post.title}"\nURL: ${post.permalink}\nContent: ${post.content.substring(0, 2000)}...`
                )
                .join('\n\n---\n\n');
        }

        addLoadingMessage();

        try {
            const config = CONFIG[provider];
            const systemPrompt =
                '你是 Njx\'Log 博客的 AI 助手，由倪家兴（Jiaxing Ni）创建。\n\n' +
                '## 你的身份\n' +
                '- 你是一个专注于技术博客的智能助手\n' +
                '- 博客主题涵盖：大模型（LLM）、多模态AI、RAG系统、机器学习等\n\n' +
                '## 你的职责\n' +
                '1. 根据提供的博客文章内容回答用户问题\n' +
                '2. 帮助用户理解技术概念\n' +
                '3. 推荐相关的博客文章\n' +
                '4. 如果问题超出博客内容范围，提供通用但有帮助的回答\n\n' +
                '## 回答风格\n' +
                '- 使用清晰、简洁的语言\n' +
                '- 适当使用 Markdown 格式（加粗、列表、代码块等）\n' +
                '- 引用文章时注明来源\n' +
                '- 如果不确定，诚实说明\n' +
                '- 支持中英文双语回答，根据用户语言自动切换\n\n' +
                '## 博客作者背景\n' +
                '- 北京理工大学 计算机/徐特立英才班\n' +
                '- 专注于：LLM 微调、RAG、多模态融合、RLHF\n' +
                '- 有百度 Hackathon、字节 Xpert 等项目经验';

            const userPrompt = context
                ? `以下是相关的博客文章内容：\n\n${context}\n\n---\n\n用户问题：${query}`
                : `用户问题：${query}\n\n（博客中未找到直接相关的文章）`;

            const response = await fetch(config.baseUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    model: config.model,
                    messages: [
                        { role: 'system', content: systemPrompt },
                        { role: 'user', content: userPrompt }
                    ],
                    temperature: 0.7,
                    max_tokens: 2000
                })
            });

            removeLoadingMessage();

            if (!response.ok) {
                let message = 'API request failed';
                try {
                    const error = await response.json();
                    message = error.error?.message || message;
                } catch (err) {
                    message = `${message} (${response.status})`;
                }
                throw new Error(message);
            }

            const data = await response.json();
            const answer = data.choices?.[0]?.message?.content || '';

            const formattedAnswer = formatMarkdownSafe(answer);
            addMessage(formattedAnswer, false, relevantPosts);
        } catch (error) {
            removeLoadingMessage();
            setErrorMessage(error.message || 'Unknown error');
        }

        sendButton.disabled = false;
        chatInput.focus();
    }

    function formatMarkdownSafe(text) {
        const container = document.createElement('div');
        const lines = String(text || '').split('\n');
        let paragraph = document.createElement('p');

        lines.forEach((line, index) => {
            if (line.trim() === '') {
                if (paragraph.childNodes.length > 0) {
                    container.appendChild(paragraph);
                    paragraph = document.createElement('p');
                }
                return;
            }

            const formattedLine = parseInlineMarkdown(line);
            paragraph.appendChild(formattedLine);

            if (index < lines.length - 1) {
                paragraph.appendChild(document.createElement('br'));
            }
        });

        if (paragraph.childNodes.length > 0) {
            container.appendChild(paragraph);
        }

        return container;
    }

    function parseInlineMarkdown(line) {
        const fragment = document.createDocumentFragment();
        const regex = /\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*([^*]+)\*/g;
        let lastIndex = 0;
        let match;

        while ((match = regex.exec(line)) !== null) {
            if (match.index > lastIndex) {
                fragment.appendChild(document.createTextNode(line.slice(lastIndex, match.index)));
            }

            if (match[1]) {
                const strong = document.createElement('strong');
                strong.textContent = match[1];
                fragment.appendChild(strong);
            } else if (match[2]) {
                const code = document.createElement('code');
                code.textContent = match[2];
                fragment.appendChild(code);
            } else if (match[3] && match[4]) {
                const link = document.createElement('a');
                link.textContent = match[3];
                link.href = match[4];
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                fragment.appendChild(link);
            } else if (match[5]) {
                const em = document.createElement('em');
                em.textContent = match[5];
                fragment.appendChild(em);
            }

            lastIndex = regex.lastIndex;
        }

        if (lastIndex < line.length) {
            fragment.appendChild(document.createTextNode(line.slice(lastIndex)));
        }

        const span = document.createElement('span');
        span.appendChild(fragment);
        return span;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
