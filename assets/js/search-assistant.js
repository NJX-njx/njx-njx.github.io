(function () {
    const CONFIG = {
        doubao: {
            baseUrl: 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
            model: 'doubao-seed-1-8-251228'
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

    const SYSTEM_PROMPT = `你是 Njx'Log 博客的 AI 助手，由倪家兴（Jiaxing Ni）创建。

## 你的身份
- 你是一个专注于技术博客的智能助手
- 博客主题涵盖：大模型（LLM）、多模态AI、RAG系统、机器学习等

## 你的职责
1. 根据提供的博客文章内容回答用户问题
2. 帮助用户理解技术概念
3. 推荐相关的博客文章
4. 如果问题超出博客内容范围，提供通用但有帮助的回答

## 回答风格
- 使用清晰、简洁的语言
- 适当使用 Markdown 格式（加粗、列表、代码块等）
- 引用文章时注明来源
- 如果不确定，诚实说明
- 支持中英文双语回答，根据用户语言自动切换`;

    // Storage key prefix for namespacing
    const STORAGE_PREFIX = 'njxlog_';

    let searchIndex = null;
    let fuse = null;
    let abortController = null;
    let chatHistory = []; // Stores {role, content}

    // DOM Elements - Search Box
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');

    // DOM Elements - AI Chat
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const apiKeyInput = document.getElementById('api-key');
    const apiProviderSelect = document.getElementById('api-provider');
    const saveApiKeyButton = document.getElementById('save-api-key');
    const apiStatus = document.getElementById('api-status');
    const clearChatButton = document.getElementById('clear-chat');
    
    // UI Init - gracefully handle missing elements
    if (!chatMessages || !chatInput || !sendButton) return;

    async function init() {
        // Load Settings - use sessionStorage for better security (cleared on tab close)
        const savedProvider = sessionStorage.getItem(STORAGE_PREFIX + 'provider');
        const savedKey = sessionStorage.getItem(STORAGE_PREFIX + 'api-key');
        if (savedProvider && CONFIG[savedProvider]) {
            apiProviderSelect.value = savedProvider;
        } else {
            apiProviderSelect.value = 'doubao'; // Default
        }
        
        if (savedKey) {
            apiKeyInput.value = savedKey;
            apiStatus.textContent = '✓ Session';
            apiStatus.className = 'status-success';
        }

        // Load Search Index
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

        // ===== Search Box Logic =====
        if (searchInput && searchResults) {
            let searchDebounce = null;
            searchInput.addEventListener('input', function() {
                clearTimeout(searchDebounce);
                searchDebounce = setTimeout(() => performSearch(this.value), 150);
            });
            // Allow keyboard navigation
            searchInput.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    searchResults.innerHTML = '';
                    searchInput.blur();
                }
            });
        }

        // Bind Events
        saveApiKeyButton.addEventListener('click', saveApiConfig);
        sendButton.addEventListener('click', () => handleSend());
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });
        
        if (clearChatButton) {
            clearChatButton.addEventListener('click', clearChat);
        }

        apiProviderSelect.addEventListener('change', updateSafetyLink);
        updateSafetyLink(); // Initial call
        
        // Initial Message
        if (chatMessages.children.length === 0) {
           addMessage({ 
               role: 'assistant', 
               content: "👋 你好！我是 Njx'Log 的 AI 助手。基于豆包 Doubao-Seed 模型，我可以回答关于博客文章的问题，或者帮你检索相关技术内容。" 
           });
        }
    }

    function updateSafetyLink() {
        const provider = apiProviderSelect.value;
        const link = document.getElementById('provider-signup-link');
        if (!link) return;

        if (provider === 'doubao') {
            link.href = 'https://www.volcengine.com/product/doubao';
            link.textContent = '火山引擎控制台';
        } else if (provider === 'deepseek') {
            link.href = 'https://platform.deepseek.com/';
            link.textContent = 'DeepSeek Platform';
        } else if (provider === 'openai') {
            link.href = 'https://platform.openai.com/';
            link.textContent = 'OpenAI Platform';
        }
    }

    function saveApiConfig() {
        const provider = apiProviderSelect.value;
        const key = apiKeyInput.value.trim();
        
        if (key) {
            // Use sessionStorage - cleared when browser tab closes (more secure)
            sessionStorage.setItem(STORAGE_PREFIX + 'provider', provider);
            sessionStorage.setItem(STORAGE_PREFIX + 'api-key', key);
            apiStatus.textContent = '✓ Session';
            apiStatus.className = 'status-success';
        } else {
            sessionStorage.removeItem(STORAGE_PREFIX + 'api-key');
            apiStatus.textContent = '✗ No key';
            apiStatus.className = 'status-error';
        }
    }

    function clearChat() {
        if (abortController) abortController.abort();
        chatHistory = [];
        chatMessages.innerHTML = '';
        addMessage({ 
            role: 'assistant', 
            content: "已清空上下文。请问有什么可以帮你的？"
        });
    }

    function searchRelevantPosts(query, limit = 3) {
        if (!searchIndex || searchIndex.length === 0) return [];
        
        let results = [];
        if (fuse) {
            results = fuse.search(query).map(r => r.item);
        } else {
            // Fallback
            const keywords = query.toLowerCase().split(/\s+/);
            results = searchIndex.map(post => {
                const text = `${post.title} ${post.content} ${post.summary}`.toLowerCase();
                const score = keywords.reduce((acc, kw) => acc + (text.includes(kw) ? 1 : 0), 0);
                return { ...post, score };
            }).filter(p => p.score > 0).sort((a, b) => b.score - a.score);
        }
        return results.slice(0, limit);
    }

    async function handleSend() {
        const query = chatInput.value.trim();
        if (!query) return;

        const apiKey = sessionStorage.getItem(STORAGE_PREFIX + 'api-key');
        const provider = sessionStorage.getItem(STORAGE_PREFIX + 'provider') || 'doubao';
        
        if (!apiKey) {
            addMessage({ 
                role: 'assistant', 
                content: '⚠️ 请先配置 API Key。点击上方的 "Configure API Key" 进行设置。' 
            });
            return;
        }

        // Add User Message
        addMessage({ role: 'user', content: query });
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendButton.disabled = true;

        // Prepare Context & Messages
        const relevantPosts = searchRelevantPosts(query);
        let contextContent = "";
        
        if (relevantPosts.length > 0) {
            const contextText = relevantPosts.map(p => 
                `Title: ${p.title}\nURL: ${p.permalink}\nExcerpt: ${p.content.substring(0, 800)}...`
            ).join('\n\n');
            contextContent = `参考以前的博客文章内容回答用户问题：\n${contextText}`;
        }

        // Construct Message History for API
        // System Prompt -> Context (as User hidden or System) -> Chat History -> Current Query
        
        const apiMessages = [
            { role: 'system', content: SYSTEM_PROMPT },
        ];

        // Add history (max last 6 messages to save tokens)
        const recentHistory = chatHistory.slice(-6); 
        // We filter out our own local "sources" or "error" messages if we stored them strangely, 
        // but here chatHistory should only contain pure role/content.
        
        apiMessages.push(...recentHistory);

        // Current message with context augmentation
        const finalUserContent = contextContent 
            ? `${contextContent}\n\n用户问题：${query}` 
            : query;
            
        apiMessages.push({ role: 'user', content: finalUserContent });
        
        // UI: Loading Bubble
        const loadingId = addLoadingMessage();
        
        // Prepare Fetch
        abortController = new AbortController();
        const config = CONFIG[provider];
        
        try {
            const response = await fetch(config.baseUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    model: config.model,
                    messages: apiMessages,
                    stream: true, // Enable Streaming
                    temperature: 0.7,
                }),
                signal: abortController.signal
            });

            if (!response.ok) {
                let errText = `API Error ${response.status}`;
                try {
                    const errJson = await response.json();
                    errText = errJson.error?.message || errText;
                } catch(e) {}
                throw new Error(errText);
            }

            // Streaming handler
            removeMessage(loadingId);
            const responseMessageId = addMessage({ role: 'assistant', content: '' }, false); // Create empty bubble
            const responseElement = document.getElementById(responseMessageId).querySelector('.message-content');
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let fullText = "";
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                // Process all complete lines
                buffer = lines.pop(); // Keep the last partial line in buffer

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith('data: ')) {
                        const dataStr = trimmed.slice(6);
                        if (dataStr === '[DONE]') continue;
                        try {
                            const data = JSON.parse(dataStr);
                            const content = data.choices?.[0]?.delta?.content || "";
                            if (content) {
                                fullText += content;
                                responseElement.innerHTML = parseMarkdown(fullText);
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } catch (e) {
                            console.warn('SSE Parse Error', e);
                        }
                    }
                }
            }

            // Finalize
            chatHistory.push({ role: 'user', content: query });
            chatHistory.push({ role: 'assistant', content: fullText });
            
            // Append Sources
            if (relevantPosts.length > 0) {
               appendSources(responseElement, relevantPosts);
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                // User stopped generation
            } else {
                removeMessage(loadingId);
                addMessage({ role: 'assistant', content: `❌ Error: ${error.message}` });
            }
        } finally {
            abortController = null;
            sendButton.disabled = false;
            chatInput.focus();
        }
    }

    // --- Helpers ---

    function generateId() {
        return 'msg-' + Math.random().toString(36).substr(2, 9);
    }

    function addMessage(msg, isHtml = false) {
        const id = generateId();
        const div = document.createElement('div');
        div.id = id;
        div.className = `message ${msg.role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isHtml) {
            contentDiv.innerHTML = msg.content;
        } else {
            contentDiv.innerHTML = parseMarkdown(msg.content);
        }
        
        div.appendChild(contentDiv);
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function addLoadingMessage() {
        const id = generateId();
        const div = document.createElement('div');
        div.id = id;
        div.className = 'message assistant loading-message';
        div.innerHTML = `<div class="loading"><span></span><span></span><span></span></div>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendSources(element, posts) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        const title = document.createElement('div');
        title.className = 'sources-title';
        title.textContent = '📚 参考文章:';
        sourcesDiv.appendChild(title);
        
        posts.forEach(post => {
            const a = document.createElement('a');
            a.className = 'source-link';
            a.href = post.permalink;
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.textContent = `→ ${post.title}`;
            sourcesDiv.appendChild(a);
        });
        element.appendChild(sourcesDiv);
    }

    function parseMarkdown(text) {
        if (!text) return '';
        
        // Store code blocks temporarily to protect them from line break replacement
        const codeBlocks = [];
        const inlineCodes = [];
        
        // Extract and protect fenced code blocks
        let processed = text.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
            codeBlocks.push({ lang, code });
            return placeholder;
        });
        
        // Extract and protect inline code
        processed = processed.replace(/`([^`]+)`/g, (match, code) => {
            const placeholder = `__INLINE_CODE_${inlineCodes.length}__`;
            inlineCodes.push(code);
            return placeholder;
        });
        
        // Escape HTML in remaining content
        processed = processed
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
        
        // Apply formatting
        processed = processed
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*([^*]+)\*/g, '<em>$1</em>'); // Italic
        
        // Links - only allow safe protocols (http, https, mailto, relative paths)
        processed = processed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
            const trimmedUrl = url.trim();
            // Allow only safe protocols or relative URLs
            if (/^(https?:\/\/|mailto:|\/|#)/.test(trimmedUrl) || !/^[a-z]+:/i.test(trimmedUrl)) {
                const safeUrl = trimmedUrl.replace(/"/g, '&quot;');
                return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${text}</a>`;
            }
            // Reject dangerous protocols (javascript:, data:, etc.)
            return `${text} (${trimmedUrl})`;
        });
        
        // Line breaks (only in non-code content)
        processed = processed
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
        
        // Restore inline code
        inlineCodes.forEach((code, i) => {
            const escapedCode = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            processed = processed.replace(`__INLINE_CODE_${i}__`, `<code>${escapedCode}</code>`);
        });
        
        // Restore code blocks
        codeBlocks.forEach((block, i) => {
            const escapedCode = block.code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            processed = processed.replace(
                `__CODE_BLOCK_${i}__`, 
                `<pre><code class="language-${block.lang}">${escapedCode}</code></pre>`
            );
        });
        
        return processed;
    }

    // ===== Search Box Functions =====
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
        
        searchResults.innerHTML = results.map(post => {
            const title = escapeHtml(post.title);
            const summary = escapeHtml((post.summary || post.content || '').substring(0, 100));
            return `<li><a href="${post.permalink}"><span class="title">${title}</span><span class="summary">${summary}...</span></a></li>`;
        }).join('');
    }
    
    function escapeHtml(text) {
        if (!text) return '';
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    // Initialize on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
