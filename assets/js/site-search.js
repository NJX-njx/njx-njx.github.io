import Fuse from './fuse.basic.min.js';
import { createSearch, queryTokens } from './search-core.js';

const searchBox = document.getElementById('searchbox');
const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');
const noResultsText = searchBox.dataset.noResults;
let search = null;

async function init() {
    try {
        const response = await fetch(searchBox.dataset.index);
        search = createSearch(await response.json(), Fuse);
    } catch (error) {
        console.error('Failed to load search index:', error);
        return;
    }

    let debounce = null;
    searchInput.addEventListener('input', () => {
        clearTimeout(debounce);
        debounce = setTimeout(() => render(searchInput.value), 150);
    });
    searchInput.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            searchResults.innerHTML = '';
            searchInput.blur();
        }
    });
    if (searchInput.value) render(searchInput.value);
}

function render(rawQuery) {
    const query = rawQuery.trim();
    if (!query) {
        searchResults.innerHTML = '';
        return;
    }
    const results = search(query, 12);
    searchResults.innerHTML = results.length
        ? results.map((chunk) => renderResult(chunk, query)).join('')
        : `<li class="no-results">${escapeHtml(noResultsText)}</li>`;
}

function renderResult(chunk, query) {
    const section = chunk.section !== chunk.title
        ? `<span class="section">${escapeHtml(chunk.section)}</span>`
        : '';
    const meta = [chunk.date, chunk.tags.slice(0, 4).join(' / ')].filter(Boolean).map(escapeHtml).join(' · ');
    const url = chunk.anchor ? `${chunk.permalink}#${chunk.anchor}` : chunk.permalink;
    return `
        <li>
            <a href="${escapeHtml(url)}">
                <span class="title">${escapeHtml(chunk.title)}</span>
                ${section}
                <span class="summary">${snippet(chunk, query)}</span>
                ${meta ? `<span class="meta">${meta}</span>` : ''}
            </a>
        </li>
    `;
}

function snippet(chunk, query) {
    const match = ['content', 'summary', 'section'].map((key) => chunk._matches.find((m) => m.key === key && m.indices.length)).find(Boolean);
    if (match) {
        const longest = match.indices.reduce((best, range) => (range[1] - range[0] > best[1] - best[0] ? range : best));
        return highlight(String(chunk[match.key]), longest[0], longest[1]);
    }

    const text = chunk.content || chunk.summary;
    const lower = text.toLowerCase();
    for (const needle of [query.toLowerCase(), ...queryTokens(query)]) {
        const at = lower.indexOf(needle);
        if (at >= 0) return highlight(text, at, at + needle.length - 1);
    }
    const compact = text.replace(/\s+/g, ' ').trim();
    return escapeHtml(compact.length > 180 ? `${compact.slice(0, 180)}...` : compact);
}

function highlight(text, start, end) {
    const from = Math.max(0, start - 70);
    const to = Math.min(text.length, end + 90);
    return [
        escapeHtml((from > 0 ? '...' : '') + text.slice(from, start)),
        `<mark>${escapeHtml(text.slice(start, end + 1))}</mark>`,
        escapeHtml(text.slice(end + 1, to) + (to < text.length ? '...' : ''))
    ].join('');
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

init();
