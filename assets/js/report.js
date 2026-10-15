// Loaded synchronously in each report's <head>, after /reports/report.css and
// before the report's own scripts, so the theme is set and RC/RD exist by the
// time the charts build their palettes. Published as /reports/report.js.
(function () {
    const root = document.documentElement;
    const saved = localStorage.getItem('pref-theme');
    const dark = saved ? saved === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'dark' : 'light';

    const probe = document.createElement('canvas').getContext('2d');

    function toHex(color) {
        probe.fillStyle = '#000000';
        probe.fillStyle = color;
        return probe.fillStyle;
    }

    // Raw value of a report.css custom property, e.g. the --serif font stack.
    window.RS = function (name) {
        return getComputedStyle(root).getPropertyValue(name).trim();
    };

    // Neutral report tokens (--paper, --ink, --line, ...) resolved from CSS, so
    // charts draw with the blog's colors for the current theme.
    window.RC = function (name) {
        return toHex(window.RS(name));
    };

    // Fixed palette colors keep their hue; on dark paper they are mixed toward
    // white so lines and labels keep their contrast.
    window.RD = function (hex) {
        if (!dark) return hex;
        const value = parseInt(toHex(hex).slice(1), 16);
        const channels = [value >> 16, (value >> 8) & 255, value & 255]
            .map((channel) => Math.round(channel + (255 - channel) * 0.4));
        return `#${channels.map((channel) => channel.toString(16).padStart(2, '0')).join('')}`;
    };

    // Translucent variant of either kind: RA("--ink", 0.3) or RA("#2251ff", 0.1).
    window.RA = function (color, alpha) {
        const value = parseInt((color.startsWith('--') ? window.RC(color) : window.RD(color)).slice(1), 16);
        return `rgba(${value >> 16},${(value >> 8) & 255},${value & 255},${alpha})`;
    };

    function fillBar() {
        const bar = document.getElementById('njx-bar');
        const slug = location.pathname.split('/').filter(Boolean).pop();
        bar.innerHTML = `
            <nav class="header-nav">
                <div class="logo">
                    <a href="/" title="Njx'Log"><img src="/blog-icon.png" alt="" height="35">Njx'Log</a>
                    <div class="logo-switches">
                        <button class="theme-toggle" type="button" aria-label="切换主题" title="切换主题">
                            <svg class="moon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
                            <svg class="sun" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
                        </button>
                        <span class="nav-sep">|</span>
                        <ul class="lang-menu"><li><a href="/en/posts/${slug}/" hreflang="en" title="Read the article in English">English</a></li></ul>
                    </div>
                </div>
                <ul class="menu">
                    <li><a href="/posts/${slug}/">阅读原文</a></li>
                    <li><a href="/reports/">全部报告</a></li>
                </ul>
            </nav>`;
        // Charts are drawn with the colors resolved at load time, so a theme
        // change needs a fresh render.
        bar.querySelector('button').addEventListener('click', () => {
            localStorage.setItem('pref-theme', dark ? 'light' : 'dark');
            location.reload();
        });
    }

    document.addEventListener('DOMContentLoaded', fillBar);
})();
