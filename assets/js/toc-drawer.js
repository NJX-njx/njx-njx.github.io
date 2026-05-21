(function () {
    const body = document.body;
    const drawer = document.getElementById('toc-drawer');
    const toggle = document.querySelector('.toc-drawer-toggle');
    const tocLinks = Array.from(document.querySelectorAll('.toc a[href^="#"]'));

    function setActiveTocLink(id) {
        if (!id) return;
        tocLinks.forEach((link) => {
            link.classList.toggle('is-active', decodeURIComponent(link.hash.slice(1)) === id);
        });
    }

    function bindActiveSections() {
        if (!tocLinks.length) return;

        const headings = tocLinks
            .map((link) => {
                const id = decodeURIComponent(link.hash.slice(1));
                return id ? document.getElementById(id) : null;
            })
            .filter(Boolean);

        if (!headings.length) return;

        setActiveTocLink(headings[0].id);

        if (!('IntersectionObserver' in window)) {
            window.addEventListener('scroll', () => {
                const current = headings
                    .filter((heading) => heading.getBoundingClientRect().top <= 140)
                    .pop();
                setActiveTocLink((current || headings[0]).id);
            }, { passive: true });
            return;
        }

        const visibleHeadings = new Map();
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    visibleHeadings.set(entry.target.id, entry.boundingClientRect.top);
                } else {
                    visibleHeadings.delete(entry.target.id);
                }
            });

            if (visibleHeadings.size) {
                const activeId = Array.from(visibleHeadings.entries())
                    .sort((a, b) => a[1] - b[1])[0][0];
                setActiveTocLink(activeId);
                return;
            }

            const current = headings
                .filter((heading) => heading.getBoundingClientRect().top <= 140)
                .pop();
            setActiveTocLink((current || headings[0]).id);
        }, {
            rootMargin: '-120px 0px -65% 0px',
            threshold: 0.01
        });

        headings.forEach((heading) => observer.observe(heading));
    }

    bindActiveSections();

    if (!drawer || !toggle) return;

    function setOpen(isOpen) {
        body.classList.toggle('toc-drawer-open', isOpen);
        toggle.setAttribute('aria-expanded', String(isOpen));
        drawer.setAttribute('aria-hidden', String(!isOpen));
    }

    toggle.addEventListener('click', () => {
        setOpen(!body.classList.contains('toc-drawer-open'));
    });

    document.querySelectorAll('[data-toc-close], .toc-drawer a').forEach((element) => {
        element.addEventListener('click', () => setOpen(false));
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') setOpen(false);
    });
})();
