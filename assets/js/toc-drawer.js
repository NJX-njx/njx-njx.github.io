(function () {
    const body = document.body;
    const toc = document.getElementById('post-toc');
    const toggle = document.querySelector('.toc-toggle');
    const closeButton = toc.querySelector('.toc-close');
    const links = Array.from(toc.querySelectorAll('a[href^="#"]'));
    const drawerMode = matchMedia('(max-width: 1040px)');

    function setActive(id) {
        links.forEach((link) => {
            link.classList.toggle('is-active', decodeURIComponent(link.hash.slice(1)) === id);
        });
    }

    const headings = links
        .map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))))
        .filter(Boolean);

    // The active section is the last heading that has scrolled past the top band.
    function updateActive() {
        const current = headings.filter((heading) => heading.getBoundingClientRect().top <= 140).pop() || headings[0];
        setActive(current.id);
    }

    let pending = false;
    window.addEventListener('scroll', () => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => {
            pending = false;
            updateActive();
        });
    }, { passive: true });
    updateActive();

    // In drawer mode a closed TOC is off screen; inert keeps its links out of the tab order.
    function setOpen(open, restoreFocus) {
        const wasOpen = body.classList.contains('toc-open');
        body.classList.toggle('toc-open', open);
        toggle.setAttribute('aria-expanded', String(open));
        toc.inert = drawerMode.matches && !open;
        if (open) closeButton.focus();
        else if (wasOpen && restoreFocus) toggle.focus();
    }

    toggle.addEventListener('click', () => setOpen(!body.classList.contains('toc-open')));
    document.querySelectorAll('[data-toc-close]').forEach((element) => {
        element.addEventListener('click', () => setOpen(false, true));
    });
    links.forEach((link) => link.addEventListener('click', () => {
        if (drawerMode.matches) setOpen(false);
    }));
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && body.classList.contains('toc-open')) setOpen(false, true);
    });
    drawerMode.addEventListener('change', () => setOpen(false));
    setOpen(false);
})();
