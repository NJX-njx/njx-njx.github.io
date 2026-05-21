(function () {
    const body = document.body;
    const drawer = document.getElementById('toc-drawer');
    const toggle = document.querySelector('.toc-drawer-toggle');

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
