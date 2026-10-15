(() => {
  const root = document.documentElement;
  let saved;
  try {
    saved = localStorage.getItem("pref-theme");
  } catch (_) {
    // 禁用站点存储时仍允许使用系统主题和页面内切换。
  }
  if (saved === "dark" || (
    saved !== "light" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  )) {
    root.dataset.theme = "dark";
  } else {
    root.dataset.theme = "light";
  }

  const button = document.querySelector("[data-report-theme-toggle]");
  button?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("pref-theme", next);
    } catch (_) {
      // 存储不可用不应阻断本次主题切换。
    }
  });

  // 章节跳转同时避让站点栏和报告自己的章节栏，适配窄屏尺寸变化。
  const subnav = document.querySelector("#chap-nav, #era-rail");
  if (subnav) {
    const updateOffset = () => root.style.setProperty(
      "--rpt-subnav-h", `${subnav.getBoundingClientRect().height}px`
    );
    updateOffset();
    window.addEventListener("resize", updateOffset);
    if (window.ResizeObserver) new ResizeObserver(updateOffset).observe(subnav);
  }
})();
