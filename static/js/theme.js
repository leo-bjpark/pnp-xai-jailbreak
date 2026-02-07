/**
 * Theme switcher: Sun = White, Moon = Black
 * Preference saved in localStorage.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "pnp-xai-theme";

  function getTheme() {
    return localStorage.getItem(STORAGE_KEY) || "light";
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(STORAGE_KEY, theme);
  }

  function toggleTheme() {
    const current = getTheme();
    const next = current === "light" ? "dark" : "light";
    setTheme(next);
  }

  function init() {
    setTheme(getTheme());
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.addEventListener("click", toggleTheme);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
