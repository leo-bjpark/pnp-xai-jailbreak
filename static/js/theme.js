/**
 * Theme switcher: Sun = White, Moon = Black
 * Preference saved in localStorage.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "pnp-xai-theme";
  const CUSTOM_THEME_ENDPOINT = "/api/theme/custom";
  let customOverrides = { light: {}, dark: {} };
  const THEME_VARS = [
    "--bg",
    "--panel",
    "--sidebar",
    "--text",
    "--muted",
    "--border",
    "--brand",
    "--accent",
    "--hover",
  ];

  function getTheme() {
    return localStorage.getItem(STORAGE_KEY) || "light";
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(STORAGE_KEY, theme);
    applyCustomOverrides(theme);
  }

  function toggleTheme() {
    const current = getTheme();
    const next = current === "light" ? "dark" : "light";
    setTheme(next);
  }

  function applyCustomOverrides(theme) {
    const overrides = (customOverrides && customOverrides[theme]) || {};
    THEME_VARS.forEach((k) => document.documentElement.style.removeProperty(k));
    Object.keys(overrides).forEach((k) => {
      if (!k.startsWith("--")) return;
      document.documentElement.style.setProperty(k, overrides[k]);
    });
  }

  async function loadCustomOverrides() {
    try {
      const res = await fetch(CUSTOM_THEME_ENDPOINT);
      const data = await res.json().catch(() => ({}));
      customOverrides = data.overrides || { light: {}, dark: {} };
    } catch (_) {}
  }

  function syncPickerInputs(theme) {
    const panel = document.getElementById("theme-custom-panel");
    if (!panel) return;
    panel.querySelectorAll("input[data-var]").forEach((input) => {
      const key = input.dataset.var;
      const override = customOverrides?.[theme]?.[key];
      if (override) {
        input.value = override;
        return;
      }
      const computed = getComputedStyle(document.documentElement).getPropertyValue(key).trim();
      if (computed) input.value = computed;
    });
  }

  async function saveOverride(theme, key, value) {
    try {
      const payload = { theme, key, value };
      const res = await fetch(CUSTOM_THEME_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.overrides) customOverrides = data.overrides;
    } catch (_) {}
  }
  async function resetOverrides(theme) {
    try {
      const payload = { theme, overrides: { light: customOverrides.light || {}, dark: customOverrides.dark || {} } };
      payload.overrides[theme] = {};
      const res = await fetch(CUSTOM_THEME_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.overrides) customOverrides = data.overrides;
    } catch (_) {}
  }

  function collectCurrentThemeValues(theme) {
    const values = {};
    THEME_VARS.forEach((key) => {
      const val = getComputedStyle(document.documentElement).getPropertyValue(key).trim();
      if (val) values[key] = val;
    });
    return { theme, values };
  }
  async function resetOverrides(theme) {
    try {
      const payload = { theme, overrides: { light: customOverrides.light || {}, dark: customOverrides.dark || {} } };
      payload.overrides[theme] = {};
      const res = await fetch(CUSTOM_THEME_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.overrides) customOverrides = data.overrides;
    } catch (_) {}
  }

  function togglePanel(show) {
    const panel = document.getElementById("theme-custom-panel");
    if (!panel) return;
    panel.style.display = show ? "block" : "none";
    panel.setAttribute("aria-hidden", show ? "false" : "true");
  }

  async function init() {
    await loadCustomOverrides();
    setTheme(getTheme());
    syncPickerInputs(getTheme());
    const btn = document.getElementById("theme-toggle");
    const panel = document.getElementById("theme-custom-panel");
    const wrap = document.getElementById("theme-custom-wrap");
    if (btn) btn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleTheme();
      syncPickerInputs(getTheme());
    });
    if (wrap && panel) {
      wrap.addEventListener("mouseenter", () => {
        togglePanel(true);
      });
      document.addEventListener("click", (e) => {
        if (e.target.closest("#theme-custom-panel") || e.target.closest("#theme-custom-wrap")) return;
        togglePanel(false);
      });
      panel.addEventListener("click", (e) => e.stopPropagation());
    }
    if (panel) {
      panel.querySelectorAll("input[data-var]").forEach((input) => {
        input.addEventListener("input", () => {
          const theme = getTheme();
          const key = input.dataset.var;
          const value = input.value;
          customOverrides[theme] = customOverrides[theme] || {};
          customOverrides[theme][key] = value;
          document.documentElement.style.setProperty(key, value);
          saveOverride(theme, key, value);
        });
      });
    }
    const resetBtn = document.getElementById("theme-custom-reset");
    if (resetBtn) {
      resetBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const theme = getTheme();
        await resetOverrides(theme);
        applyCustomOverrides(theme);
        syncPickerInputs(theme);
      });
    }
    const updateBtn = document.getElementById("theme-custom-update");
    if (updateBtn) {
      updateBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const payload = collectCurrentThemeValues(getTheme());
        try {
          const res = await fetch(CUSTOM_THEME_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              overrides: {
                light: payload.theme === "light" ? payload.values : customOverrides.light || {},
                dark: payload.theme === "dark" ? payload.values : customOverrides.dark || {},
              },
            }),
          });
          const data = await res.json().catch(() => ({}));
          if (res.ok && data.overrides) customOverrides = data.overrides;
          applyCustomOverrides(getTheme());
          syncPickerInputs(getTheme());
        } catch (_) {}
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
