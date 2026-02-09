/**
 * Session and Result memory panels: list items with individual delete.
 */

(function () {
  "use strict";

  const sessionBtn = document.getElementById("btn-clear-session");
  const sessionDropdown = document.getElementById("session-dropdown");
  const sessionTbody = document.getElementById("session-tbody");
  const sessionEmptyRow = document.getElementById("session-empty-row");

  const resultBtn = document.getElementById("btn-clear-result");
  const resultDropdown = document.getElementById("result-dropdown");
  const resultTbody = document.getElementById("result-tbody");
  const resultEmptyRow = document.getElementById("result-empty-row");

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = String(s);
    return div.innerHTML;
  }

  function toggleDropdown(dd, btn, fetchFn) {
    if (!dd) return;
    const isOpen = dd.classList.contains("visible");
    if (isOpen) {
      dd.classList.remove("visible");
      dd.setAttribute("aria-hidden", "true");
    } else {
      if (fetchFn) fetchFn();
      dd.classList.add("visible");
      dd.setAttribute("aria-hidden", "false");
    }
  }

  async function refreshMemorySummary() {
    const sessionEl = document.getElementById("memory-summary-session");
    const resultEl = document.getElementById("memory-summary-result");
    const variableEl = document.getElementById("memory-summary-variable");
    if (!sessionEl || !resultEl || !variableEl) return;
    try {
      const res = await fetch("/api/memory/summary");
      const data = await res.json().catch(() => ({}));
      const fmt = (v) => (v != null && typeof v === "number" ? v.toFixed(2) + " GB" : "—");
      sessionEl.textContent = fmt(data.session_gb);
      resultEl.textContent = fmt(data.result_gb);
      variableEl.textContent = fmt(data.variable_gb);
    } catch (_) {
      sessionEl.textContent = "—";
      resultEl.textContent = "—";
      variableEl.textContent = "—";
    }
  }

  async function refreshSessionList() {
    if (!sessionTbody || !sessionEmptyRow) return;
    try {
      const res = await fetch("/api/memory/session/list");
      const data = await res.json().catch(() => ({}));
      const caches = Array.isArray(data.caches) ? data.caches : [];

      sessionTbody.querySelectorAll(".session-data-row").forEach((el) => el.remove());
      sessionEmptyRow.style.display = caches.length > 0 ? "none" : "table-row";

      caches.forEach((c) => {
        const keyLabel = [c.task, c.model, c.treatment, c.name].filter(Boolean).join(" | ") || c.key || "—";
        const tr = document.createElement("tr");
        tr.className = "session-data-row";
        tr.innerHTML =
          '<td class="data-vars-td-name" title="' + escapeHtml(keyLabel) + '">' + escapeHtml(keyLabel) + '</td>' +
          '<td class="data-vars-td-action"><button type="button" class="data-vars-btn-delete" title="삭제">×</button></td>';
        const key = c.key;
        tr.querySelector(".data-vars-btn-delete").addEventListener("click", async () => {
          if (!confirm("이 Session cache를 제거하시겠습니까?")) return;
          try {
            const r = await fetch("/api/memory/session/unregister/" + encodeURIComponent(key), { method: "DELETE" });
            if (r.ok) {
              refreshSessionList();
              if (typeof updateSessionUI === "function") updateSessionUI();
            }
          } catch (e) { console.error(e); }
        });
        sessionTbody.appendChild(tr);
      });
    } catch (err) {
      console.error(err);
      if (sessionEmptyRow) sessionEmptyRow.style.display = "table-row";
    }
    refreshMemorySummary();
  }

  async function refreshResultList() {
    if (!resultTbody || !resultEmptyRow) return;
    try {
      const res = await fetch("/api/tasks");
      const data = await res.json().catch(() => ({}));
      const tasksRaw = data.tasks || data;
      const tasks = [];
      if (typeof tasksRaw === "object" && !Array.isArray(tasksRaw)) {
        Object.values(tasksRaw).forEach((arr) => {
          if (Array.isArray(arr)) tasks.push(...arr);
        });
      } else if (Array.isArray(tasksRaw)) tasks.push(...tasksRaw);

      resultTbody.querySelectorAll(".result-data-row").forEach((el) => el.remove());
      resultEmptyRow.style.display = tasks.length > 0 ? "none" : "table-row";

      tasks.forEach((t) => {
        const tr = document.createElement("tr");
        tr.className = "result-data-row";
        const title = (t.title || t.id || "").trim() || "—";
        const meta = [t.model || "", t.xai_level || ""].filter(Boolean).join(" · ") || "—";
        tr.innerHTML =
          '<td class="data-vars-td-name" title="' + escapeHtml(title) + '">' + escapeHtml(title) + '</td>' +
          '<td class="data-vars-td-memory">' + escapeHtml(meta) + '</td>' +
          '<td class="data-vars-td-action"><button type="button" class="data-vars-btn-delete" title="삭제">×</button></td>';
        const taskId = t.id;
        tr.querySelector(".data-vars-btn-delete").addEventListener("click", async () => {
          if (!confirm("이 task를 삭제하시겠습니까?")) return;
          try {
            const sessRes = await fetch("/api/memory/session/list");
            const sessData = await sessRes.json().catch(() => ({}));
            const sessCaches = Array.isArray(sessData.caches) ? sessData.caches : [];
            for (const sc of sessCaches) {
              if (sc.task === taskId && sc.key) {
                await fetch("/api/memory/session/unregister/" + encodeURIComponent(sc.key), { method: "DELETE" });
              }
            }
            const r = await fetch("/api/tasks/" + encodeURIComponent(taskId), { method: "DELETE" });
            if (r.ok) {
              refreshSessionList();
              refreshResultList();
              if (typeof renderTaskList === "function") {
                const d = await fetch("/api/tasks").then((r) => r.json());
                renderTaskList(d.tasks || d, d.xai_level_names || {});
              }
            }
          } catch (e) { console.error(e); }
        });
        resultTbody.appendChild(tr);
      });
    } catch (err) {
      console.error(err);
      if (resultEmptyRow) resultEmptyRow.style.display = "table-row";
    }
    refreshMemorySummary();
  }

  if (sessionBtn && sessionDropdown) {
    sessionBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      toggleDropdown(sessionDropdown, sessionBtn, refreshSessionList);
    });
  }
  if (resultBtn && resultDropdown) {
    resultBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      toggleDropdown(resultDropdown, resultBtn, refreshResultList);
    });
  }

  const dataVarsDropdown = document.getElementById("data-vars-dropdown");
  document.addEventListener("click", () => {
    [sessionDropdown, resultDropdown, dataVarsDropdown].forEach((dd) => {
      if (dd && dd.classList.contains("visible")) {
        dd.classList.remove("visible");
        dd.setAttribute("aria-hidden", "true");
      }
    });
  });
  if (sessionDropdown) sessionDropdown.addEventListener("click", (e) => e.stopPropagation());
  if (resultDropdown) resultDropdown.addEventListener("click", (e) => e.stopPropagation());
  if (dataVarsDropdown) dataVarsDropdown.addEventListener("click", (e) => e.stopPropagation());

  window.refreshSessionList = refreshSessionList;
  window.refreshResultList = refreshResultList;
  window.refreshMemorySummary = refreshMemorySummary;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => refreshMemorySummary());
  } else {
    refreshMemorySummary();
  }
})();
