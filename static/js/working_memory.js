/**
 * Working Memory panel
 * --------------------
 * All UI logic for RAM / GPU working-memory panel lives here.
 *
 * Responsibilities:
 * - Fetch `/api/data-vars` summary from backend working_memory.py
 * - Render:
 *    - loaded model memory (GPU / RAM)
 *    - saved variables (estimated RAM usage, created_at, etc.)
 * - Handle delete / reset actions
 */

(function () {
  "use strict";

  const btnDataVars = document.getElementById("btn-data-vars");
  const dataVarsDropdown = document.getElementById("data-vars-dropdown");
  const dataVarsTbody = document.getElementById("data-vars-tbody");
  const dataVarsEmptyRow = document.getElementById("data-vars-empty-row");

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = String(s);
    return div.innerHTML;
  }

  function formatModelMemory(gpuGb, ramGb) {
    const g = gpuGb != null ? "GPU: " + gpuGb + " GB" : "";
    const r = ramGb != null ? "RAM: " + ramGb + " GB" : "";
    return [g, r].filter(Boolean).join(" · ") || "—";
  }

  function formatVarMemoryMb(mb) {
    if (mb == null) return "—";
    return "RAM: ~" + mb + " MB";
  }

  async function refreshWorkingMemoryList() {
    if (!dataVarsTbody || !dataVarsEmptyRow) return;
    try {
      const res = await fetch("/api/data-vars");
      const data = await res.json().catch(() => ({}));
      const loadedModel = data.loaded_model || null;
      const variables = Array.isArray(data.variables) ? data.variables : [];
      const hasAny = !!loadedModel || variables.length > 0;

      dataVarsEmptyRow.style.display = hasAny ? "none" : "table-row";
      dataVarsTbody
        .querySelectorAll(".data-vars-data-row")
        .forEach((el) => el.remove());

      // Loaded model row (GPU / RAM)
      if (loadedModel) {
        const tr = document.createElement("tr");
        tr.className = "data-vars-data-row data-vars-model-row";
        const memStr = formatModelMemory(
          loadedModel.memory_gpu_gb,
          loadedModel.memory_ram_gb
        );
        tr.innerHTML =
          '<td class="data-vars-td-name" title="Loaded model">' +
          escapeHtml(loadedModel.name || "") +
          '</td>' +
          '<td class="data-vars-td-memory">' +
          escapeHtml(memStr) +
          "</td>" +
          '<td class="data-vars-td-action"><button type="button" class="data-vars-btn-delete" data-kind="model" title="Unload model">×</button></td>';
        const btn = tr.querySelector(".data-vars-btn-delete");
        btn.addEventListener("click", async () => {
          try {
            const r = await fetch("/api/empty_cache", { method: "POST" });
            if (r.ok) {
              refreshWorkingMemoryList();
              if (typeof window.refreshMemorySummary === "function") window.refreshMemorySummary();
              if (typeof window.refreshSidebarVariableList === "function") window.refreshSidebarVariableList();
            }
          } catch (e) {
            console.error("Failed to empty cache:", e);
          }
        });
        dataVarsTbody.appendChild(tr);
      }

      // Saved working‑memory variables (RAM only, approx.)
      variables.forEach((v) => {
        const tr = document.createElement("tr");
        tr.className = "data-vars-data-row data-vars-var-row";
        const memStr = formatVarMemoryMb(v.memory_ram_mb);
        const varId = v.id || "";
        const varName = v.name || varId || "";
        tr.innerHTML =
          '<td class="data-vars-td-name" title="' +
          escapeHtml(varName) +
          '">' +
          escapeHtml(varName) +
          "</td>" +
          '<td class="data-vars-td-memory">' +
          escapeHtml(memStr) +
          "</td>" +
          '<td class="data-vars-td-action"><button type="button" class="data-vars-btn-delete" data-kind="var" title="Remove variable">×</button></td>';
        const btn = tr.querySelector(".data-vars-btn-delete");
        btn.addEventListener("click", async () => {
          try {
            const r = await fetch(
              "/api/data-vars/" + encodeURIComponent(varId || varName),
              { method: "DELETE" }
            );
            if (r.ok) {
              refreshWorkingMemoryList();
              if (typeof window.refreshMemorySummary === "function") window.refreshMemorySummary();
              if (typeof window.refreshSidebarVariableList === "function") window.refreshSidebarVariableList();
            }
          } catch (e) {
            console.error("Failed to delete variable:", e);
          }
        });
        dataVarsTbody.appendChild(tr);
      });
    } catch (err) {
      console.error("Failed to refresh working memory list:", err);
      dataVarsEmptyRow.style.display = "table-row";
      const cell = document.getElementById("data-vars-empty");
      if (cell) cell.textContent = "Failed to load.";
    }
  }

  // Expose for debugging / manual refresh
  window.refreshWorkingMemoryList = refreshWorkingMemoryList;

  if (btnDataVars && dataVarsDropdown) {
    btnDataVars.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isOpen = dataVarsDropdown.classList.contains("visible");
      if (isOpen) {
        dataVarsDropdown.classList.remove("visible");
        dataVarsDropdown.setAttribute("aria-hidden", "true");
      } else {
        refreshWorkingMemoryList().then(() => {
          if (typeof window.refreshMemorySummary === "function") window.refreshMemorySummary();
          if (typeof window.refreshSidebarVariableList === "function") window.refreshSidebarVariableList();
        });
        dataVarsDropdown.classList.add("visible");
        dataVarsDropdown.setAttribute("aria-hidden", "false");
      }
    });

    document.addEventListener("click", () => {
      if (dataVarsDropdown.classList.contains("visible")) {
        dataVarsDropdown.classList.remove("visible");
        dataVarsDropdown.setAttribute("aria-hidden", "true");
      }
    });

    dataVarsDropdown.addEventListener("click", (e) => e.stopPropagation());
  }
})();
