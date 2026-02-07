/**
 * PnP-XAI-LLM - Main app logic
 * - Session management (Loaded Model + Treatment)
 * - Create XAI = new session
 * - Run with mismatch -> confirm modal
 * - Task persistence
 */

(function () {
  "use strict";

  const API = {
    tasks: () => fetch("/api/tasks").then((r) => r.json()).then((d) => d.tasks || d),
    tasksWithMeta: () => fetch("/api/tasks").then((r) => r.json()),
    task: (id) => fetch(`/api/tasks/${id}`).then((r) => r.json()),
    createTask: (data) =>
      fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    updateTask: (taskId, data) =>
      fetch(`/api/tasks/${taskId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    session: () => fetch("/api/session").then((r) => r.json()),
    setSession: (data) =>
      fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    loadModel: (data) =>
      fetch("/api/load_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    run: (data) =>
      fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json().then((body) => ({ ok: r.ok, ...body }))),
  };

  // State
  let session = { loaded_model: null, treatment: null };
  let loadInProgress = false;
  let pendingRunAfterConfirm = null;
  let currentTaskLevel = "0.1"; // Selected XAI level when creating task
  let runAbortController = null; // abort the current /api/run request when user clicks Stop
  let taskLinkNavigateTimeout = null; // delay single-click navigate so double-click can cancel it for rename

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const el = {
    btnCreateTask: $("#btn-create-task"),
    createTaskWrap: document.querySelector(".create-task-wrap"),
    createTaskDropdown: $("#create-task-dropdown"),
    loadedModelDisplay: $("#loaded-model-display"),
    loadedModelText: $("#loaded-model-text"),
    taskPanelList: $("#task-panel-list"),
    inputSettingPanel: $("#input-setting-panel"),
    inputSettingTrigger: $("#input-setting-trigger"),
    sidebarModel: $("#sidebar-model"),
    sidebarTreatment: $("#sidebar-treatment"),
    btnLoadModel: $("#btn-load-model"),
    btnRun: $("#btn-run"),
    resultsPlaceholder: $("#results-placeholder"),
    resultsContent: $("#results-content"),
    modalConfirm: $("#modal-confirm-load"),
    modalMessage: $("#modal-message"),
    modalCancel: $("#modal-cancel"),
    modalConfirmBtn: $("#modal-confirm"),
  };

  // ----- Sync UI with session -----
  function updateSessionUI() {
    const wantModel = session.loaded_model || "";
    const wantHasModel = !!session.loaded_model;
    const curText = (el.loadedModelText && el.loadedModelText.textContent) || "";
    const curHasModel = el.loadedModelDisplay && el.loadedModelDisplay.classList.contains("has-model");
    const displayValue = wantModel || "—";
    if (curText !== displayValue || curHasModel !== wantHasModel) {
      if (el.loadedModelDisplay) {
        if (wantHasModel) el.loadedModelDisplay.classList.add("has-model");
        else el.loadedModelDisplay.classList.remove("has-model");
      }
      if (el.loadedModelText) el.loadedModelText.textContent = displayValue;
    }
    updateLoadButtonState();
    updateInputSettingTriggerText();
  }

  // Input Setting trigger: on task page show Task's run model/treatment (fixed); before RUN show None
  function updateInputSettingTriggerText() {
    const trigger = el.inputSettingTrigger;
    if (!trigger) return;
    const taskTitle = trigger.dataset.taskTitle || (document.querySelector(".input-setting-header")?.textContent?.split("—")[1]?.trim()) || "—";
    const onTaskPage = !!window.PNP_CURRENT_TASK_ID;
    const hasTaskModel = onTaskPage && trigger.dataset.taskModel != null && trigger.dataset.taskModel !== "";
    const model = hasTaskModel
      ? trigger.dataset.taskModel
      : (onTaskPage ? "None" : ((el.sidebarModel?.value || "").trim() || "—"));
    const treatment = (onTaskPage && trigger.dataset.taskTreatment != null)
      ? (trigger.dataset.taskTreatment || "None")
      : (onTaskPage ? "None" : ((el.sidebarTreatment?.value || "").trim() || "None"));
    const name = taskTitle;
    trigger.textContent = `${taskTitle} · ${model} · ${treatment} · ${name}`;
  }

  // Load button: disabled while loading, or when selected model === loaded model
  function updateLoadButtonState() {
    if (!el.btnLoadModel) return;
    const selected = el.sidebarModel ? el.sidebarModel.value : "";
    if (loadInProgress) {
      el.btnLoadModel.disabled = true;
      el.btnLoadModel.textContent = "Loading…";
    } else if (selected && selected === session.loaded_model) {
      el.btnLoadModel.disabled = true;
      el.btnLoadModel.textContent = "Load";
    } else {
      el.btnLoadModel.disabled = false;
      el.btnLoadModel.textContent = "Load";
    }
  }

  // ----- Load Model (left sidebar) -----
  el.sidebarModel?.addEventListener("change", () => {
    updateLoadButtonState();
    updateInputSettingTriggerText();
  });
  el.sidebarTreatment?.addEventListener("input", () => updateInputSettingTriggerText());
  el.sidebarTreatment?.addEventListener("change", () => updateInputSettingTriggerText());

  el.btnLoadModel.addEventListener("click", async () => {
    const model = el.sidebarModel.value;
    const treatment = el.sidebarTreatment.value.trim() || "";
    loadInProgress = true;
    updateLoadButtonState();
    try {
      const res = await API.loadModel({ model, treatment });
      if (res.error) throw new Error(res.error);
      session = { loaded_model: model, treatment };
      updateSessionUI();
    } catch (err) {
      alert("Model load failed: " + err.message);
      updateLoadButtonState();
    } finally {
      loadInProgress = false;
      updateLoadButtonState();
    }
  });

  // ----- Sidebar section toggle (Loaded Model, Treatments, Created Task Panels) -----
  document.querySelector(".sidebar")?.addEventListener("click", (e) => {
    const title = e.target.closest(".sidebar-section-title");
    if (!title) return;
    // Don't toggle when clicking inner task-panel-group-title
    if (e.target.closest(".task-panel-group-title")) return;
    const section = title.closest(".sidebar-section");
    if (!section || !section.querySelector(".sidebar-section-body")) return;
    e.preventDefault();
    e.stopPropagation();
    section.classList.toggle("collapsed");
  });

  // ----- Input Setting panel toggle (task-specific, right side) -----
  el.inputSettingTrigger.addEventListener("click", () => {
    el.inputSettingPanel.classList.toggle("visible");
    el.inputSettingTrigger.classList.toggle("has-setting", el.inputSettingPanel.classList.contains("visible"));
  });

  // ----- Right panel: open in separate window (independent of main page) -----
  let panelWindow = null;
  const btnRightPanel = document.getElementById("btn-right-panel");
  const PANEL_NAME = "pnpRightPanel";
  const PANEL_FEATURES = "width=360,height=800,scrollbars=yes,resizable=yes";

  function updatePanelButtonActive() {
    if (!btnRightPanel) return;
    btnRightPanel.classList.toggle("active", panelWindow && !panelWindow.closed);
  }

  function toggleRightPanel(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (!btnRightPanel) return;
    if (panelWindow && !panelWindow.closed) {
      panelWindow.close();
      panelWindow = null;
    } else {
      panelWindow = window.open("/panel", PANEL_NAME, PANEL_FEATURES);
    }
    updatePanelButtonActive();
  }

  if (btnRightPanel) {
    btnRightPanel.addEventListener("click", toggleRightPanel, true);
  }
  setInterval(function () {
    if (panelWindow && panelWindow.closed) {
      panelWindow = null;
      updatePanelButtonActive();
    }
  }, 500);

  // ----- Create XAI: show full-screen name input, then add task -----
  const modalCreateName = document.getElementById("modal-create-task-name");
  const createNameInput = document.getElementById("create-task-name-input");
  const createTaskCancel = document.getElementById("create-task-cancel");
  const createTaskConfirm = document.getElementById("create-task-confirm");

  let pendingCreate = null; // { level, name }

  function showCreateTaskModal(level, defaultName) {
    if (!level) return;
    pendingCreate = { level, defaultName };
    if (createNameInput) createNameInput.value = defaultName || "";
    if (createNameInput) createNameInput.placeholder = "Enter task name";
    modalCreateName?.classList.add("visible");
    createNameInput?.focus();
  }

  function hideCreateTaskModal() {
    modalCreateName.classList.remove("visible");
    pendingCreate = null;
  }

  async function submitCreateTask() {
    if (!pendingCreate) return;
    const title = createNameInput.value.trim() || "Task";
    const { level } = pendingCreate;
    hideCreateTaskModal();
    try {
      const res = await API.createTask({
        xai_level: level,
        title,
        model: "",
        treatment: "",
        result: {},
      });
      if (res.error) throw new Error(res.error);
      const data = await API.tasksWithMeta();
      renderTaskList(data.tasks || data, data.xai_level_names || {});
    } catch (err) {
      alert("Failed to add task: " + err.message);
    }
  }

  // Create XAI: click main button to toggle dropdown (hover also works)
  el.btnCreateTask?.addEventListener("click", (e) => {
    e.stopPropagation();
    el.createTaskWrap?.classList.toggle("dropdown-open");
  });

  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (e.target.closest(".create-task-wrap")) return;
    el.createTaskWrap?.classList.remove("dropdown-open");
  });

  $$(".create-task-option").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      el.createTaskWrap?.classList.remove("dropdown-open");
      const level = btn.dataset.level;
      const name = btn.dataset.name || "";
      if (level) showCreateTaskModal(level, name);
    });
  });

  createTaskCancel?.addEventListener("click", hideCreateTaskModal);
  createTaskConfirm?.addEventListener("click", submitCreateTask);
  createNameInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submitCreateTask();
    if (e.key === "Escape") hideCreateTaskModal();
  });
  modalCreateName?.addEventListener("click", (e) => {
    if (e.target === modalCreateName) hideCreateTaskModal();
  });

  // Run button: show Play+RUN or Square+Stop
  function setRunButtonState(isRunning) {
    if (!el.btnRun) return;
    if (isRunning) {
      el.btnRun.classList.add("is-running");
      el.btnRun.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg> Stop';
    } else {
      el.btnRun.classList.remove("is-running");
      el.btnRun.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> RUN';
    }
  }

  // ----- Run with session check -----
  async function doRun(forceLoadModel = false) {
    const model = el.sidebarModel.value;
    const treatment = el.sidebarTreatment.value.trim() || "";
    const inputSetting = { ...gatherTaskInput(), model, treatment };

    if (forceLoadModel) {
      try {
        const res = await API.loadModel({ model, treatment });
        if (res.error) throw new Error(res.error);
        session = { loaded_model: model, treatment };
        updateSessionUI();
      } catch (err) {
        alert("Model load failed: " + err.message);
        return;
      }
    }

    runAbortController = new AbortController();
    const generationStatus = document.getElementById("generation-status");
    setRunButtonState(true);
    if (generationStatus) {
      generationStatus.textContent = "Working on it...";
      generationStatus.classList.add("visible");
    }
    try {
      const r = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, treatment, input_setting: inputSetting }),
        signal: runAbortController.signal,
      });
      const res = await r.json().then((body) => ({ ok: r.ok, ...body }));

      if (!res.ok && res.error === "session_mismatch") {
        pendingRunAfterConfirm = { model, treatment };
        el.modalMessage.textContent =
          "Loaded Model + Treatment does not match the current session. Load the model with this setting?";
        el.modalConfirm.classList.add("visible");
        return;
      }

      if (res.error) {
        alert(res.error);
        return;
      }

      const taskId = window.PNP_CURRENT_TASK_ID;
      const isCompletionResult = res && "generated_text" in res;

      function renderResultContent() {
        if (isCompletionResult) {
          return `
            <div class="results-completion-wrap">
              <h3>Generated</h3>
              <pre class="results-completion-text">${escapeHtml(String(res.generated_text ?? ""))}</pre>
              <details class="results-completion-meta">
                <summary>Parameters &amp; full result</summary>
                <pre class="results-json">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
              </details>
            </div>
          `;
        }
        return `
          <div style="padding: 16px;">
            <h3>Run Result</h3>
            <pre style="background: var(--panel); padding: 12px; border-radius: 6px; overflow: auto; font-size: 12px;">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
          </div>
        `;
      }

      el.resultsPlaceholder.classList.add("hidden");
      el.resultsContent.classList.add("visible");
      el.resultsContent.innerHTML = renderResultContent();

      if (taskId) {
        await API.updateTask(taskId, { result: res, model, treatment });
        if (el.inputSettingTrigger) {
          el.inputSettingTrigger.dataset.taskModel = model;
          el.inputSettingTrigger.dataset.taskTreatment = treatment;
        }
        updateInputSettingTriggerText();
        document.getElementById("results-content").innerHTML = renderResultContent();
        document.getElementById("results-placeholder")?.classList.add("hidden");
        document.getElementById("results-content")?.classList.add("visible");
      } else {
        const title = prompt("Enter task title (will be saved):", "Task " + new Date().toLocaleString());
        if (title) {
          await API.createTask({
            xai_level: currentTaskLevel,
            title,
            model,
            treatment,
            result: res,
          });
          const data = await API.tasksWithMeta();
          renderTaskList(data.tasks || data, data.xai_level_names || {});
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      alert(err.message || String(err));
    } finally {
      runAbortController = null;
      setRunButtonState(false);
      if (document.getElementById("generation-status")) {
        const gs = document.getElementById("generation-status");
        gs.textContent = "";
        gs.classList.remove("visible");
      }
    }
  }

  // ----- Gather task-specific input from right Input Setting panel -----
  function gatherTaskInput() {
    const body = document.getElementById("input-setting-body");
    const inputs = body ? body.querySelectorAll("[data-task-input]") : [];
    const obj = {};
    inputs.forEach((el) => {
      const key = el.dataset.taskInput || el.name || el.id;
      if (key) obj[key] = el.value !== undefined ? el.value : el.textContent;
    });
    return obj;
  }

  el.btnRun.addEventListener("click", () => {
    if (el.btnRun.classList.contains("is-running")) {
      if (!confirm("Generation을 중단하시겠습니까?")) return;
      if (runAbortController) runAbortController.abort();
      return;
    }
    doRun(false);
  });

  // ----- Modal: confirm model load -----
  el.modalCancel.addEventListener("click", () => {
    el.modalConfirm.classList.remove("visible");
    pendingRunAfterConfirm = null;
  });

  // ----- Export / Import -----
  const btnExport = document.getElementById("btn-export");
  const btnImport = document.getElementById("btn-import");
  const importFileInput = document.getElementById("import-file-input");

  if (btnExport) {
    btnExport.addEventListener("click", () => {
      window.location.href = "/api/memory/export";
    });
  }

  if (btnImport && importFileInput) {
    btnImport.addEventListener("click", () => importFileInput.click());
    importFileInput.addEventListener("change", async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        const res = await fetch("/api/memory/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
        const json = await res.json();
        if (json.error) throw new Error(json.error);
        location.reload();
      } catch (err) {
        alert("Import failed: " + err.message);
      }
      importFileInput.value = "";
    });
  }

  el.modalConfirmBtn.addEventListener("click", async () => {
    el.modalConfirm.classList.remove("visible");
    if (pendingRunAfterConfirm) {
      const { model, treatment } = pendingRunAfterConfirm;
      pendingRunAfterConfirm = null;
      await doRun(true);
    }
  });

  // ----- Render task list -----
  function renderTaskList(tasks, xaiLevelNames = {}) {
    if (!tasks || typeof tasks !== "object") return;
    const entries = Object.entries(tasks).filter(([, items]) => Array.isArray(items) && items.length > 0);

    // Skip re-render if current DOM already matches new data (avoids flicker when refetch runs)
    if (el.taskPanelList) {
      if (entries.length === 0) {
        if (el.taskPanelList.querySelector(".task-panel-empty")) return;
      } else {
        const newItems = entries.flatMap(([, items]) => items.map((t) => ({ id: t.id, title: (t.title || "").trim() })));
        const existingItems = Array.from(el.taskPanelList.querySelectorAll(".task-panel-item")).map((li) => ({
          id: li.dataset.taskId || "",
          title: (li.querySelector(".task-panel-item-link")?.textContent || "").trim(),
        }));
        if (newItems.length === existingItems.length && newItems.every((n, i) => existingItems[i] && n.id === existingItems[i].id && n.title === existingItems[i].title)) {
          return;
        }
      }
    }

    const fragment = document.createDocumentFragment();
    for (const [level, items] of entries) {
      const levelKey = level.replace("xai_level_", "").replace(/_/g, ".");
      const levelLabel = levelKey + (xaiLevelNames[levelKey] ? " — " + xaiLevelNames[levelKey] : "");
      const group = document.createElement("li");
      group.className = "task-panel-group";
      group.innerHTML = `
        <div class="task-panel-group-title">
          <span class="task-panel-chevron" aria-hidden="true">▼</span>
          <span class="task-panel-group-label">${escapeHtml(levelLabel)}</span>
        </div>
        <ul class="task-panel-sublist">
          ${items
                .map(
                  (t) => `
            <li class="task-panel-item" data-task-id="${t.id}" data-level="${level}">
              <a href="/task/${t.id}" class="task-panel-item-link">${escapeHtml(t.title)}</a>
              <button type="button" class="task-panel-delete" data-task-id="${t.id}" title="Delete">×</button>
            </li>
          `
                )
                .join("")}
        </ul>
      `;
      fragment.appendChild(group);
    }

    el.taskPanelList.innerHTML = "";
    if (entries.length === 0) {
      el.taskPanelList.innerHTML = '<li class="task-panel-empty">No tasks created yet.</li>';
    } else {
      el.taskPanelList.appendChild(fragment);
    }
  }

  // ----- Load task into right panel (click panel item) -----
  function loadTaskIntoView(task) {
    if (!task) return;
    window.PNP_CURRENT_TASK_ID = task.id;
    currentTaskLevel = task.xai_level || "0.1";

    // Load default (current session), overlay with task data if present
    if (el.sidebarModel && task.model) el.sidebarModel.value = task.model;
    if (el.sidebarTreatment && task.treatment !== undefined) el.sidebarTreatment.value = String(task.treatment);
    session = {
      loaded_model: task.model || session.loaded_model,
      treatment: task.treatment !== undefined && task.treatment !== null ? String(task.treatment) : session.treatment,
    };
    updateSessionUI();

    // Input setting header
    const header = document.querySelector(".input-setting-header");
    if (header) header.textContent = `${task.xai_level || ""} — ${task.title || ""} (task-specific input)`;

    // 0.1.1 Completion: open Input Setting panel so form is visible
    if (task.xai_level === "0.1.1") {
      el.inputSettingPanel?.classList.add("visible");
      el.inputSettingTrigger?.classList.add("has-setting");
    }

    // Results: overlay task.result if present, else show placeholder
    if (task.result && Object.keys(task.result).length > 0) {
      el.resultsPlaceholder?.classList.add("hidden");
      el.resultsContent?.classList.add("visible");
      const res = task.result;
      const isCompletion = res && "generated_text" in res;
      if (isCompletion) {
        el.resultsContent.innerHTML = `
          <div class="results-completion-wrap">
            <h3>Generated</h3>
            <pre class="results-completion-text">${escapeHtml(String(res.generated_text ?? ""))}</pre>
            <details class="results-completion-meta">
              <summary>Parameters &amp; full result</summary>
              <pre class="results-json">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
            </details>
          </div>
        `;
      } else {
        el.resultsContent.innerHTML = `
          <div style="padding: 16px;">
            <h3 style="margin-top:0;">${escapeHtml(task.title || "")}</h3>
            <p><strong>XAI Level:</strong> ${escapeHtml(task.xai_level || "—")}</p>
            <p><strong>Model:</strong> ${escapeHtml(task.model || "—")}</p>
            <p><strong>Treatment:</strong> ${escapeHtml(task.treatment || "—")}</p>
            <p><strong>Created:</strong> ${escapeHtml(task.created_at || "—")}</p>
            <pre style="background: var(--panel); padding: 12px; border-radius: 6px; overflow: auto; font-size: 12px;">${escapeHtml(JSON.stringify(task.result || {}, null, 2))}</pre>
          </div>
        `;
      }
    } else {
      el.resultsPlaceholder?.classList.remove("hidden");
      el.resultsContent?.classList.remove("visible");
      if (el.resultsContent) el.resultsContent.innerHTML = "";
      if (el.resultsPlaceholder) el.resultsPlaceholder.innerHTML = `
        <span class="brand-title">PnP-XAI-LLM</span>
        <ul class="feature-list"><li></li><li></li><li></li></ul>
      `;
    }

    // Update active state
    el.taskPanelList?.querySelectorAll(".task-panel-item").forEach((li) => {
      li.classList.toggle("active", String(li.dataset.taskId) === String(task.id));
    });
  }

  // Double-click task title → inline rename (cancel delayed single-click navigate)
  el.taskPanelList?.addEventListener("dblclick", (e) => {
    const link = e.target.closest(".task-panel-item-link");
    if (!link) return;
    e.preventDefault();
    if (taskLinkNavigateTimeout) {
      clearTimeout(taskLinkNavigateTimeout);
      taskLinkNavigateTimeout = null;
    }
    const item = link.closest(".task-panel-item");
    const taskId = item?.dataset.taskId;
    if (!taskId) return;

    const currentTitle = link.textContent.trim();
    const input = document.createElement("input");
    input.type = "text";
    input.className = "task-panel-item-edit";
    input.value = currentTitle;
    input.setAttribute("data-task-id", taskId);

    const finish = (save) => {
      const newTitle = input.value.trim();
      input.removeEventListener("blur", onBlur);
      input.removeEventListener("keydown", onKey);
      link.textContent = save && newTitle ? newTitle : currentTitle;
      link.style.display = "";
      input.replaceWith(link);
      if (save && newTitle && newTitle !== currentTitle) {
        API.updateTask(taskId, { title: newTitle }).then((res) => {
          if (res.error) link.textContent = currentTitle;
        });
        const header = document.querySelector(".input-setting-header");
        if (header && window.PNP_CURRENT_TASK_ID === taskId) {
          const level = item?.dataset?.level?.replace("xai_level_", "").replace(/_/g, ".") || "";
          header.textContent = `${level} — ${newTitle} (task-specific input)`;
        }
      }
    };

    const onBlur = () => finish(true);
    const onKey = (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        input.blur();
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        finish(false);
      }
    };

    link.style.display = "none";
    link.after(input);
    input.focus();
    input.select();
    input.addEventListener("blur", onBlur);
    input.addEventListener("keydown", onKey);
  });

  // Toggle group expand/collapse (VSCode-style)
  el.taskPanelList?.addEventListener("click", async (e) => {
    const header = e.target.closest(".task-panel-group-title");
    if (header) {
      e.preventDefault();
      const group = header.closest(".task-panel-group");
      if (group) group.classList.toggle("collapsed");
      return;
    }

    const link = e.target.closest(".task-panel-item-link");
    if (link) {
      e.preventDefault();
      const href = link.getAttribute("href");
      if (href && href.startsWith("/task/")) {
        if (taskLinkNavigateTimeout) clearTimeout(taskLinkNavigateTimeout);
        taskLinkNavigateTimeout = setTimeout(() => {
          taskLinkNavigateTimeout = null;
          window.location.href = href;
        }, 250);
      }
      return;
    }

    const btn = e.target.closest(".task-panel-delete");
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const taskId = btn.dataset.taskId;
    if (!taskId) return;
    try {
      const res = await fetch(`/api/tasks/${taskId}`, { method: "DELETE" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const meta = await API.tasksWithMeta();
      renderTaskList(meta.tasks || meta, meta.xai_level_names || {});
      if (window.PNP_CURRENT_TASK_ID === taskId) {
        window.location.href = "/";
      }
    } catch (err) {
      alert("Failed to delete: " + err.message);
    }
  });

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  // ----- Init -----
  async function init() {
    try {
      const params = new URLSearchParams(location.search);
      const createLevel = params.get("create");
      if (createLevel) {
        currentTaskLevel = createLevel;
        session = { loaded_model: null, treatment: null };
        history.replaceState({}, "", "/");
      } else {
        session = await API.session();
      }
      if (session.loaded_model && el.sidebarModel) {
        el.sidebarModel.value = session.loaded_model;
      }
      if (session.treatment && el.sidebarTreatment) {
        el.sidebarTreatment.value = session.treatment;
      }
      updateSessionUI();
      const data = await API.tasksWithMeta();
      renderTaskList(data.tasks || data, data.xai_level_names || {});
      if (createLevel) {
        const option = document.querySelector(`.create-task-option[data-level="${createLevel}"]`);
        showCreateTaskModal(createLevel, "");
      }
      // 0.1.1 Completion task page: open Input Setting so Temperature/Input String form is visible
      if (window.PNP_CURRENT_TASK_LEVEL === "0.1.1" && el.inputSettingPanel) {
        el.inputSettingPanel.classList.add("visible");
        el.inputSettingTrigger?.classList.add("has-setting");
      }
    } catch (e) {
      console.error("Init error:", e);
    }
  }

  init();
})();
