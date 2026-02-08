(function () {
  // 공통: 페이지 로드 시 localStorage에서 로드된 모델 복원 (다른 패널로 이동해도 유지)
  var storedModel = localStorage.getItem("loadedModel");
  var loadedSpan = document.getElementById("global-loaded-model");
  if (loadedSpan) loadedSpan.textContent = storedModel || "—";
  var globalSelect = document.getElementById("model");
  if (globalSelect && storedModel) {
    var found = Array.prototype.some.call(globalSelect.options, function (o) { return o.value === storedModel; });
    if (found) globalSelect.value = storedModel;
  }

  const loadBtn = document.getElementById("load-model");
  const modelSelect = document.getElementById("model");
  const cudaDevicesInput = document.getElementById("cuda-devices");
  const cudaExportBtn = document.getElementById("cuda-export-btn");
  const statusEl = document.getElementById("model-status");
  const statusPlaceholder = document.getElementById("model-status-placeholder");
  const statusBody = document.getElementById("model-status-body");
  const statusError = document.getElementById("model-status-error");
  const statusName = document.getElementById("status-name");
  const statusLayers = document.getElementById("status-layers");
  const statusHeads = document.getElementById("status-heads");
  const statusParams = document.getElementById("status-params");
  const statusDevices = document.getElementById("status-devices");

  // CUDA_VISIBLE_DEVICES: load current value, Export sets it on server
  if (cudaDevicesInput && cudaExportBtn) {
    fetch("/api/cuda_devices")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var v = data.cuda_visible_devices;
        if (v != null && v !== undefined) cudaDevicesInput.value = v;
      })
      .catch(function () {});
    cudaExportBtn.addEventListener("click", function () {
      var devices = (cudaDevicesInput.value || "").trim() || "0";
      cudaExportBtn.disabled = true;
      fetch("/api/set_cuda_devices", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ devices: devices }),
      })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.error) throw new Error(data.error);
          if (cudaDevicesInput) cudaDevicesInput.value = data.cuda_visible_devices || devices;
        })
        .catch(function (err) { alert(err.message || "Export failed"); })
        .finally(function () { cudaExportBtn.disabled = false; });
    });
  }

  if (!loadBtn || !modelSelect) return;

  function setStatus(msg, isError) {
    if (!statusEl) return;
    statusEl.style.color = isError ? "#f87171" : "var(--text)";
    statusEl.innerHTML = msg;
  }

  function showStatusPlaceholder() {
    if (statusPlaceholder) statusPlaceholder.hidden = false;
    if (statusBody) statusBody.hidden = true;
    if (statusError) statusError.hidden = true;
  }

  function showStatusBody() {
    if (statusPlaceholder) statusPlaceholder.hidden = true;
    if (statusBody) statusBody.hidden = false;
    if (statusError) statusError.hidden = true;
  }

  function showStatusError(msg) {
    if (statusPlaceholder) statusPlaceholder.hidden = true;
    if (statusBody) statusBody.hidden = true;
    if (statusError) {
      statusError.hidden = false;
      statusError.textContent = msg;
    }
  }

  function formatParams(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(2) + "K";
    return String(n);
  }

  var loadedModelKey = null;
  var chatMessages = [];
  var conversationId = null;
  var conversationWrap = document.getElementById("conversation-wrap");
  var conversationLog = document.getElementById("conversation-log");
  var conversationInput = document.getElementById("conversation-input");
  var conversationSend = document.getElementById("conversation-send");
  var conversationClearCache = document.getElementById("conversation-clear-cache");
  var conversationPlaceholder = document.getElementById("conversation-placeholder");
  var statusCacheTokens = document.getElementById("status-cache-tokens");

  function enableConversation(modelKey) {
    loadedModelKey = modelKey;
    conversationId = null;
    if (conversationPlaceholder) conversationPlaceholder.style.display = "none";
    if (conversationWrap) conversationWrap.style.display = "flex";
    appendInfo("새로운 모델이 로드되었습니다: " + modelKey);
  }

  function appendMessage(role, content) {
    if (!conversationLog) return;
    var msg = document.createElement("div");
    msg.className = "chat-msg " + role;
    var roleLabel = document.createElement("div");
    roleLabel.className = "chat-role";
    roleLabel.textContent = role === "user" ? "User" : "Assistant";
    var bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.textContent = content;
    msg.appendChild(roleLabel);
    msg.appendChild(bubble);
    conversationLog.appendChild(msg);
    conversationLog.scrollTop = conversationLog.scrollHeight;
  }

  function appendAssistantPlaceholder() {
    if (!conversationLog) return null;
    var msg = document.createElement("div");
    msg.className = "chat-msg assistant";
    var roleLabel = document.createElement("div");
    roleLabel.className = "chat-role";
    roleLabel.textContent = "Assistant";
    var bubble = document.createElement("div");
    bubble.className = "chat-bubble chat-generating";
    bubble.textContent = "...";
    msg.appendChild(roleLabel);
    msg.appendChild(bubble);
    conversationLog.appendChild(msg);
    conversationLog.scrollTop = conversationLog.scrollHeight;
    return bubble;
  }

  function appendInfo(text) {
    if (!conversationLog) return;
    var msg = document.createElement("div");
    msg.className = "chat-msg info";
    var bubble = document.createElement("div");
    bubble.className = "chat-bubble chat-info";
    bubble.textContent = "[INFO] " + text;
    msg.appendChild(bubble);
    conversationLog.appendChild(msg);
    conversationLog.scrollTop = conversationLog.scrollHeight;
  }

  function renderLoadedStatus(data) {
    if (!statusName) return;
    statusName.textContent = data.name || data.model_key;
    if (statusLayers) statusLayers.textContent = data.num_layers != null ? String(data.num_layers) : "—";
    if (statusHeads) statusHeads.textContent = data.num_heads != null ? String(data.num_heads) : "—";
    if (statusParams) statusParams.textContent = formatParams(data.num_parameters || 0);
    if (statusDevices) {
      statusDevices.innerHTML = "";
      (data.device_status || []).forEach(function (d) {
        var span = document.createElement("span");
        span.className = "status-device-item";
        var text = d.device + " — " + d.memory_gb + " GB used";
        if (d.capacity_gb != null) text += " / " + d.capacity_gb + " GB capacity";
        span.textContent = text;
        statusDevices.appendChild(span);
      });
    }
    if (statusCacheTokens) statusCacheTokens.textContent = "—";
    showStatusBody();
    if (data.model_key) enableConversation(data.model_key);
  }

  loadBtn.addEventListener("click", function () {
    const modelKey = (modelSelect.value || "").trim();
    if (!modelKey) {
      setStatus("모델을 선택하세요.", true);
      return;
    }
    loadBtn.disabled = true;
    setStatus("로딩 중…", false);
    if (statusEl) statusEl.innerHTML = "<span class=\"loading-inline\"><span class=\"loading-spinner\"></span>로딩 중…</span>";
    if (statusPlaceholder) {
      statusPlaceholder.innerHTML = "<span class=\"loading-inline\"><span class=\"loading-spinner\"></span>로딩 중…</span>";
      showStatusPlaceholder();
    }
    var globalLoadedEl = document.getElementById("global-loaded-model");
    if (globalLoadedEl) globalLoadedEl.innerHTML = "<span class=\"loading-inline\"><span class=\"loading-spinner\"></span>로딩 중…</span>";

    fetch("/api/load_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelKey }),
    })
      .then(function (r) {
        return r.json().catch(function () {
          return { error: r.status === 0 ? "서버에 연결할 수 없습니다." : "Server error (" + r.status + ")" };
        });
      })
      .then(function (data) {
        if (data.error) throw new Error(data.error);
        setStatus("Loaded: " + modelKey, false);
        var loadedEl = document.getElementById("global-loaded-model");
        if (loadedEl) loadedEl.textContent = modelKey;
        localStorage.setItem("loadedModel", modelKey);
        return fetch("/api/model_status?model=" + encodeURIComponent(modelKey));
      })
      .then(function (r) {
        return r.json().catch(function () { return { error: "Status parse error" }; });
      })
      .then(function (data) {
        if (data.error) throw new Error(data.error);
        renderLoadedStatus(data);
      })
      .catch(function (err) {
        var msg = err.message || "로드 실패";
        if (msg === "Failed to fetch") msg = "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.";
        setStatus(msg, true);
        if (statusPlaceholder) statusPlaceholder.textContent = "Load를 실행하면 여기에 상태가 표시됩니다.";
        showStatusError(msg);
        var loadedEl = document.getElementById("global-loaded-model");
        if (loadedEl) loadedEl.textContent = localStorage.getItem("loadedModel") || "—";
      })
      .finally(function () {
        loadBtn.disabled = false;
        if (statusPlaceholder && statusPlaceholder.hidden === false && statusPlaceholder.textContent === "로딩 중…") {
          statusPlaceholder.textContent = "Load를 실행하면 여기에 상태가 표시됩니다.";
        }
      });
  });

  if (conversationSend && conversationInput && conversationLog) {
    conversationSend.addEventListener("click", function () {
      var text = (conversationInput.value || "").trim();
      if (!text || !loadedModelKey) return;
      chatMessages.push({ role: "user", content: text });
      appendMessage("user", text);
      var placeholderBubble = appendAssistantPlaceholder();
      conversationInput.value = "";
      conversationSend.disabled = true;

      var body = conversationId
        ? { model: loadedModelKey, conversation_id: conversationId, content: text }
        : { model: loadedModelKey, messages: chatMessages };

      fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(function (r) {
          return r.json().catch(function () {
            return { error: r.status === 0 ? "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요." : "Server error (" + r.status + ")" };
          });
        })
        .then(function (data) {
          if (data.error) throw new Error(data.error);
          var content = (data.message && data.message.content) ? data.message.content : "";
          chatMessages.push({ role: "assistant", content: content });
          if (data.conversation_id) conversationId = data.conversation_id;
          if (statusCacheTokens && data.cache_token_count != null) statusCacheTokens.textContent = data.cache_token_count;
          if (placeholderBubble) {
            placeholderBubble.textContent = content;
            placeholderBubble.classList.remove("chat-generating");
          } else {
            appendMessage("assistant", content);
          }
        })
        .catch(function (err) {
          var msg = err.message || "Request failed";
          if (msg === "Failed to fetch") {
            msg = "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요. (Failed to fetch)";
          }
          var errMsg = "[Error: " + msg + "]";
          if (placeholderBubble) {
            placeholderBubble.textContent = errMsg;
            placeholderBubble.classList.remove("chat-generating");
          } else {
            appendMessage("assistant", errMsg);
          }
        })
        .finally(function () {
          conversationSend.disabled = false;
        });
    });
    conversationInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        conversationSend.click();
      }
    });
  }

  if (conversationClearCache) {
    conversationClearCache.addEventListener("click", function () {
      if (conversationId) {
        fetch("/api/chat_clear_cache", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conversation_id: conversationId }),
        }).catch(function () {});
      }
      conversationId = null;
      chatMessages = [];
      if (statusCacheTokens) statusCacheTokens.textContent = "—";
      appendInfo("캐시가 삭제되었습니다.");
    });
  }
})();
