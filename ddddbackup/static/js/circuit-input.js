/**
 * Circuit Input Component
 * 모드별로 서로 다른 입력 박스 레이아웃을 가지며,
 * 각 모드는 최종적으로 해석 툴에 넘길 payload(텍스트/필드)를 제공한다.
 */
window.CircuitInput = (function () {
  var root = null;
  var modeSelect = null;
  var boxContainer = null;
  var currentMode = "generate";
  var currentBox = null; // { getPayload, validate }

  var MODES = {
    simple: {
      label: "단순 Forward",
      description: "주어진 입력 텍스트를 인코딩한 뒤 forward 해석",
      buildBox: function () {
        var wrap = document.createElement("div");
        wrap.className = "circuit-input-box circuit-input-box--simple";
        var label = document.createElement("label");
        label.className = "circuit-label";
        label.textContent = "입력 텍스트";
        var textarea = document.createElement("textarea");
        textarea.className = "circuit-prompt-input";
        textarea.rows = 4;
        textarea.placeholder = "인코딩할 텍스트를 입력하세요. 이 텍스트가 그대로 forward 해석에 사용됩니다.";
        textarea.setAttribute("data-field", "prompt");
        wrap.appendChild(label);
        wrap.appendChild(textarea);
        return {
          nodes: wrap,
          getPayload: function () {
            return { mode: "simple", prompt: (textarea.value || "").trim() };
          },
          validate: function () {
            if (!(textarea.value || "").trim()) return "입력 텍스트를 입력하세요.";
            return null;
          },
        };
      },
    },
    generate: {
      label: "생성기반",
      description: "프롬프트로 답을 생성한 뒤, 전체 시퀀스에 대해 forward 해석",
      buildBox: function () {
        var wrap = document.createElement("div");
        wrap.className = "circuit-input-box circuit-input-box--generate";
        var label = document.createElement("label");
        label.className = "circuit-label";
        label.textContent = "Prompt";
        var textarea = document.createElement("textarea");
        textarea.className = "circuit-prompt-input";
        textarea.rows = 4;
        textarea.placeholder = "프롬프트를 입력하세요. 생성 후 prompt + 생성 결과 전체에 대해 해석합니다.";
        textarea.setAttribute("data-field", "prompt");
        wrap.appendChild(label);
        wrap.appendChild(textarea);
        return {
          nodes: wrap,
          getPayload: function () {
            return { mode: "generate", prompt: (textarea.value || "").trim(), max_new_tokens: 64 };
          },
          validate: function () {
            if (!(textarea.value || "").trim()) return "Prompt를 입력하세요.";
            return null;
          },
        };
      },
    },
    user_qa: {
      label: "사용자 입력 질문-대답",
      description: "질문과 대답을 직접 입력한 뒤, chat_template으로 합쳐 forward 해석",
      buildBox: function () {
        var wrap = document.createElement("div");
        wrap.className = "circuit-input-box circuit-input-box--user_qa";
        var box1 = document.createElement("div");
        box1.className = "circuit-input-subbox";
        var label1 = document.createElement("label");
        label1.className = "circuit-label";
        label1.textContent = "User prompt (질문)";
        var textarea1 = document.createElement("textarea");
        textarea1.className = "circuit-prompt-input";
        textarea1.rows = 3;
        textarea1.placeholder = "사용자 질문 텍스트";
        textarea1.setAttribute("data-field", "user_prompt");
        box1.appendChild(label1);
        box1.appendChild(textarea1);
        var box2 = document.createElement("div");
        box2.className = "circuit-input-subbox";
        var label2 = document.createElement("label");
        label2.className = "circuit-label";
        label2.textContent = "Generation (대답)";
        var textarea2 = document.createElement("textarea");
        textarea2.className = "circuit-prompt-input";
        textarea2.rows = 3;
        textarea2.placeholder = "모델 대답 또는 직접 입력한 대답 텍스트";
        textarea2.setAttribute("data-field", "generation");
        box2.appendChild(label2);
        box2.appendChild(textarea2);
        wrap.appendChild(box1);
        wrap.appendChild(box2);
        return {
          nodes: wrap,
          getPayload: function () {
            return {
              mode: "user_qa",
              user_prompt: (textarea1.value || "").trim(),
              generation: (textarea2.value || "").trim(),
            };
          },
          validate: function () {
            var u = (textarea1.value || "").trim();
            var g = (textarea2.value || "").trim();
            if (!u) return "User prompt (질문)를 입력하세요.";
            if (!g) return "Generation (대답)을 입력하세요.";
            return null;
          },
        };
      },
    },
  };

  function renderModeSelector() {
    var row = document.createElement("div");
    row.className = "circuit-mode-row";
    var label = document.createElement("label");
    label.className = "circuit-label";
    label.textContent = "입력 모드";
    modeSelect = document.createElement("select");
    modeSelect.id = "circuit-mode";
    modeSelect.className = "toolbar-select circuit-mode-select";
    Object.keys(MODES).forEach(function (key) {
      var opt = document.createElement("option");
      opt.value = key;
      opt.textContent = MODES[key].label;
      modeSelect.appendChild(opt);
    });
    modeSelect.value = currentMode;
    row.appendChild(label);
    row.appendChild(modeSelect);
    return row;
  }

  function renderBoxForMode(mode) {
    var config = MODES[mode];
    if (!config) return null;
    return config.buildBox();
  }

  function showBox(mode) {
    currentMode = mode;
    if (!boxContainer) return;
    boxContainer.innerHTML = "";
    currentBox = renderBoxForMode(mode);
    if (currentBox && currentBox.nodes) {
      boxContainer.appendChild(currentBox.nodes);
    }
  }

  function init(containerId) {
    root = document.getElementById(containerId);
    if (!root) return;
    root.innerHTML = "";
    root.className = "circuit-input-card";

    root.appendChild(renderModeSelector());
    boxContainer = document.createElement("div");
    boxContainer.className = "circuit-input-box-container";
    root.appendChild(boxContainer);
    showBox(currentMode);

    if (modeSelect) {
      modeSelect.addEventListener("change", function () {
        showBox(modeSelect.value);
      });
    }

    var actions = document.createElement("div");
    actions.className = "circuit-actions";
    var runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.id = "circuit-run";
    runBtn.className = "toolbar-button";
    runBtn.textContent = "Run";
    var statusSpan = document.createElement("span");
    statusSpan.className = "circuit-status";
    statusSpan.id = "circuit-status";
    actions.appendChild(runBtn);
    actions.appendChild(statusSpan);
    root.appendChild(actions);
  }

  function getPayload() {
    if (!currentBox || !currentBox.getPayload) return null;
    var base = currentBox.getPayload();
    base.model = (document.getElementById("model") && document.getElementById("model").value) || "";
    return base;
  }

  function validate() {
    if (!currentBox || !currentBox.validate) return "모드를 선택해 주세요.";
    return currentBox.validate();
  }

  return {
    init: init,
    getPayload: getPayload,
    validate: validate,
    getMode: function () {
      return currentMode;
    },
  };
})();
