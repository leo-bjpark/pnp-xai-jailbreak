(function () {
  if (window.CircuitInput) {
    CircuitInput.init("circuit-input-root");
  }

  var modelSelect = document.getElementById("model");
  var runBtn = document.getElementById("circuit-run");
  var statusEl = document.getElementById("circuit-status");
  var nextTokenEl = document.getElementById("circuit-next-token");
  var figureEl = document.getElementById("circuit-heatmap-figure");
  var errorEl = document.getElementById("circuit-error");

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
  }
  function showError(msg) {
    if (errorEl) {
      errorEl.textContent = msg;
      errorEl.hidden = false;
    }
  }
  function hideError() {
    if (errorEl) errorEl.hidden = true;
  }

  var currentView = "heatmap";

  function attentionToColor(v, maxVal) {
    if (maxVal <= 0) return "rgb(42, 47, 66)";
    var p = Math.min(1, v / maxVal);
    var r = Math.round(34 + p * (97 - 34));
    var g = Math.round(42 + p * (218 - 42));
    var b = Math.round(66 + p * (251 - 66));
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function buildHeatmap(tokens, layerAttn, maxVal) {
    var grid = document.createElement("div");
    grid.className = "circuit-heatmap-grid";
    grid.style.setProperty("--circuit-cols", tokens.length + 1);
    grid.style.setProperty("--circuit-rows", layerAttn.length + 1);

    for (var layer = layerAttn.length - 1; layer >= 0; layer--) {
      var yLabel = document.createElement("div");
      yLabel.className = "circuit-heatmap-y-label";
      yLabel.textContent = "L" + layer;
      yLabel.title = "Layer " + layer;
      grid.appendChild(yLabel);
      var rowData = layerAttn[layer];
      for (var pos = 0; pos < rowData.length; pos++) {
        var cell = document.createElement("div");
        cell.className = "circuit-heatmap-cell";
        cell.style.background = attentionToColor(rowData[pos], maxVal);
        cell.title = "Layer " + layer + ", Token " + pos + " (" + tokens[pos] + "): " + rowData[pos].toFixed(4);
        grid.appendChild(cell);
      }
    }
    var corner = document.createElement("div");
    corner.className = "circuit-heatmap-cell circuit-heatmap-corner";
    corner.setAttribute("aria-label", "Y: Layer / X: Token");
    grid.appendChild(corner);
    tokens.forEach(function (t, i) {
      var th = document.createElement("div");
      th.className = "circuit-heatmap-x-label";
      th.textContent = t;
      th.title = "Token " + i + ": " + t;
      grid.appendChild(th);
    });
    return grid;
  }

  function buildCircuitGraph(tokens, layerAttn, maxVal) {
    var L = layerAttn.length;
    var T = tokens.length;
    var leftPad = 40;
    var topPad = 32;
    var bottomPad = 28;
    var nodeR = 3;
    var layerHeight = 28;
    var tokenGap = 20;
    var w = leftPad + (T - 1) * tokenGap + 24;
    var h = topPad + (L - 1) * layerHeight + bottomPad;

    function x(tokenIdx) { return leftPad + tokenIdx * tokenGap; }
    function y(layerIdx) { return topPad + (L - 1 - layerIdx) * layerHeight; }

    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "circuit-graph-svg");
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "auto");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

    var gLabels = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gLabels.setAttribute("class", "circuit-graph-labels");
    for (var ly = 0; ly < L; ly++) {
      var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "circuit-layer-label");
      t.setAttribute("x", 6);
      t.setAttribute("y", y(ly) + 4);
      t.textContent = "L" + ly;
      gLabels.appendChild(t);
    }
    var labelY = topPad + (L - 1) * layerHeight + 18;
    for (var ti = 0; ti < T; ti++) {
      var tt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tt.setAttribute("class", "circuit-token-label");
      tt.setAttribute("x", x(ti));
      tt.setAttribute("y", labelY);
      tt.setAttribute("text-anchor", "middle");
      tt.textContent = tokens[ti];
      gLabels.appendChild(tt);
    }

    var defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    var filter = document.createElementNS("http://www.w3.org/2000/svg", "filter");
    filter.setAttribute("id", "circuit-glow");
    filter.setAttribute("x", "-20%");
    filter.setAttribute("y", "-20%");
    filter.setAttribute("width", "140%");
    filter.setAttribute("height", "140%");
    var feGaussian = document.createElementNS("http://www.w3.org/2000/svg", "feGaussianBlur");
    feGaussian.setAttribute("stdDeviation", "1");
    feGaussian.setAttribute("result", "blur");
    var feMerge = document.createElementNS("http://www.w3.org/2000/svg", "feMerge");
    var feMergeNode1 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
    feMergeNode1.setAttribute("in", "blur");
    var feMergeNode2 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
    feMergeNode2.setAttribute("in", "SourceGraphic");
    filter.appendChild(feGaussian);
    filter.appendChild(feMerge);
    feMerge.appendChild(feMergeNode1);
    feMerge.appendChild(feMergeNode2);
    defs.appendChild(filter);
    svg.appendChild(defs);

    var gEdges = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gEdges.setAttribute("class", "circuit-edges");
    var gNodes = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gNodes.setAttribute("class", "circuit-nodes");

    var circuitThreshold = maxVal > 0 ? 0.35 * maxVal : 0;
    var lastIdx = T - 1;
    for (var layer = 0; layer < L - 1; layer++) {
      var x1 = x(lastIdx);
      var y1 = y(layer);
      var rowData = layerAttn[layer];
      for (var k = 0; k < rowData.length; k++) {
        var weight = rowData[k];
        var isCircuit = weight >= circuitThreshold;
        var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", "circuit-edge " + (isCircuit ? "circuit-edge--strong" : "circuit-edge--weak"));
        line.setAttribute("x1", x1);
        line.setAttribute("y1", y1);
        line.setAttribute("x2", x(k));
        line.setAttribute("y2", y(layer + 1));
        line.setAttribute("data-layer", layer);
        line.setAttribute("data-token", k);
        line.setAttribute("data-weight", weight.toFixed(4));
        gEdges.appendChild(line);
      }
    }

    for (var ly = 0; ly < L; ly++) {
      for (var tx = 0; tx < T; tx++) {
        var cx = x(tx);
        var cy = y(ly);
        var circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", "circuit-node");
        circle.setAttribute("cx", cx);
        circle.setAttribute("cy", cy);
        circle.setAttribute("r", nodeR);
        circle.setAttribute("data-layer", ly);
        circle.setAttribute("data-token", tx);
        if (ly < layerAttn.length && tx < layerAttn[ly].length) {
          var v = layerAttn[ly][tx];
          circle.setAttribute("data-weight", v.toFixed(4));
          circle.setAttribute("title", "Layer " + ly + ", Token " + tx + " (" + (tokens[tx] || "") + "): " + v.toFixed(4));
        }
        gNodes.appendChild(circle);
      }
    }

    svg.appendChild(gEdges);
    svg.appendChild(gNodes);
    svg.appendChild(gLabels);
    return svg;
  }

  function setView(view) {
    currentView = view;
    var heatmapContainer = figureEl && figureEl.querySelector(".circuit-heatmap-container");
    var graphContainer = figureEl && figureEl.querySelector(".circuit-graph-container");
    var tabs = document.querySelectorAll(".circuit-view-tab");
    tabs.forEach(function (t) {
      t.classList.toggle("active", t.getAttribute("data-view") === view);
    });
    if (heatmapContainer) heatmapContainer.classList.toggle("hidden", view !== "heatmap");
    if (graphContainer) graphContainer.classList.toggle("hidden", view !== "circuit");
  }

  function renderResult(data) {
    hideError();
    if (nextTokenEl) nextTokenEl.textContent = data.top_next_token || "—";

    var tokens = data.tokens || [];
    var layerAttn = data.layer_attention_from_last || [];
    if (tokens.length === 0 || layerAttn.length === 0) {
      if (figureEl) figureEl.innerHTML = "";
      return;
    }

    var maxVal = 0;
    for (var i = 0; i < layerAttn.length; i++) {
      for (var j = 0; j < layerAttn[i].length; j++) {
        if (layerAttn[i][j] > maxVal) maxVal = layerAttn[i][j];
      }
    }
    if (maxVal <= 0) maxVal = 1;

    if (figureEl) {
      figureEl.innerHTML = "";
      var heatmapContainer = document.createElement("div");
      heatmapContainer.className = "circuit-heatmap-container" + (currentView !== "heatmap" ? " hidden" : "");
      heatmapContainer.appendChild(buildHeatmap(tokens, layerAttn, maxVal));
      var graphContainer = document.createElement("div");
      graphContainer.className = "circuit-graph-container" + (currentView !== "circuit" ? " hidden" : "");
      graphContainer.appendChild(buildCircuitGraph(tokens, layerAttn, maxVal));
      figureEl.appendChild(heatmapContainer);
      figureEl.appendChild(graphContainer);
    }
  }

  document.querySelectorAll(".circuit-view-tab").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var view = btn.getAttribute("data-view");
      if (view) setView(view);
    });
  });

  if (runBtn && modelSelect && window.CircuitInput) {
    runBtn.addEventListener("click", function () {
      var err = CircuitInput.validate();
      if (err) {
        showError(err);
        return;
      }
      var body = CircuitInput.getPayload();
      if (!body || !body.model) {
        showError("모델을 선택하세요.");
        return;
      }

      runBtn.disabled = true;
      if (statusEl) statusEl.innerHTML = "<span class=\"loading-inline\"><span class=\"loading-spinner\"></span>Running…</span>";
      hideError();
      if (nextTokenEl) nextTokenEl.textContent = "—";
      if (figureEl) figureEl.innerHTML = "";

      fetch("/api/circuit_run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(function (r) {
          return r.json().catch(function () {
            return { error: r.status === 0 ? "서버에 연결할 수 없습니다." : "Server error (" + r.status + ")" };
          });
        })
        .then(function (data) {
          if (data.error) throw new Error(data.error);
          setStatus("Done.");
          document.body.classList.add("circuit-has-result");
          renderResult(data);
        })
        .catch(function (err) {
          var msg = err.message || "Request failed";
          if (msg === "Failed to fetch") msg = "서버에 연결할 수 없습니다.";
          setStatus("");
          document.body.classList.remove("circuit-has-result");
          showError(msg);
        })
        .finally(function () {
          runBtn.disabled = false;
        });
    });
  }
})();
