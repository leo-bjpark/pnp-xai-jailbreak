/**
 * Computation Test: 노드·레이어 가정 시각화.
 * Test Token = "Can I Make a bomb? No You can't." / tokens, num_layers 는 COMPUTATION_TEST_DATA 에서 로드.
 */
(function () {
  var data = window.COMPUTATION_TEST_DATA || { tokens: [], num_layers: 6 };
  var tokens = data.tokens || [];
  var L = data.num_layers || 6;
  var T = tokens.length;

  var sentenceEl = document.getElementById("comp-test-sentence");
  var tokenListEl = document.getElementById("comp-token-list");
  var numLayersEl = document.getElementById("comp-num-layers");
  var figureEl = document.getElementById("comp-viz-figure");

  if (tokenListEl) tokenListEl.textContent = tokens.join(", ") || "—";
  if (numLayersEl) numLayersEl.textContent = L;

  if (!figureEl || T === 0) return;

  var leftPad = 36;
  var topPad = 24;
  var bottomPad = 24;
  var nodeR = 3;
  var layerHeight = 24;
  var tokenGap = 22;
  var w = leftPad + (T - 1) * tokenGap + 20;
  var h = topPad + (L - 1) * layerHeight + bottomPad;

  function x(ti) { return leftPad + ti * tokenGap; }
  function y(ly) { return topPad + (L - 1 - ly) * layerHeight; }

  var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "comp-viz-svg");
  svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  svg.setAttribute("width", "100%");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  var gEdges = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gEdges.setAttribute("class", "comp-viz-edges");
  var lastIdx = T - 1;
  for (var layer = 0; layer < L - 1; layer++) {
    var x1 = x(lastIdx);
    var y1 = y(layer);
    for (var k = 0; k < T; k++) {
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "comp-viz-edge");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x(k));
      line.setAttribute("y2", y(layer + 1));
      gEdges.appendChild(line);
    }
  }

  var gNodes = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gNodes.setAttribute("class", "comp-viz-nodes");
  for (var ly = 0; ly < L; ly++) {
    for (var ti = 0; ti < T; ti++) {
      var circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("class", "comp-viz-node");
      circle.setAttribute("cx", x(ti));
      circle.setAttribute("cy", y(ly));
      circle.setAttribute("r", nodeR);
      circle.setAttribute("data-layer", ly);
      circle.setAttribute("data-token", ti);
      circle.setAttribute("title", "L" + ly + " · " + (tokens[ti] || ""));
      gNodes.appendChild(circle);
    }
  }

  var gLabels = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gLabels.setAttribute("class", "comp-viz-labels");
  for (var ly = 0; ly < L; ly++) {
    var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("class", "comp-viz-layer-label");
    t.setAttribute("x", 6);
    t.setAttribute("y", y(ly) + 4);
    t.textContent = "L" + ly;
    gLabels.appendChild(t);
  }
  var labelY = topPad + (L - 1) * layerHeight + 16;
  for (var ti = 0; ti < T; ti++) {
    var tt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    tt.setAttribute("class", "comp-viz-token-label");
    tt.setAttribute("x", x(ti));
    tt.setAttribute("y", labelY);
    tt.textContent = tokens[ti];
    gLabels.appendChild(tt);
  }

  svg.appendChild(gEdges);
  svg.appendChild(gNodes);
  svg.appendChild(gLabels);
  figureEl.appendChild(svg);
})();
