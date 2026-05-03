/* =========================================================
   app.js  –  UI controller for PropNet demo
   Now wired to the live Flask backend at /api/sample.
   ========================================================= */

const renderer = new GraphRenderer('graph-canvas');
const API_BASE = ''; // same-origin (Flask serves the site)
let currentGraph = null;     // last fetched payload

// Global node-click handler wired in by graph.js
window.onNodeClick = (node, deg) => {
  renderNodeInfo(node, deg);
};

// ── Bootstrap ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Hero counters fire immediately — they are above the fold.
  initCounters();
  initNavScroll();
  initTabSwitcher();
  buildAblationChart();

  // Build the live-demo control panel and load the index.
  buildLiveControls();
  await loadIndex();
  await fetchSample({ label: 'any' });
});


// ── Sidebar list (real test graphs + filter buttons) ──────
let TEST_INDEX = [];

function buildLiveControls() {
  const list = document.getElementById('example-list');
  if (!list) return;
  list.innerHTML = `
    <div class="live-controls">
      <button class="live-btn live-btn-fake" data-label="fake">🎲 Random FAKE</button>
      <button class="live-btn live-btn-real" data-label="real">🎲 Random REAL</button>
      <button class="live-btn live-btn-any"  data-label="any">🎲 Any</button>
    </div>
    <div class="live-status" id="live-status">Loading test set…</div>
    <h4 class="sidebar-subhead">Test-set browse</h4>
    <ul id="example-list-inner" class="example-list-inner"></ul>
  `;
  list.querySelectorAll('.live-btn').forEach(btn => {
    btn.addEventListener('click', () => fetchSample({ label: btn.dataset.label }));
  });
}

async function loadIndex() {
  try {
    const r = await fetch(`${API_BASE}/api/index`);
    const json = await r.json();
    TEST_INDEX = json.items || [];
    const inner = document.getElementById('example-list-inner');
    if (!inner) return;
    inner.innerHTML = '';
    TEST_INDEX.slice(0, 80).forEach(item => {
      const li = document.createElement('li');
      li.className = `example-item ${item.label === 1 ? 'fake' : 'real'}`;
      li.innerHTML = `
        <span class="ex-badge ${item.label === 1 ? 'badge-fake' : 'badge-real'}">${item.label === 1 ? 'FAKE' : 'REAL'}</span>
        <div class="ex-meta">
          <span class="ex-title">${item.id}</span>
          <span class="ex-ds">${item.dataset}</span>
        </div>
        <span class="ex-nodes">${item.num_nodes}n</span>
      `;
      li.addEventListener('click', () => fetchSample({ idx: item.idx }));
      inner.appendChild(li);
    });
    setStatus(`${TEST_INDEX.length} test-set graphs available`);
  } catch (err) {
    setStatus('⚠ Backend offline. Start with: python website/api/server.py');
    console.error(err);
  }
}

function setStatus(msg) {
  const el = document.getElementById('live-status');
  if (el) el.textContent = msg;
}


// ── Fetch + render one sample ─────────────────────────────
async function fetchSample({ label = 'any', idx = null } = {}) {
  setStatus('Running PropNet inference…');
  try {
    const url = idx != null
      ? `${API_BASE}/api/sample?idx=${idx}`
      : `${API_BASE}/api/sample?label=${encodeURIComponent(label)}`;
    const t0 = performance.now();
    const r = await fetch(url);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${r.status}`);
    }
    const data = await r.json();
    const dt = (performance.now() - t0).toFixed(0);
    currentGraph = data;
    renderSample(data, dt);
    setStatus(`Inference complete · ${dt} ms · graph ${data.id}`);
  } catch (err) {
    setStatus(`⚠ ${err.message}`);
    console.error(err);
  }
}

function renderSample(data, latencyMs) {
  // Reset inspector
  const insp = document.getElementById('tweet-inspector');
  if (insp) insp.innerHTML =
    '<div class="inspector-empty">Click any node in the graph to see its position in the cascade</div>';

  // Header info
  document.getElementById('graph-title').textContent = `${data.id}`;
  document.getElementById('graph-dataset').textContent = data.dataset;
  const labelBadge = document.getElementById('graph-label-badge');
  labelBadge.textContent = data.label_name;
  labelBadge.className = `graph-main-badge ${data.label === 1 ? 'badge-fake' : 'badge-real'}`;
  const summaryEl = document.getElementById('graph-summary');
  if (summaryEl) {
    const fusionPred = data.predictions.PropNet_Fusion;
    const verdict = fusionPred.correct ? 'Correct' : 'Misclassified';
    summaryEl.innerHTML = `
      <strong>Real UPFD test graph</strong> · ${data.num_nodes} nodes ·
      ${data.num_edges} edges · cascade depth ${data.max_depth} ·
      PropNet verdict: <strong>${verdict}</strong>
    `;
  }

  // Hide the static BERT-fail alert (it was tied to a hardcoded id).
  const alertBanner = document.getElementById('bert-alert');
  if (alertBanner) alertBanner.style.display = 'none';

  // Render graph
  renderer.render({ nodes: data.nodes, links: data.links }, data.label);

  // Confidence hero card (the headline numbers for sir)
  renderConfidenceCard(data, latencyMs);

  // Motif z-score panel
  renderMotifPanel(data);

  // Per-model predictions
  renderPredictions(data);
}


// ── Confidence card (NEW — prominent numeric display) ─────
function renderConfidenceCard(data, latencyMs) {
  const card = document.getElementById('confidence-card');
  if (!card) return;
  const fusion = data.predictions.PropNet_Fusion;
  const pFake = fusion.p_fake;
  const pReal = fusion.p_real;
  const conf = Math.max(pFake, pReal);
  const decision = pFake > pReal ? 'FAKE' : 'REAL';
  const decisionClass = pFake > pReal ? 'badge-fake' : 'badge-real';
  const correctCls = fusion.correct ? 'verdict-correct' : 'verdict-wrong';
  const correctTxt = fusion.correct ? '✓ Correct' : '✗ Misclassified';
  const alphaPct = (fusion.alpha * 100).toFixed(1);
  const textPct  = ((1 - fusion.alpha) * 100).toFixed(1);

  card.innerHTML = `
    <div class="conf-row-top">
      <span class="conf-label">PropNet Fusion verdict</span>
      <span class="conf-decision ${decisionClass}">${decision}</span>
      <span class="conf-correct ${correctCls}">${correctTxt}</span>
    </div>
    <div class="conf-row-confidence">
      <div class="conf-big-num">${(conf * 100).toFixed(2)}<span class="conf-pct">%</span></div>
      <div class="conf-big-label">confidence</div>
    </div>
    <div class="conf-grid">
      <div class="conf-stat">
        <span class="conf-k">p(REAL)</span>
        <span class="conf-v">${pReal.toFixed(4)}</span>
      </div>
      <div class="conf-stat">
        <span class="conf-k">p(FAKE)</span>
        <span class="conf-v">${pFake.toFixed(4)}</span>
      </div>
      <div class="conf-stat">
        <span class="conf-k">α gate</span>
        <span class="conf-v">${fusion.alpha.toFixed(3)}</span>
      </div>
      <div class="conf-stat">
        <span class="conf-k">latency</span>
        <span class="conf-v">${latencyMs} ms</span>
      </div>
    </div>
    <div class="conf-alpha-bar">
      <span class="alpha-stream alpha-graph" style="width:${alphaPct}%">
        Graph stream ${alphaPct}%
      </span>
      <span class="alpha-stream alpha-text" style="width:${textPct}%">
        Motif stream ${textPct}%
      </span>
    </div>
    <div class="conf-meta">
      Ground truth: <strong>${data.label_name}</strong> ·
      Source: <strong>${data.dataset}</strong> ·
      ID <code class="mono">${data.id}</code>
    </div>
  `;
}


// ── Motif panel (z-score bars) ────────────────────────────
function renderMotifPanel(data) {
  const container = document.getElementById('motif-bars');
  if (!container) return;
  container.innerHTML = '';

  Object.entries(data.motif).forEach(([key, { z, raw }]) => {
    const label = MOTIF_LABELS[key] || key;
    const isFakeInd = FAKE_INDICATIVE.has(key);
    const clamped = Math.max(-2.5, Math.min(2.5, z));
    const pct = ((clamped + 2.5) / 5) * 100;
    const midPct = 50;
    const isPositive = clamped >= 0;

    let barClass = 'bar-neutral';
    if (isFakeInd && data.label === 1 && isPositive)        barClass = 'bar-fake';
    else if (!isFakeInd && data.label === 0 && isPositive)  barClass = 'bar-real';
    else if (isFakeInd && data.label === 0 && !isPositive)  barClass = 'bar-real';
    else if (!isFakeInd && data.label === 1 && !isPositive) barClass = 'bar-fake';

    const row = document.createElement('div');
    row.className = 'motif-row';
    row.innerHTML = `
      <span class="motif-label">${label}</span>
      <div class="motif-track">
        <div class="motif-mid"></div>
        <div class="motif-fill ${barClass}"
             style="left:${isPositive ? midPct : pct}%;
                    width:${Math.abs(pct - midPct)}%;"></div>
      </div>
      <span class="motif-val ${z > 0 ? 'pos' : 'neg'}" title="raw=${raw.toFixed(3)}">
        ${z > 0 ? '+' : ''}${z.toFixed(2)}σ
      </span>
    `;
    container.appendChild(row);
  });
}


// ── Predictions panel ─────────────────────────────────────
const MODEL_DISPLAY = {
  RF_motif:       { name: 'RF Motif',       icon: '🌲' },
  TextOnly_BERT:  { name: 'Text-Only BERT', icon: '📝' },
  GraphOnly_GAT:  { name: 'Graph-Only GAT', icon: '🕸️' },
  PropNet_Fusion: { name: 'PropNet Fusion', icon: '⚡' },
};

const MODEL_ORDER = ['GraphOnly_GAT', 'PropNet_Fusion'];

function renderPredictions(data) {
  const container = document.getElementById('pred-rows');
  if (!container) return;
  container.innerHTML = '';

  MODEL_ORDER.forEach(modelKey => {
    const pred = data.predictions[modelKey];
    if (!pred) return;
    const { name, icon } = MODEL_DISPLAY[modelKey];
    const pFake = pred.p_fake;
    const pReal = 1 - pFake;
    const isFusion = modelKey === 'PropNet_Fusion';
    const row = document.createElement('div');
    row.className = `pred-row ${isFusion ? 'pred-hero' : ''}`;

    const verdict = pred.correct
      ? `<span class="verdict correct">✓</span>`
      : `<span class="verdict wrong">✗</span>`;

    const alphaNote = (isFusion && pred.alpha != null)
      ? `<span class="alpha-note">α = ${pred.alpha.toFixed(3)}</span>`
      : '';

    row.innerHTML = `
      <div class="pred-header">
        <span class="pred-name">${icon} ${name}</span>
        ${verdict}
        ${alphaNote}
      </div>
      <div class="pred-bars">
        <div class="pred-bar-wrap">
          <span class="pred-bar-label real-lbl">REAL</span>
          <div class="pred-track">
            <div class="pred-fill real-fill" style="width:0%" data-target="${(pReal*100).toFixed(2)}"></div>
          </div>
          <span class="pred-bar-pct">${(pReal*100).toFixed(2)}%</span>
        </div>
        <div class="pred-bar-wrap">
          <span class="pred-bar-label fake-lbl">FAKE</span>
          <div class="pred-track">
            <div class="pred-fill fake-fill" style="width:0%" data-target="${(pFake*100).toFixed(2)}"></div>
          </div>
          <span class="pred-bar-pct">${(pFake*100).toFixed(2)}%</span>
        </div>
      </div>
    `;
    container.appendChild(row);
  });

  requestAnimationFrame(() => {
    document.querySelectorAll('.pred-fill').forEach(bar => {
      const target = bar.dataset.target;
      bar.style.transition = 'width 0.7s cubic-bezier(0.4,0,0.2,1)';
      bar.style.width = target + '%';
    });
  });
}


// ── Ablation bar chart (still uses the static MODEL_SUMMARY) ──
function buildAblationChart() {
  const container = document.getElementById('ablation-chart');
  if (!container || typeof MODEL_SUMMARY === 'undefined') return;
  container.innerHTML = '';

  const maxAUC = 1.0;
  MODEL_SUMMARY.forEach((m) => {
    const isBest = m.model === 'PropNet_Fusion';
    const wrap = document.createElement('div');
    wrap.className = `abl-col ${isBest ? 'abl-hero' : ''}`;
    const pct = (m.auc / maxAUC * 100).toFixed(1);
    const names = {
      RF_motif: 'RF Motif', TextOnly_BERT: 'Text BERT',
      GraphOnly_GAT: 'Graph GAT',
      PropNet_Fusion: 'PropNet ⚡',
    };
    wrap.innerHTML = `
      <div class="abl-bar-wrap">
        <div class="abl-bar" style="height:0%" data-target="${pct}">
          <span class="abl-auc">${m.auc.toFixed(3)}</span>
        </div>
      </div>
      <span class="abl-std">±${m.auc_std.toFixed(3)}</span>
      <span class="abl-name">${names[m.model]}</span>
    `;
    container.appendChild(wrap);
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('.abl-bar').forEach((bar, i) => {
          setTimeout(() => {
            bar.style.transition = 'height 0.8s cubic-bezier(0.4,0,0.2,1)';
            bar.style.height = bar.dataset.target + '%';
          }, i * 120);
        });
        observer.disconnect();
      }
    });
  }, { threshold: 0.3 });
  observer.observe(container);
}


// ── Hero stat counters (fire immediately, not on scroll) ──
function initCounters() {
  const counters = [
    { id: 'stat-auc',    target: 98.9, decimals: 1, suffix: '%' },
    { id: 'stat-graphs', target: 5778, decimals: 0, suffix: '' },
    { id: 'stat-seeds',  target: 5,    decimals: 0, suffix: '' },
    { id: 'stat-models', target: 4,    decimals: 0, suffix: '' },
  ];
  counters.forEach(c => {
    const el = document.getElementById(c.id);
    if (!el) return;
    let start = 0; const duration = 1600;
    const step = ts => {
      if (!start) start = ts;
      const prog = Math.min((ts - start) / duration, 1);
      const val = prog * c.target;
      el.textContent = (c.decimals ? val.toFixed(c.decimals) : Math.floor(val)) + c.suffix;
      if (prog < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

function initNavScroll() {
  const nav = document.querySelector('.navbar');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 40);
  });
}

function initTabSwitcher() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });
}


// ── Node inspector — now shows REAL graph-structural info ──
function renderNodeInfo(node, deg) {
  const panel = document.getElementById('tweet-inspector');
  if (!panel) return;

  const roleLabel = node.group === 'root' ? 'Original Article (root)' :
                    node.group === 'hub'  ? 'First-hop reshare (hub)' :
                                            'Downstream share (leaf)';

  const depthDesc = node.depth === 0 ? 'Root — original posting'
                  : node.depth === 1 ? `Depth 1 — first-hop reshare`
                                     : `Depth ${node.depth} — downstream cascade`;

  panel.innerHTML = `
    <div class="tweet-card">
      <div class="tweet-card-header">
        <div class="tweet-avatar" style="background:#4338ca">N${node.id}</div>
        <div class="tweet-user">
          <div class="tweet-handle">${roleLabel}</div>
          <div class="tweet-username">node #${node.id}</div>
        </div>
        <span class="tweet-role-badge">${node.group}</span>
      </div>
      <div class="tweet-body" style="padding:14px 16px;font-style:italic;color:var(--text-2);font-size:0.85rem;line-height:1.5;">
        UPFD nodes carry only DistilBERT embeddings — no raw text is bundled with
        the dataset. The model uses these 768-dim vectors plus the propagation
        topology you see on the left.
      </div>
      <div class="tweet-struct-row">
        <div class="struct-pill">
          <span class="sp-label">Graph role</span>
          <span class="sp-val">${node.group}</span>
        </div>
        <div class="struct-pill">
          <span class="sp-label">Cascade depth</span>
          <span class="sp-val">${node.depth}</span>
        </div>
        <div class="struct-pill">
          <span class="sp-label">Degree</span>
          <span class="sp-val">${deg}</span>
        </div>
        <div class="struct-pill">
          <span class="sp-label">Node ID</span>
          <span class="sp-val">#${node.id}</span>
        </div>
      </div>
    </div>
  `;
}
