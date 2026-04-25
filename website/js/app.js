/* =========================================================
   app.js  –  UI controller for PropNet demo
   ========================================================= */

const renderer = new GraphRenderer('graph-canvas');
let currentIdx = 0;
let currentExId = '';

// Global node-click handler wired in by graph.js
window.onNodeClick = (node, deg) => {
  renderNodeInfo(node, deg, currentExId);
};

// ── Bootstrap ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildSidebar();
  buildAblationChart();
  selectExample(0);
  initCounters();
  initNavScroll();
  initTabSwitcher();
});


// ── Sidebar list ──────────────────────────────────────────
function buildSidebar() {
  const list = document.getElementById('example-list');
  list.innerHTML = '';
  EXAMPLES.forEach((ex, i) => {
    const li = document.createElement('li');
    li.className = `example-item ${ex.label === 1 ? 'fake' : 'real'}`;
    li.id = `ex-item-${i}`;
    li.innerHTML = `
      <span class="ex-badge ${ex.label === 1 ? 'badge-fake' : 'badge-real'}">${ex.badge}</span>
      <div class="ex-meta">
        <span class="ex-title">${ex.title}</span>
        <span class="ex-ds">${ex.dataset}</span>
      </div>
      <span class="ex-nodes">${ex.graph.nodes.length}n</span>
    `;
    li.addEventListener('click', () => selectExample(i));
    list.appendChild(li);
  });
}

// ── Select example ────────────────────────────────────────
function selectExample(idx) {
  currentIdx = idx;
  const ex = EXAMPLES[idx];
  currentExId = ex.id;

  // Reset node inspector
  document.getElementById('tweet-inspector').innerHTML =
    '<div class="inspector-empty">Click any node in the graph to inspect its tweet metadata</div>';

  // Sidebar highlight
  document.querySelectorAll('.example-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`ex-item-${idx}`)?.classList.add('active');

  // Header info
  document.getElementById('graph-title').textContent  = ex.title;
  document.getElementById('graph-dataset').textContent = ex.dataset;
  const labelBadge = document.getElementById('graph-label-badge');
  labelBadge.textContent  = ex.badge;
  labelBadge.className    = `graph-main-badge ${ex.label === 1 ? 'badge-fake' : 'badge-real'}`;
  document.getElementById('graph-summary').textContent = ex.summary;

  // Alert banner for the BERT-fails case
  const alertBanner = document.getElementById('bert-alert');
  if (ex.id === 'pol_fake_002') {
    alertBanner.style.display = 'flex';
  } else {
    alertBanner.style.display = 'none';
  }

  // Render graph
  renderer.render(ex.graph, ex.label);

  // Motif panel
  renderMotifPanel(ex);

  // Predictions panel
  renderPredictions(ex);
}

// ── Motif panel ───────────────────────────────────────────
function renderMotifPanel(ex) {
  const container = document.getElementById('motif-bars');
  container.innerHTML = '';

  Object.entries(ex.motif).forEach(([key, { z }]) => {
    const label  = MOTIF_LABELS[key] || key;
    const isFakeInd = FAKE_INDICATIVE.has(key);
    // z is clamped to [-2.5, 2.5] for display
    const clamped = Math.max(-2.5, Math.min(2.5, z));
    const pct = ((clamped + 2.5) / 5) * 100;   // 0–100
    const midPct = 50;
    const isPositive = clamped >= 0;

    // Colour: red if fake-indicative + positive, green if real-indicative + negative, else muted
    let barClass = 'bar-neutral';
    if (isFakeInd && ex.label === 1 && isPositive)    barClass = 'bar-fake';
    else if (!isFakeInd && ex.label === 0 && isPositive) barClass = 'bar-real';
    else if (isFakeInd && ex.label === 0 && !isPositive) barClass = 'bar-real';
    else if (!isFakeInd && ex.label === 1 && !isPositive) barClass = 'bar-fake';

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
      <span class="motif-val ${z > 0 ? 'pos' : 'neg'}">${z > 0 ? '+' : ''}${z.toFixed(2)}σ</span>
    `;
    container.appendChild(row);
  });
}

// ── Predictions panel ─────────────────────────────────────
const MODEL_DISPLAY = {
  RF_motif:       { name: 'RF Motif',       icon: '🌲' },
  TextOnly_BERT:  { name: 'Text-Only BERT', icon: '📝' },
  GraphOnly_GAT:  { name: 'Graph-Only GAT', icon: '🕸️' },
  Hybrid_GAT:     { name: 'Hybrid GAT',     icon: '🔀' },
  PropNet_Fusion: { name: 'PropNet Fusion', icon: '⚡' },
};

function renderPredictions(ex) {
  const container = document.getElementById('pred-rows');
  container.innerHTML = '';

  Object.entries(ex.predictions).forEach(([modelKey, pred]) => {
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
      ? `<span class="alpha-note">α = ${pred.alpha.toFixed(2)}</span>`
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
            <div class="pred-fill real-fill" style="width:0%" data-target="${(pReal*100).toFixed(1)}"></div>
          </div>
          <span class="pred-bar-pct">${(pReal*100).toFixed(1)}%</span>
        </div>
        <div class="pred-bar-wrap">
          <span class="pred-bar-label fake-lbl">FAKE</span>
          <div class="pred-track">
            <div class="pred-fill fake-fill" style="width:0%" data-target="${(pFake*100).toFixed(1)}"></div>
          </div>
          <span class="pred-bar-pct">${(pFake*100).toFixed(1)}%</span>
        </div>
      </div>
    `;
    container.appendChild(row);
  });

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll('.pred-fill').forEach(bar => {
      const target = bar.dataset.target;
      bar.style.transition = 'width 0.7s cubic-bezier(0.4,0,0.2,1)';
      bar.style.width = target + '%';
    });
  });
}

// ── Ablation bar chart (vanilla canvas-free) ──────────────
function buildAblationChart() {
  const container = document.getElementById('ablation-chart');
  container.innerHTML = '';

  const maxAUC = 1.0;
  MODEL_SUMMARY.forEach((m, i) => {
    const isBest = m.model === 'PropNet_Fusion';
    const wrap = document.createElement('div');
    wrap.className = `abl-col ${isBest ? 'abl-hero' : ''}`;
    const pct = (m.auc / maxAUC * 100).toFixed(1);
    const names = {
      RF_motif: 'RF Motif', TextOnly_BERT: 'Text BERT',
      GraphOnly_GAT: 'Graph GAT', Hybrid_GAT: 'Hybrid GAT',
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

  // Animate bars when in view
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

// ── Hero stat counters ────────────────────────────────────
function initCounters() {
  const counters = [
    { id: 'stat-auc',    target: 99.3, decimals: 1, suffix: '%' },
    { id: 'stat-graphs', target: 5778, decimals: 0, suffix: '' },
    { id: 'stat-seeds',  target: 5,    decimals: 0, suffix: '' },
    { id: 'stat-models', target: 5,    decimals: 0, suffix: '' },
  ];
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
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
      observer.disconnect();
    });
  }, { threshold: 0.5 });
  const hero = document.getElementById('hero-stats');
  if (hero) observer.observe(hero);
}

// ── Sticky nav scroll ─────────────────────────────────────
function initNavScroll() {
  const nav = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 40);
  });
}

// ── Results tab switcher ──────────────────────────────────
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

// ── Node inspector (tweet metadata) ──────────────────────
function renderNodeInfo(node, deg, exId) {
  const panel = document.getElementById('tweet-inspector');
  const tweet = pickTweet(exId, node.group, node.id);

  const roleLabel = node.group === 'root' ? 'Original Post' :
                    node.group === 'hub'  ? 'Influencer Retweet' : 'Secondary Share';

  const depthDesc = node.depth === 0 ? 'Root — original posting'
                  : node.depth === 1 ? `Depth 1 — first-hop reshare`
                  : `Depth ${node.depth} — downstream spread`;

  if (!tweet) {
    panel.innerHTML = `<div class="inspector-empty">No tweet metadata for this node (depth ${node.depth})</div>`;
    return;
  }

  const color  = avatarColor(tweet.handle);
  const inits  = initials(tweet.name);

  panel.innerHTML = `
    <div class="tweet-card">
      <div class="tweet-card-header">
        <div class="tweet-avatar" style="background:${color}">${inits}</div>
        <div class="tweet-user">
          <div class="tweet-handle">${tweet.name}</div>
          <div class="tweet-username">@${tweet.handle}</div>
        </div>
        <span class="tweet-role-badge">${roleLabel}</span>
      </div>
      <div class="tweet-body">${tweet.text}</div>
      <div class="tweet-meta-row">
        <div class="tweet-meta-item">
          <span class="meta-label">Retweets</span>
          <span class="meta-val">${tweet.rt}</span>
        </div>
        <div class="tweet-meta-item">
          <span class="meta-label">Likes</span>
          <span class="meta-val">${tweet.likes}</span>
        </div>
        <div class="tweet-meta-item">
          <span class="meta-label">Replies</span>
          <span class="meta-val">${tweet.replies}</span>
        </div>
        <div class="tweet-meta-item">
          <span class="meta-label">Posted</span>
          <span class="meta-val" style="font-size:0.68rem">${tweet.time}</span>
        </div>
      </div>
      <div class="tweet-struct-row">
        <div class="struct-pill">
          <span class="sp-label">Graph Role</span>
          <span class="sp-val">${node.group}</span>
        </div>
        <div class="struct-pill">
          <span class="sp-label">Cascade Depth</span>
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

