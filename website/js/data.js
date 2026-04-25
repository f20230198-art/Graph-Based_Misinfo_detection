/* =========================================================
   data.js  –  PropNet demo graph data
   Synthetic propagation graphs modelled after UPFD statistics.
   ========================================================= */

// ── Seeded RNG ────────────────────────────────────────────
function makeSeed(n) {
  let s = n | 0;
  return () => {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

// ── Graph builders ────────────────────────────────────────
function buildFakeGraph(numNodes, numHubs, crossFrac, seed) {
  const rng = makeSeed(seed);
  const nodes = [];
  const edgeSet = new Set();
  const addEdge = (a, b) => { if (a !== b) edgeSet.add(`${Math.min(a,b)}_${Math.max(a,b)}`); };

  nodes.push({ id: 0, depth: 0, group: 'root' });
  for (let h = 1; h <= numHubs; h++) {
    nodes.push({ id: h, depth: 1, group: 'hub' });
    addEdge(0, h);
  }
  let nid = numHubs + 1;
  while (nid < numNodes) {
    const hub = 1 + Math.floor(rng() * numHubs);
    nodes.push({ id: nid, depth: 2, group: 'leaf' });
    addEdge(hub, nid);
    nid++;
  }
  const numCross = Math.floor(numNodes * crossFrac);
  for (let i = 0, tries = 0; i < numCross && tries < numCross * 8; tries++) {
    const a = 1 + Math.floor(rng() * (numNodes - 1));
    const b = 1 + Math.floor(rng() * (numNodes - 1));
    const key = `${Math.min(a,b)}_${Math.max(a,b)}`;
    if (a !== b && !edgeSet.has(key)) { edgeSet.add(key); i++; }
  }
  const links = [...edgeSet].map(k => {
    const [s, t] = k.split('_').map(Number);
    return { source: s, target: t };
  });
  return { nodes, links };
}

function buildRealGraph(numNodes, branchLo, branchHi, seed) {
  const rng = makeSeed(seed);
  const nodes = [{ id: 0, depth: 0, group: 'root' }];
  const links = [];
  let queue = [0], nid = 1, depth = 0;
  while (nid < numNodes && queue.length > 0 && depth < 7) {
    depth++;
    const next = [];
    for (const parent of queue) {
      const k = branchLo + Math.floor(rng() * (branchHi - branchLo + 1));
      for (let c = 0; c < k && nid < numNodes; c++) {
        nodes.push({ id: nid, depth, group: depth === 1 ? 'hub' : 'leaf' });
        links.push({ source: parent, target: nid });
        next.push(nid++);
      }
    }
    queue = next;
  }
  return { nodes, links };
}

// ── 8 curated demo examples ────────────────────────────────
const EXAMPLES = [
  // ── FAKE ──────────────────────────────────────────────────
  {
    id: 'pol_fake_001',
    label: 1,
    title: 'Fabricated Political Claim',
    dataset: 'PolitiFact',
    badge: 'FAKE',
    summary: 'Dense hub-and-spoke topology — coordinated retweet burst through 6 gateway accounts.',
    graph: buildFakeGraph(50, 6, 0.14, 42),
    motif: {
      triangle_count:       { raw: 4.12, z:  1.81 },
      max_star_size:        { raw: 3.87, z:  1.44 },
      avg_clustering:       { raw: 0.52, z:  1.19 },
      density:              { raw: 0.18, z:  0.73 },
      num_nodes_log:        { raw: 3.91, z:  0.21 },
      root_out_degree_frac: { raw: 0.24, z:  0.88 },
      depth_proxy:          { raw: 1.61, z: -1.20 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.84, correct: true  },
      TextOnly_BERT:  { p_fake: 0.76, correct: true  },
      GraphOnly_GAT:  { p_fake: 0.89, correct: true  },
      Hybrid_GAT:     { p_fake: 0.93, correct: true  },
      PropNet_Fusion: { p_fake: 0.981, alpha: 0.79, correct: true },
    },
  },
  {
    id: 'pol_fake_002',
    label: 1,
    title: 'Polished Disinformation (Text Fools BERT)',
    dataset: 'PolitiFact',
    badge: 'FAKE',
    summary: 'Professionally written — TextOnly BERT is fooled. PropNet catches it via the extreme star structure.',
    graph: buildFakeGraph(30, 1, 0.05, 77),
    motif: {
      triangle_count:       { raw: 0.69, z: -0.42 },
      max_star_size:        { raw: 5.24, z:  2.61 },
      avg_clustering:       { raw: 0.09, z: -0.88 },
      density:              { raw: 0.26, z:  1.31 },
      num_nodes_log:        { raw: 3.40, z: -0.18 },
      root_out_degree_frac: { raw: 0.41, z:  1.72 },
      depth_proxy:          { raw: 0.69, z: -1.80 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.71, correct: true  },
      TextOnly_BERT:  { p_fake: 0.38, correct: false },   // ← BERT fails!
      GraphOnly_GAT:  { p_fake: 0.82, correct: true  },
      Hybrid_GAT:     { p_fake: 0.79, correct: true  },
      PropNet_Fusion: { p_fake: 0.951, alpha: 0.87, correct: true },
    },
  },
  {
    id: 'gos_fake_001',
    label: 1,
    title: 'Celebrity Rumour Cascade',
    dataset: 'GossipCop',
    badge: 'FAKE',
    summary: 'Multiple overlapping clusters — bot-like amplification across 8 hub accounts with heavy triangulation.',
    graph: buildFakeGraph(68, 8, 0.18, 13),
    motif: {
      triangle_count:       { raw: 5.81, z:  2.34 },
      max_star_size:        { raw: 3.97, z:  1.52 },
      avg_clustering:       { raw: 0.61, z:  1.58 },
      density:              { raw: 0.15, z:  0.41 },
      num_nodes_log:        { raw: 4.22, z:  0.63 },
      root_out_degree_frac: { raw: 0.19, z:  0.44 },
      depth_proxy:          { raw: 1.39, z: -1.44 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.91, correct: true },
      TextOnly_BERT:  { p_fake: 0.68, correct: true },
      GraphOnly_GAT:  { p_fake: 0.94, correct: true },
      Hybrid_GAT:     { p_fake: 0.96, correct: true },
      PropNet_Fusion: { p_fake: 0.994, alpha: 0.71, correct: true },
    },
  },
  {
    id: 'gos_fake_002',
    label: 1,
    title: 'Health Hoax — Viral Spread',
    dataset: 'GossipCop',
    badge: 'FAKE',
    summary: 'Four broadcasting hubs driving rapid secondary sharing — high root-out-degree fraction.',
    graph: buildFakeGraph(42, 4, 0.12, 99),
    motif: {
      triangle_count:       { raw: 3.04, z:  1.24 },
      max_star_size:        { raw: 3.43, z:  1.08 },
      avg_clustering:       { raw: 0.44, z:  0.91 },
      density:              { raw: 0.21, z:  0.97 },
      num_nodes_log:        { raw: 3.74, z:  0.09 },
      root_out_degree_frac: { raw: 0.28, z:  1.14 },
      depth_proxy:          { raw: 1.10, z: -1.61 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.77, correct: true },
      TextOnly_BERT:  { p_fake: 0.81, correct: true },
      GraphOnly_GAT:  { p_fake: 0.87, correct: true },
      Hybrid_GAT:     { p_fake: 0.91, correct: true },
      PropNet_Fusion: { p_fake: 0.963, alpha: 0.68, correct: true },
    },
  },

  // ── REAL ──────────────────────────────────────────────────
  {
    id: 'pol_real_001',
    label: 0,
    title: 'Verified Political Report',
    dataset: 'PolitiFact',
    badge: 'REAL',
    summary: 'Classic branching tree — organic spread through diverse, independent users. Low clustering.',
    graph: buildRealGraph(38, 1, 3, 55),
    motif: {
      triangle_count:       { raw: 0.29, z: -1.42 },
      max_star_size:        { raw: 1.94, z: -0.71 },
      avg_clustering:       { raw: 0.07, z: -1.14 },
      density:              { raw: 0.06, z: -1.21 },
      num_nodes_log:        { raw: 3.64, z: -0.01 },
      root_out_degree_frac: { raw: 0.07, z: -0.82 },
      depth_proxy:          { raw: 2.08, z:  1.41 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.11, correct: true },
      TextOnly_BERT:  { p_fake: 0.22, correct: true },
      GraphOnly_GAT:  { p_fake: 0.09, correct: true },
      Hybrid_GAT:     { p_fake: 0.07, correct: true },
      PropNet_Fusion: { p_fake: 0.031, alpha: 0.33, correct: true },
    },
  },
  {
    id: 'pol_real_002',
    label: 0,
    title: 'Fact-Checked Breaking News',
    dataset: 'PolitiFact',
    badge: 'REAL',
    summary: 'Deep, sparse cascade — slow organic diffusion characteristic of real journalism.',
    graph: buildRealGraph(26, 1, 2, 88),
    motif: {
      triangle_count:       { raw: 0.00, z: -1.58 },
      max_star_size:        { raw: 1.61, z: -0.88 },
      avg_clustering:       { raw: 0.00, z: -1.28 },
      density:              { raw: 0.09, z: -0.88 },
      num_nodes_log:        { raw: 3.26, z: -0.37 },
      root_out_degree_frac: { raw: 0.09, z: -0.61 },
      depth_proxy:          { raw: 2.40, z:  1.78 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.08, correct: true },
      TextOnly_BERT:  { p_fake: 0.17, correct: true },
      GraphOnly_GAT:  { p_fake: 0.06, correct: true },
      Hybrid_GAT:     { p_fake: 0.05, correct: true },
      PropNet_Fusion: { p_fake: 0.021, alpha: 0.28, correct: true },
    },
  },
  {
    id: 'gos_real_001',
    label: 0,
    title: 'Legitimate Celebrity News',
    dataset: 'GossipCop',
    badge: 'REAL',
    summary: 'Wide, balanced tree with depth 5 — real coverage spreads gradually across unrelated accounts.',
    graph: buildRealGraph(52, 2, 3, 21),
    motif: {
      triangle_count:       { raw: 0.69, z: -0.98 },
      max_star_size:        { raw: 2.20, z: -0.54 },
      avg_clustering:       { raw: 0.11, z: -0.98 },
      density:              { raw: 0.05, z: -1.28 },
      num_nodes_log:        { raw: 3.95, z:  0.27 },
      root_out_degree_frac: { raw: 0.06, z: -0.88 },
      depth_proxy:          { raw: 2.56, z:  1.98 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.14, correct: true },
      TextOnly_BERT:  { p_fake: 0.29, correct: true },
      GraphOnly_GAT:  { p_fake: 0.11, correct: true },
      Hybrid_GAT:     { p_fake: 0.09, correct: true },
      PropNet_Fusion: { p_fake: 0.044, alpha: 0.38, correct: true },
    },
  },
  {
    id: 'gos_real_002',
    label: 0,
    title: 'Verified Entertainment Report',
    dataset: 'GossipCop',
    badge: 'REAL',
    summary: 'Moderate-depth tree — healthy diversity of sharing accounts, minimal triangulation.',
    graph: buildRealGraph(33, 1, 3, 66),
    motif: {
      triangle_count:       { raw: 0.00, z: -1.58 },
      max_star_size:        { raw: 2.30, z: -0.47 },
      avg_clustering:       { raw: 0.04, z: -1.11 },
      density:              { raw: 0.07, z: -1.11 },
      num_nodes_log:        { raw: 3.50, z: -0.08 },
      root_out_degree_frac: { raw: 0.08, z: -0.71 },
      depth_proxy:          { raw: 2.20, z:  1.58 },
    },
    predictions: {
      RF_motif:       { p_fake: 0.12, correct: true },
      TextOnly_BERT:  { p_fake: 0.24, correct: true },
      GraphOnly_GAT:  { p_fake: 0.13, correct: true },
      Hybrid_GAT:     { p_fake: 0.10, correct: true },
      PropNet_Fusion: { p_fake: 0.037, alpha: 0.31, correct: true },
    },
  },
];

// ── Aggregate summary stats (from results/upfd_multiseed_summary.csv) ──────
const MODEL_SUMMARY = [
  { model: 'RF_motif',       auc: 0.847, auc_std: 0.012, f1: 0.781, bal_acc: 0.782 },
  { model: 'TextOnly_BERT',  auc: 0.931, auc_std: 0.008, f1: 0.871, bal_acc: 0.872 },
  { model: 'GraphOnly_GAT',  auc: 0.958, auc_std: 0.005, f1: 0.904, bal_acc: 0.905 },
  { model: 'Hybrid_GAT',     auc: 0.981, auc_std: 0.003, f1: 0.942, bal_acc: 0.943 },
  { model: 'PropNet_Fusion', auc: 0.993, auc_std: 0.001, f1: 0.971, bal_acc: 0.972 },
];

const MOTIF_LABELS = {
  triangle_count:       'Triangle Count',
  max_star_size:        'Max Star Size',
  avg_clustering:       'Avg Clustering',
  density:              'Graph Density',
  num_nodes_log:        'Graph Size (log)',
  root_out_degree_frac: 'Root Hub Fraction',
  depth_proxy:          'Cascade Depth',
};

// Features where HIGH value → indicative of FAKE (for coloring)
const FAKE_INDICATIVE = new Set([
  'triangle_count','max_star_size','avg_clustering',
  'density','root_out_degree_frac',
]);

// ── Mock tweet metadata per example ───────────────────────
// Each example has root / hub / leaf tweet templates.
// Picked by node group + hashed node id for variety.
const TWEET_TEMPLATES = {
  pol_fake_001: {
    root: [
      { handle: 'BreakingPolitics24', name: 'Breaking Politics', text: 'CONFIRMED: Senior official caught accepting payments from foreign lobby group. Documents leaked to our source. Read the full story — share before it\'s removed.', time: '11:42 AM · Apr 14, 2024', rt: '4,218', likes: '11.2K', replies: '892' },
    ],
    hub: [
      { handle: 'TruthPatriot_US', name: 'Truth Patriot', text: 'I\'ve been saying this for MONTHS. The evidence was always there. Everyone needs to see this. RT before they delete it. #Corruption #ExposeThem', time: '11:47 AM · Apr 14, 2024', rt: '1,832', likes: '5.6K', replies: '341' },
      { handle: 'AmericaFirst2024', name: 'America First', text: 'This confirms what whistleblowers have been trying to tell us. The mainstream media won\'t touch this. Share widely.', time: '11:51 AM · Apr 14, 2024', rt: '2,104', likes: '6.8K', replies: '419' },
      { handle: 'RealNewsNetwork', name: 'Real News Network', text: 'Our analysts have verified the documents. This is REAL. Tag your representatives and demand answers. #AccountabilityNow', time: '11:55 AM · Apr 14, 2024', rt: '987', likes: '3.1K', replies: '207' },
    ],
    leaf: [
      { handle: 'concerned_citizen_88', name: 'Mark H.', text: 'Shared this with my family. Absolutely shocking if true.', time: '12:03 PM · Apr 14, 2024', rt: '42', likes: '128', replies: '7' },
      { handle: 'wake_up_people2024', name: 'Jenny W.', text: 'RT @TruthPatriot_US this is exactly what I was afraid of 😡', time: '12:08 PM · Apr 14, 2024', rt: '18', likes: '54', replies: '3' },
      { handle: 'patriot_news_fan', name: 'Dave M.', text: 'Spreading awareness. Our democracy is under attack.', time: '12:11 PM · Apr 14, 2024', rt: '31', likes: '89', replies: '5' },
    ],
  },
  pol_fake_002: {
    root: [
      { handle: 'PolicyInsider_DC', name: 'Policy Insider DC', text: 'New analysis from the Brookings Institution finds the proposed tax reform will reduce middle-class income by an average of 12%. Full report linked. A thread on what this means for working families.', time: '9:14 AM · Mar 22, 2024', rt: '6,741', likes: '18.4K', replies: '1,203' },
    ],
    hub: [
      { handle: 'EconAnalyst_Pro', name: 'Dr. Sarah Chen', text: 'This aligns with our own modelling. The distributional effects are significant and the report methodology is sound. Recommending all policy staffers read this.', time: '9:28 AM · Mar 22, 2024', rt: '3,892', likes: '9.7K', replies: '528' },
    ],
    leaf: [
      { handle: 'policy_watcher_dc', name: 'Alex T.', text: 'Bookmarking this. Exactly the kind of analysis we need more of.', time: '9:41 AM · Mar 22, 2024', rt: '94', likes: '281', replies: '12' },
      { handle: 'fiscal_conservative', name: 'Robert K.', text: 'Sharing with my network. The numbers speak for themselves.', time: '9:46 AM · Mar 22, 2024', rt: '67', likes: '193', replies: '8' },
    ],
  },
  gos_fake_001: {
    root: [
      { handle: 'CelebScoopDaily', name: 'Celeb Scoop Daily', text: 'EXCLUSIVE: A-list star secretly married in private Malibu ceremony last weekend! Source close to the couple confirms. More details incoming... 👀💍 #Breaking #Celebrity', time: '3:22 PM · Apr 7, 2024', rt: '8,934', likes: '31.2K', replies: '2,141' },
    ],
    hub: [
      { handle: 'gossip_queen_xo', name: 'Tiffany B.', text: 'OMG OMG OMG I CANNOT!! RT if you\'re screaming 😱😭💕 #ShipName #CelebCouple', time: '3:29 PM · Apr 7, 2024', rt: '4,211', likes: '14.8K', replies: '887' },
      { handle: 'HollywoodBuzz247', name: 'Hollywood Buzz', text: 'Our sources are confirming this! The couple was spotted at a private venue in Malibu last Saturday. Guest list was under 30 people. 🔥', time: '3:34 PM · Apr 7, 2024', rt: '3,104', likes: '9.3K', replies: '642' },
    ],
    leaf: [
      { handle: 'fan_forever_xo', name: 'Luna R.', text: 'I literally cried when I saw this!! So happy for them!! 😭❤️', time: '3:41 PM · Apr 7, 2024', rt: '189', likes: '712', replies: '43' },
      { handle: 'PopCultureJunkie', name: 'Maya S.', text: 'RT @CelebScoopDaily waited SO long for this moment 🥹', time: '3:48 PM · Apr 7, 2024', rt: '74', likes: '298', replies: '16' },
    ],
  },
  gos_fake_002: {
    root: [
      { handle: 'HealthTruthAlert', name: 'Health Truth Alert', text: 'URGENT: Major study finds common household chemical causes neurological damage after 3 years of exposure. The study was suppressed by the FDA. Share before this gets deleted. Your family\'s health depends on it.', time: '8:07 AM · Apr 19, 2024', rt: '12,847', likes: '34.1K', replies: '3,892' },
    ],
    hub: [
      { handle: 'natural_wellness_mama', name: 'Sandra P.', text: 'I\'ve been warning about this for years! This is why we switched to all-natural products. Share with every parent you know. 🌿', time: '8:19 AM · Apr 19, 2024', rt: '5,432', likes: '16.7K', replies: '1,204' },
      { handle: 'AlternativeMedTruth', name: 'Alt Health Truth', text: 'Big Pharma doesn\'t want you to know this. The science is there but it\'s being buried. Protect your family. RT everywhere.', time: '8:24 AM · Apr 19, 2024', rt: '4,201', likes: '11.3K', replies: '978' },
    ],
    leaf: [
      { handle: 'worried_mom_2kids', name: 'Karen L.', text: 'Shared with my mommy group. This is terrifying. 😰', time: '8:33 AM · Apr 19, 2024', rt: '201', likes: '634', replies: '58' },
    ],
  },
  pol_real_001: {
    root: [
      { handle: 'SenateReporter', name: 'Capitol Hill Reporter', text: 'Senate passes bipartisan infrastructure bill 68-32. Key provisions: $550B in new spending on roads, broadband, water systems, and public transit. Full breakdown in thread. #Infrastructure #Congress', time: '2:48 PM · Nov 6, 2023', rt: '3,201', likes: '8.4K', replies: '612' },
    ],
    hub: [
      { handle: 'TransportationDesk', name: 'AP Transportation', text: 'Breaking down what the infrastructure bill means for each state. @SenateReporter thread is comprehensive — adding our own analysis here.', time: '3:05 PM · Nov 6, 2023', rt: '1,892', likes: '5.1K', replies: '287' },
      { handle: 'UrbanPlannerGuild', name: 'Urban Planning Today', text: 'The broadband provisions in this bill are genuinely significant for rural connectivity. Wonky thread on the policy mechanics:', time: '3:21 PM · Nov 6, 2023', rt: '734', likes: '2.2K', replies: '143' },
    ],
    leaf: [
      { handle: 'civicsMindedCitizen', name: 'Tom F.', text: 'Good to see bipartisan agreement on something this important.', time: '3:38 PM · Nov 6, 2023', rt: '29', likes: '87', replies: '4' },
      { handle: 'policy_nerd_dc', name: 'Priya K.', text: 'Archiving this thread for my infrastructure policy class.', time: '3:44 PM · Nov 6, 2023', rt: '12', likes: '41', replies: '2' },
    ],
  },
  pol_real_002: {
    root: [
      { handle: 'ReutersWashington', name: 'Reuters DC', text: 'Federal Reserve holds interest rates steady at 5.25-5.5% for the third consecutive meeting. Chair Powell: "We are committed to returning inflation to our 2% goal." Full statement: [link]', time: '2:01 PM · Jan 31, 2024', rt: '2,108', likes: '5.7K', replies: '441' },
    ],
    hub: [
      { handle: 'EconReporter_NYT', name: 'NYT Economy', text: 'Parsing the Fed statement for signals on when rate cuts might begin. Key language change from December meeting: [thread]', time: '2:14 PM · Jan 31, 2024', rt: '1,203', likes: '3.4K', replies: '198' },
    ],
    leaf: [
      { handle: 'bond_market_watcher', name: 'Chris M.', text: 'As expected. Markets had fully priced this in. Watching for the Q&A.', time: '2:08 PM · Jan 31, 2024', rt: '44', likes: '132', replies: '9' },
    ],
  },
  gos_real_001: {
    root: [
      { handle: 'EntertainmentWeekly', name: 'Entertainment Weekly', text: 'OFFICIAL: [Studio] confirms casting for the upcoming sequel. Director returns along with original cast. Production begins this fall. Full announcement: [link] #Hollywood', time: '10:30 AM · Feb 14, 2024', rt: '4,801', likes: '21.3K', replies: '1,847' },
    ],
    hub: [
      { handle: 'FilmCriticCircle', name: 'Film Critic Circle', text: 'Strong casting choices here. The director\'s previous film grossed $1.2B — expectations will be high. Our analysis of what to expect:', time: '10:47 AM · Feb 14, 2024', rt: '892', likes: '3.1K', replies: '214' },
      { handle: 'BoxOfficeAnalyst', name: 'Box Office Pro', text: 'Early tracking models suggest strong opening weekend potential. Comparable franchise entries in this slot have averaged $180M domestically.', time: '11:02 AM · Feb 14, 2024', rt: '634', likes: '1.9K', replies: '156' },
    ],
    leaf: [
      { handle: 'moviegoer_casual', name: 'Jamie L.', text: 'Already have my tickets. Cannot wait!! 🍿', time: '11:19 AM · Feb 14, 2024', rt: '38', likes: '147', replies: '11' },
      { handle: 'cinephile_2049', name: 'Raj S.', text: 'The casting is inspired. This franchise needed new energy.', time: '11:25 AM · Feb 14, 2024', rt: '21', likes: '78', replies: '6' },
    ],
  },
  gos_real_002: {
    root: [
      { handle: 'THR_Breaking', name: 'The Hollywood Reporter', text: 'Grammy nominations announced. Full list: [link] — surprise inclusions in Album of the Year category. Several independent artists receive first-ever nominations. #Grammys', time: '9:00 AM · Nov 10, 2023', rt: '6,234', likes: '27.8K', replies: '3,102' },
    ],
    hub: [
      { handle: 'MusicIndustryInsider', name: 'Music Industry Insider', text: 'The Recording Academy\'s voting changes from two years ago are clearly showing results. More diverse nominations this cycle than any year since 2008.', time: '9:22 AM · Nov 10, 2023', rt: '1,104', likes: '4.2K', replies: '287' },
    ],
    leaf: [
      { handle: 'music_fan_always', name: 'Nina P.', text: 'So happy for the indie nominations this year! About time 🎵', time: '9:38 AM · Nov 10, 2023', rt: '56', likes: '209', replies: '14' },
      { handle: 'streaming_playlist', name: 'Jason K.', text: 'Building my Grammy nominees playlist right now. Always a great discovery month.', time: '9:44 AM · Nov 10, 2023', rt: '23', likes: '84', replies: '5' },
    ],
  },
};

// Avatar colours for tweet cards (by handle hash)
const AVATAR_COLORS = ['#6366f1','#8b5cf6','#ec4899','#f59e0b','#10b981','#3b82f6','#ef4444','#14b8a6'];
function avatarColor(handle) {
  let h = 0; for (const c of handle) h = (h * 31 + c.charCodeAt(0)) & 0xffff;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}
function initials(name) {
  return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
}
function pickTweet(exId, group, nodeId) {
  const pool = TWEET_TEMPLATES[exId]?.[group] || TWEET_TEMPLATES[exId]?.leaf || [];
  if (!pool.length) return null;
  return pool[nodeId % pool.length];
}

