/* =========================================================
   data.js  –  Static demo constants
   Live graph data is fetched from the Flask backend (/api/sample);
   this file only carries the small lookup tables the frontend needs
   to label / colour what comes back.
   ========================================================= */

// Aggregate summary stats from results/upfd_multiseed_summary.csv (5 seeds).
// Used by the static Ablation chart.
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

// Features where HIGH value → indicative of FAKE (used for bar coloring).
const FAKE_INDICATIVE = new Set([
  'triangle_count', 'max_star_size', 'avg_clustering',
  'density', 'root_out_degree_frac',
]);
