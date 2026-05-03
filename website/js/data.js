/* =========================================================
   data.js  –  Static demo constants
   Live graph data is fetched from the Flask backend (/api/sample);
   this file only carries the small lookup tables the frontend needs
   to label / colour what comes back.
   ========================================================= */

// Aggregate summary stats from results/upfd_multiseed_summary.csv (5 seeds).
// Used by the static Ablation chart.
// NOTE: numbers below are placeholders — overwrite from results/upfd_multiseed_summary.csv
// after the fresh training run completes.
const MODEL_SUMMARY = [
  { model: 'RF_motif',       auc: 0.795, auc_std: 0.011, f1: 0.733, bal_acc: 0.731 },
  { model: 'TextOnly_BERT',  auc: 0.973, auc_std: 0.001, f1: 0.936, bal_acc: 0.935 },
  { model: 'GraphOnly_GAT',  auc: 0.725, auc_std: 0.146, f1: 0.184, bal_acc: 0.508 },
  { model: 'PropNet_Fusion', auc: 0.989, auc_std: 0.004, f1: 0.955, bal_acc: 0.955 },
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
