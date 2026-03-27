"""
Script 3: Analysis, figures, and selective routing
Run in Colab with: %run scripts/03_analysis_routing.py
~30 min, CPU-only (no GPU needed)

Requires:
  - scripts/02_inference.py to have completed (results on Drive)
  - query_labels.json filled in (see instructions printed below)
"""
import os, sys, json
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── Results directory ──────────────────────────────────────────────────────────
_repo_results = os.path.join(PROJECT_ROOT, 'results')
_has_data = os.path.isdir(_repo_results) and any(
    f.endswith('.json') for f in os.listdir(_repo_results)
)

if _has_data:
    RESULTS_DIR = _repo_results
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        RESULTS_DIR = '/content/drive/MyDrive/cot_reranking_results'
    except Exception:
        RESULTS_DIR = _repo_results

FIGURES_DIR = f'{RESULTS_DIR}/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

from src.data_utils import load_json, save_json

DATASETS = ['nfcorpus', 'scifact', 'trec-covid', 'bright_biology', 'bright_economics']

# ── Main results table ─────────────────────────────────────────────────────────
summary = load_json(f'{RESULTS_DIR}/summary_ndcg.json')
print(f"{'Dataset':<22} {'BM25':>8} {'Direct':>8} {'Reason':>8} {'Δ(D-R)':>8}")
print('-' * 58)
for name, vals in summary.items():
    delta = vals['direct'] - vals['reason']
    print(f"{name:<22} {vals['bm25']:>8.4f} {vals['direct']:>8.4f} {vals['reason']:>8.4f} {delta:>+8.4f}")

# ── Query labels ───────────────────────────────────────────────────────────────
labels_path = os.path.join(PROJECT_ROOT, 'data', 'query_labels.json')
if not os.path.exists(labels_path):
    labels_path = f'{RESULTS_DIR}/query_labels.json'

if not os.path.exists(labels_path):
    # Generate template from actual query IDs
    template = {}
    for name in DATASETS:
        queries = load_json(f'{RESULTS_DIR}/{name}_queries.json')
        for qid in list(queries.keys())[:40]:
            template[f'{name}__{qid}'] = {'query': queries[qid], 'label': None}

    template_out = f'{RESULTS_DIR}/query_labels_template.json'
    with open(template_out, 'w') as f:
        json.dump(template, f, indent=2)

    print(f'\n✓ Template saved: {len(template)} queries → {template_out}')
    print('\nACTION REQUIRED:')
    print('  1. Download query_labels_template.json from Drive')
    print('  2. Fill each "label" field: "simple", "medium", or "complex"')
    print('  3. Save as query_labels.json in the same Drive folder')
    print('  4. Re-run this script')
    raise SystemExit('Labels not found — complete labeling and re-run')

# ── Train classifier ───────────────────────────────────────────────────────────
from src.classifier import load_labels, train_classifier, predict_complexity

queries_labeled, labels = load_labels(labels_path)
encoder, clf = train_classifier(queries_labeled, labels)

all_complexity = {}
for name in DATASETS:
    queries = load_json(f'{RESULTS_DIR}/{name}_queries.json')
    all_complexity[name] = predict_complexity(queries, encoder, clf)
    print(f'{name}: {dict(Counter(all_complexity[name].values()))}')

save_json(all_complexity, f'{RESULTS_DIR}/all_complexity.json')

# ── Experiment 2: Complexity stratification ────────────────────────────────────
complexity_ndcg = {'direct': defaultdict(list), 'reason': defaultdict(list)}

for name in DATASETS:
    pq_direct  = load_json(f'{RESULTS_DIR}/{name}_pq_direct.json')
    pq_reason  = load_json(f'{RESULTS_DIR}/{name}_pq_reason.json')
    complexity = all_complexity[name]
    for qid, label in complexity.items():
        if qid in pq_direct: complexity_ndcg['direct'][label].append(pq_direct[qid])
        if qid in pq_reason: complexity_ndcg['reason'][label].append(pq_reason[qid])

print('\nNDCG@10 by complexity:')
for label in ['simple', 'medium', 'complex']:
    d = np.mean(complexity_ndcg['direct'][label])
    r = np.mean(complexity_ndcg['reason'][label])
    _, p = scipy_stats.mannwhitneyu(
        complexity_ndcg['direct'][label],
        complexity_ndcg['reason'][label],
        alternative='greater'
    )
    print(f'  {label:8s}: Direct={d:.4f}  Reason={r:.4f}  Δ={d - r:+.4f}  p={p:.4f}')

labels_plot = ['Simple', 'Medium', 'Complex']
direct_avg  = [np.mean(complexity_ndcg['direct'][l.lower()]) for l in labels_plot]
reason_avg  = [np.mean(complexity_ndcg['reason'][l.lower()]) for l in labels_plot]
x, width = np.arange(len(labels_plot)), 0.35

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width/2, direct_avg, width, label='Direct-Point', color='#2196F3', alpha=0.85)
ax.bar(x + width/2, reason_avg, width, label='Reason-Point', color='#FF5722', alpha=0.85)
ax.set_xlabel('Query Complexity'); ax.set_ylabel('NDCG@10')
ax.set_title('NDCG@10 by Query Complexity: Direct vs. Reason')
ax.set_xticks(x); ax.set_xticklabels(labels_plot); ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig1_complexity_bars.pdf', bbox_inches='tight')
plt.savefig(f'{FIGURES_DIR}/fig1_complexity_bars.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ Figure 1 saved')

# ── Experiment 3: CoT length vs NDCG correlation ───────────────────────────────
all_lengths, all_ndcg_r = [], []
for name in DATASETS:
    cot_lengths = load_json(f'{RESULTS_DIR}/{name}_cot_lengths.json')
    pq_reason   = load_json(f'{RESULTS_DIR}/{name}_pq_reason.json')
    for qid in cot_lengths:
        if qid in pq_reason:
            all_lengths.append(cot_lengths[qid])
            all_ndcg_r.append(pq_reason[qid])

pearson_r,  pearson_p  = scipy_stats.pearsonr(all_lengths, all_ndcg_r)
spearman_r, spearman_p = scipy_stats.spearmanr(all_lengths, all_ndcg_r)
print(f'\nPearson  r={pearson_r:.4f} (p={pearson_p:.4f})')
print(f'Spearman ρ={spearman_r:.4f} (p={spearman_p:.4f})')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.scatter(all_lengths, all_ndcg_r, alpha=0.3, s=12, color='#FF5722')
m, b = np.polyfit(all_lengths, all_ndcg_r, 1)
x_line = np.linspace(min(all_lengths), max(all_lengths), 100)
ax.plot(x_line, m * x_line + b, 'k--', linewidth=1.5, label=f'r={pearson_r:.3f}')
ax.set_xlabel('Avg CoT Length (tokens)'); ax.set_ylabel('Per-Query NDCG@10')
ax.set_title('CoT Length vs. Ranking Quality'); ax.legend()

binned = [[], [], []]
for l, n in zip(all_lengths, all_ndcg_r):
    binned[0 if l < 50 else (1 if l < 150 else 2)].append(n)
ax2 = axes[1]
bp = ax2.boxplot(binned, labels=['Short\n(<50)', 'Medium\n(50-150)', 'Long\n(>150)'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#4CAF50', '#FFC107', '#F44336']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax2.set_xlabel('CoT Length Bin'); ax2.set_ylabel('Per-Query NDCG@10')
ax2.set_title('NDCG@10 by CoT Length Bin')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig2_length_scatter.pdf', bbox_inches='tight')
plt.savefig(f'{FIGURES_DIR}/fig2_length_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ Figure 2 saved')

# ── Experiment 4: Selective routing ───────────────────────────────────────────
from src.routing import selective_route
from src.metrics import compute_per_query_ndcg, mean_ndcg

routing_summary = {}
for name in DATASETS:
    qrels      = load_json(f'{RESULTS_DIR}/{name}_qrels.json')
    direct     = load_json(f'{RESULTS_DIR}/{name}_direct.json')
    reason     = load_json(f'{RESULTS_DIR}/{name}_reason.json')
    complexity = all_complexity[name]

    routed    = selective_route(direct, reason, complexity)
    pq_routed = compute_per_query_ndcg(routed, qrels)

    routing_summary[name] = {
        'direct': summary[name]['direct'],
        'reason': summary[name]['reason'],
        'routed': mean_ndcg(pq_routed),
    }
    d, r, ro = routing_summary[name]['direct'], routing_summary[name]['reason'], routing_summary[name]['routed']
    print(f'{name:<22} Direct={d:.4f}  Reason={r:.4f}  Routed={ro:.4f}  Δ={ro - d:+.4f}')

save_json(routing_summary, f'{RESULTS_DIR}/routing_summary.json')
print('✓ Routing results saved')

# ── LaTeX tables ───────────────────────────────────────────────────────────────
display_names = {
    'nfcorpus':         'NFCorpus',
    'scifact':          'SciFact',
    'trec-covid':       'TREC-COVID',
    'bright_biology':   'BRIGHT-Bio',
    'bright_economics': 'BRIGHT-Econ',
}

print('\n% Table 1: Main Results (NDCG@10)')
print('\\begin{tabular}{lccc}\\toprule')
print('Dataset & BM25 & Direct-Point & Reason-Point \\\\\\midrule')
for name, vals in summary.items():
    d, r = vals['direct'], vals['reason']
    bd = f'\\textbf{{{d:.4f}}}' if d >= r else f'{d:.4f}'
    br = f'\\textbf{{{r:.4f}}}' if r > d  else f'{r:.4f}'
    print(f"  {display_names[name]} & {vals['bm25']:.4f} & {bd} & {br} \\\\")
print('\\bottomrule\\end{tabular}\n')

print('% Table 2: Selective Routing Results (NDCG@10)')
print('\\begin{tabular}{lccc}\\toprule')
print('Dataset & Direct-Point & Reason-Point & Selective-Router \\\\\\midrule')
for name, vals in routing_summary.items():
    best = max(vals.values())
    def fmt(v): return f'\\textbf{{{v:.4f}}}' if abs(v - best) < 1e-6 else f'{v:.4f}'
    print(f'  {display_names[name]} & {fmt(vals["direct"])} & {fmt(vals["reason"])} & {fmt(vals["routed"])} \\\\')
print('\\bottomrule\\end{tabular}')
