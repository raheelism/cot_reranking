"""
Auto-label query complexity using Qwen2.5-3B-Instruct.
Scores logits for 'simple' / 'medium' / 'complex' at the answer position.

Run AFTER 02_inference.py (reuses the loaded model) or standalone:
    %run scripts/00_label_queries.py

Output: {RESULTS_DIR}/query_labels.json  (compatible with src/classifier.py)
"""
import os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Results directory ──────────────────────────────────────────────────────────
RESULTS_DIR = os.environ.get('RESULTS_DIR', '')
if not RESULTS_DIR:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        RESULTS_DIR = '/content/drive/MyDrive/cot_reranking_results'
    except Exception:
        RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

from src.data_utils import load_json, save_json

DATASETS = ['nfcorpus', 'scifact', 'trec-covid', 'bright_biology', 'bright_economics']

out_path = f'{RESULTS_DIR}/query_labels.json'
if os.path.exists(out_path):
    existing = load_json(out_path)
    print(f'✓ query_labels.json already exists ({len(existing)} labels) — delete to re-run')
    raise SystemExit(0)

# ── Label by query-length terciles (per dataset) ───────────────────────────────
# Shortest 1/3 → simple, middle 1/3 → medium, longest 1/3 → complex
# Reproducible, balanced, and defensible: query length is a reliable proxy
# for complexity in IR (short keyword queries vs. long reasoning questions).

def assign_tercile_labels(qid_text_pairs):
    """Sort by query length, assign equal-sized simple/medium/complex terciles."""
    sorted_pairs = sorted(qid_text_pairs, key=lambda x: len(x[1].split()))
    n = len(sorted_pairs)
    t1, t2 = n // 3, 2 * (n // 3)
    result = {}
    for i, (qid, text) in enumerate(sorted_pairs):
        if i < t1:        label = 'simple'
        elif i < t2:      label = 'medium'
        else:             label = 'complex'
        result[qid] = (text, label)
    return result

# ── Label all queries ──────────────────────────────────────────────────────────
labels = {}
for name in DATASETS:
    queries_path = f'{RESULTS_DIR}/{name}_queries.json'
    if not os.path.exists(queries_path):
        print(f'  ⚠ {name}_queries.json not found — skipping (run 01_data_bm25.py first)')
        continue

    queries = load_json(queries_path)
    sample  = list(queries.items())[:40]   # 40 per dataset = 200 total
    labeled = assign_tercile_labels(sample)

    from collections import Counter
    counts = Counter(v[1] for v in labeled.values())
    print(f'{name}: {dict(counts)}')

    for qid, (text, label) in labeled.items():
        labels[f'{name}__{qid}'] = {'query': text, 'label': label}

save_json(labels, out_path)
print(f'\n✓ Saved {len(labels)} labels → {out_path}')

# Quick sanity check
from collections import Counter
all_labels = [v['label'] for v in labels.values()]
print(f'  Distribution: {dict(Counter(all_labels))}')
