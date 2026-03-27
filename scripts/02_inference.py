"""
Script 2: Direct-Point + Reason-Point inference on all 5 datasets
Run in Colab/Kaggle with: %run scripts/02_inference.py
~3-4 hours on T4 GPU — requires GPU runtime

RESUME SAFE: automatically skips datasets already completed.
Results are loaded from Drive for skipped datasets so summary stays complete.

Platforms:
  Colab:  RESULTS_DIR auto-set to Google Drive
  Kaggle: set RESULTS_DIR env var or edit below, sync Drive via rclone
"""
import os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

assert torch.cuda.is_available(), 'GPU required. Runtime > Change runtime type > T4 GPU'
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── Results directory ──────────────────────────────────────────────────────────
# Priority: repo/results/ if it has JSON files → Colab Drive → repo/results/ empty
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

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'✓ Results dir: {RESULTS_DIR}')
_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith('.json'))
print(f'  Found {len(_files)} JSON files: {_files if _files else "none"}')

# ── Checkpoint helper ──────────────────────────────────────────────────────────
def is_done(name):
    """Return True if all output files for this dataset already exist."""
    required = [
        f'{RESULTS_DIR}/{name}_direct.json',
        f'{RESULTS_DIR}/{name}_reason.json',
        f'{RESULTS_DIR}/{name}_cot_lengths.json',
        f'{RESULTS_DIR}/{name}_pq_direct.json',
        f'{RESULTS_DIR}/{name}_pq_reason.json',
        f'{RESULTS_DIR}/{name}_pq_bm25.json',
    ]
    missing = [os.path.basename(f) for f in required if not os.path.exists(f)]
    if missing:
        print(f'  Missing for {name}: {missing}')
    return len(missing) == 0

# ── Load model ─────────────────────────────────────────────────────────────────
from src.reranker import load_model, score_direct, rerank_dataset
tokenizer, model = load_model()

# Sanity check
s_rel = score_direct('where do elephants live',
    'Elephants inhabit savannas, forests, and deserts across Africa and South Asia.',
    tokenizer, model)
s_irr = score_direct('where do elephants live',
    'The stock market closed higher after positive earnings reports.',
    tokenizer, model)
print(f'Sanity check — Relevant: {s_rel:.4f}, Irrelevant: {s_irr:.4f}')
assert s_rel > s_irr, 'Direct-Point sanity check failed'
print('✓ Model OK\n')

# ── Load datasets ──────────────────────────────────────────────────────────────
from src.data_utils import load_beir_dataset, load_bright_dataset, load_json, save_json
from src.metrics import compute_per_query_ndcg, mean_ndcg

DATASETS = ['nfcorpus', 'scifact', 'trec-covid', 'bright_biology', 'bright_economics']

all_data = {}
for name in ['nfcorpus', 'scifact', 'trec-covid']:
    corpus, queries, qrels = load_beir_dataset(name)
    all_data[name] = {'corpus': corpus, 'queries': queries, 'qrels': qrels}
for subset in ['biology', 'economics']:
    corpus, queries, qrels = load_bright_dataset(subset)
    all_data[f'bright_{subset}'] = {'corpus': corpus, 'queries': queries, 'qrels': qrels}

# ── Inference loop with resume ─────────────────────────────────────────────────
summary = {}
for name in DATASETS:
    if is_done(name):
        pq_direct = load_json(f'{RESULTS_DIR}/{name}_pq_direct.json')
        pq_reason = load_json(f'{RESULTS_DIR}/{name}_pq_reason.json')
        pq_bm25   = load_json(f'{RESULTS_DIR}/{name}_pq_bm25.json')
        summary[name] = {
            'bm25':   mean_ndcg(pq_bm25),
            'direct': mean_ndcg(pq_direct),
            'reason': mean_ndcg(pq_reason),
        }
        d, r = summary[name]['direct'], summary[name]['reason']
        print(f'⏭  {name} — already done (Direct={d:.4f}, Reason={r:.4f}) — skipping')
        continue

    corpus  = all_data[name]['corpus']
    queries = all_data[name]['queries']
    qrels   = all_data[name]['qrels']

    bm25_path = f'{RESULTS_DIR}/{name}_bm25.json'
    if not os.path.exists(bm25_path):
        print(f'  BM25 not found for {name} — generating now (CPU, ~5 min)...')
        from src.data_utils import build_bm25_index, retrieve_bm25_top_k
        bm25_idx, corpus_ids = build_bm25_index(corpus)
        bm25_results = retrieve_bm25_top_k(bm25_idx, corpus_ids, queries, k=100)
        save_json(bm25_results, bm25_path)
        save_json(qrels,   f'{RESULTS_DIR}/{name}_qrels.json')
        save_json(queries, f'{RESULTS_DIR}/{name}_queries.json')
    else:
        bm25_results = load_json(bm25_path)

    print(f'\n=== {name}: Direct-Point ===')
    direct_results, _ = rerank_dataset(corpus, queries, bm25_results, tokenizer, model, mode='direct')
    save_json(direct_results, f'{RESULTS_DIR}/{name}_direct.json')

    print(f'\n=== {name}: Reason-Point ===')
    reason_results, cot_lengths = rerank_dataset(corpus, queries, bm25_results, tokenizer, model, mode='reason')
    save_json(reason_results,  f'{RESULTS_DIR}/{name}_reason.json')
    save_json(cot_lengths,     f'{RESULTS_DIR}/{name}_cot_lengths.json')

    pq_direct = compute_per_query_ndcg(direct_results, qrels)
    pq_reason = compute_per_query_ndcg(reason_results, qrels)
    pq_bm25   = compute_per_query_ndcg(bm25_results,   qrels)

    save_json(pq_direct, f'{RESULTS_DIR}/{name}_pq_direct.json')
    save_json(pq_reason, f'{RESULTS_DIR}/{name}_pq_reason.json')
    save_json(pq_bm25,   f'{RESULTS_DIR}/{name}_pq_bm25.json')

    summary[name] = {
        'bm25':   mean_ndcg(pq_bm25),
        'direct': mean_ndcg(pq_direct),
        'reason': mean_ndcg(pq_reason),
    }
    d, r = summary[name]['direct'], summary[name]['reason']
    print(f'  BM25={summary[name]["bm25"]:.4f} | Direct={d:.4f} | Reason={r:.4f} | Δ={d - r:+.4f}')

save_json(summary, f'{RESULTS_DIR}/summary_ndcg.json')
print('\n✓ All inference complete and saved')
