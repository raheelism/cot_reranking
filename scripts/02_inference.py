"""
Script 2: Direct-Point + Reason-Point inference on all 5 datasets
Run in Colab with: %run scripts/02_inference.py
~3-4 hours on T4 GPU — requires GPU runtime
"""
import os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

assert torch.cuda.is_available(), 'GPU required. Runtime > Change runtime type > T4 GPU'
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── Drive mount ────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

RESULTS_DIR = '/content/drive/MyDrive/cot_reranking_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'✓ Results dir: {RESULTS_DIR}')

# ── Load model ─────────────────────────────────────────────────────────────────
from src.reranker import load_model, score_direct, score_reason, rerank_dataset
tokenizer, model = load_model()

# ── Sanity check ───────────────────────────────────────────────────────────────
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

# ── Inference loop ─────────────────────────────────────────────────────────────
summary = {}
for name in DATASETS:
    corpus       = all_data[name]['corpus']
    queries      = all_data[name]['queries']
    qrels        = all_data[name]['qrels']
    bm25_results = load_json(f'{RESULTS_DIR}/{name}_bm25.json')

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
print('\n✓ All inference complete and saved to Drive')
