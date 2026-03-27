"""
Script 1: Data loading + BM25 retrieval
Run in Colab with: %run scripts/01_data_bm25.py
~2 hours on CPU
"""
import os, sys

# Project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'✓ Results dir: {RESULTS_DIR}')

# ── BEIR datasets ──────────────────────────────────────────────────────────────
from src.data_utils import load_beir_dataset, load_bright_dataset
from src.data_utils import build_bm25_index, retrieve_bm25_top_k, save_json
from beir.retrieval.evaluation import EvaluateRetrieval

all_data = {}

for name in ['nfcorpus', 'scifact', 'trec-covid']:
    corpus, queries, qrels = load_beir_dataset(name)
    bm25, corpus_ids = build_bm25_index(corpus)
    results = retrieve_bm25_top_k(bm25, corpus_ids, queries, k=100)

    ndcg, *_ = EvaluateRetrieval.evaluate(qrels, results, [10])
    print(f"  BM25 NDCG@10: {ndcg['NDCG@10']:.4f}")
    assert ndcg['NDCG@10'] > 0.10, f'BM25 baseline too low on {name}'

    all_data[name] = {'corpus': corpus, 'queries': queries, 'qrels': qrels}
    save_json(results,  f'{RESULTS_DIR}/{name}_bm25.json')
    save_json(qrels,    f'{RESULTS_DIR}/{name}_qrels.json')
    save_json(queries,  f'{RESULTS_DIR}/{name}_queries.json')
    print(f'✓ {name} done\n')

# ── BRIGHT datasets ────────────────────────────────────────────────────────────
for subset in ['biology', 'economics']:
    name = f'bright_{subset}'
    corpus, queries, qrels = load_bright_dataset(subset)
    bm25, corpus_ids = build_bm25_index(corpus)
    results = retrieve_bm25_top_k(bm25, corpus_ids, queries, k=100)

    all_data[name] = {'corpus': corpus, 'queries': queries, 'qrels': qrels}
    save_json(results,  f'{RESULTS_DIR}/{name}_bm25.json')
    save_json(qrels,    f'{RESULTS_DIR}/{name}_qrels.json')
    save_json(queries,  f'{RESULTS_DIR}/{name}_queries.json')
    print(f'✓ {name} done\n')

print('✓ All BM25 results saved to Drive')
