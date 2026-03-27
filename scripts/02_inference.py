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
# Priority: RESULTS_DIR env var → Colab Drive → repo/results/
RESULTS_DIR = os.environ.get('RESULTS_DIR', '')
if not RESULTS_DIR:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        RESULTS_DIR = '/content/drive/MyDrive/cot_reranking_results'
    except Exception:
        RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'✓ Results dir: {RESULTS_DIR}')
_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith('.json'))
print(f'  Found {len(_files)} JSON files: {_files if _files else "none"}')

# ── Checkpoint helper ──────────────────────────────────────────────────────────
def is_done(name):
    """Return True if all final output files for this dataset exist."""
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
        # Check for in-progress partial files
        for mode in ('direct', 'reason'):
            p = f'{RESULTS_DIR}/{name}_{mode}_partial.json'
            if os.path.exists(p):
                n = len(load_json(p))
                print(f'  Found partial {mode} checkpoint: {n} queries done')
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

    from src.reranker import score_direct, score_reason
    import torch

    def run_mode(mode):
        """Run inference for one mode with per-query checkpointing every 50 queries."""
        final_path     = f'{RESULTS_DIR}/{name}_{mode}.json'
        partial_scores = f'{RESULTS_DIR}/{name}_{mode}_partial.json'
        partial_cots   = f'{RESULTS_DIR}/{name}_cot_partial.json'

        # Final file already exists — skip inference entirely
        if os.path.exists(final_path):
            print(f'  ⏭ {name} {mode} scores already saved — skipping inference')
            cots = load_json(f'{RESULTS_DIR}/{name}_cot_lengths.json') \
                   if mode == 'reason' and os.path.exists(f'{RESULTS_DIR}/{name}_cot_lengths.json') \
                   else {}
            return load_json(final_path), cots

        # Resume from partial if exists
        scores_so_far = load_json(partial_scores) if os.path.exists(partial_scores) else {}
        cots_so_far   = load_json(partial_cots)   if os.path.exists(partial_cots)   else {}
        done_qids     = set(scores_so_far.keys())

        remaining = {qid: q for qid, q in queries.items() if qid not in done_qids}
        if done_qids:
            print(f'  Resuming {mode}: {len(done_qids)} done, {len(remaining)} remaining')

        print(f'\n=== {name}: {mode}-Point ===')
        for i, (qid, query_text) in enumerate(remaining.items()):
            if qid not in bm25_results:
                continue
            top_docs = list(bm25_results[qid].keys())
            doc_scores, lengths = {}, []

            for did in top_docs:
                passage = corpus.get(did, {}).get('text', '')
                if not passage:
                    doc_scores[did] = 0.0
                    continue
                if mode == 'direct':
                    doc_scores[did] = score_direct(query_text, passage, tokenizer, model)
                else:
                    s, cot_len, _ = score_reason(query_text, passage, tokenizer, model,
                                                 max_cot_tokens=128)
                    doc_scores[did] = s
                    lengths.append(cot_len)

            scores_so_far[qid] = doc_scores
            if mode == 'reason' and lengths:
                cots_so_far[qid] = sum(lengths) / len(lengths)

            done_total = len(done_qids) + i + 1
            total      = len(queries)
            if done_total % 10 == 0:
                print(f'  [{mode}] {done_total}/{total} queries done')

            # Save checkpoint every 50 queries
            if (i + 1) % 50 == 0:
                save_json(scores_so_far, partial_scores)
                if mode == 'reason':
                    save_json(cots_so_far, partial_cots)
                print(f'  ✓ Checkpoint saved at {done_total}/{total}')

        # Final save — remove partial files
        save_json(scores_so_far, f'{RESULTS_DIR}/{name}_{mode}.json')
        if mode == 'reason':
            save_json(cots_so_far, f'{RESULTS_DIR}/{name}_cot_lengths.json')
        for f in [partial_scores, partial_cots]:
            if os.path.exists(f): os.remove(f)
        return scores_so_far, cots_so_far

    direct_results, _ = run_mode('direct')
    reason_results, cot_lengths = run_mode('reason')

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
