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

# ── Load model ─────────────────────────────────────────────────────────────────
from src.reranker import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse already-loaded model if available in session, else load fresh
try:
    tokenizer  # noqa: F821 — defined in prior %run
    model      # noqa: F821
    print('✓ Reusing model already in session')
except NameError:
    tokenizer, model = load_model()

# ── Labeling function ──────────────────────────────────────────────────────────
SYSTEM = (
    "You are a query complexity classifier for information retrieval. "
    "Classify the query into exactly one category:\n"
    "  simple  — factual lookup, single concept, short answer\n"
    "  medium  — requires background knowledge or multi-step lookup\n"
    "  complex — requires multi-hop reasoning, domain expertise, or abstract analysis"
)

def label_query(query_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Query: {query_text}\n\nAnswer with one word only (simple / medium / complex):"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits[0, -1, :]  # last token position

    # Get token IDs for each class (bare + space-prefixed)
    def tok(word):
        return [tokenizer.encode(v, add_special_tokens=False)[-1]
                for v in (word, ' ' + word, word.capitalize(), ' ' + word.capitalize())]

    simple_ids  = tok('simple')
    medium_ids  = tok('medium')
    complex_ids = tok('complex')

    score_simple  = logits[simple_ids].max().item()
    score_medium  = logits[medium_ids].max().item()
    score_complex = logits[complex_ids].max().item()

    del inputs, out
    torch.cuda.empty_cache()

    best = max(score_simple, score_medium, score_complex)
    if best == score_simple:  return 'simple'
    if best == score_medium:  return 'medium'
    return 'complex'

# ── Label all queries ──────────────────────────────────────────────────────────
labels = {}
for name in DATASETS:
    queries_path = f'{RESULTS_DIR}/{name}_queries.json'
    if not os.path.exists(queries_path):
        print(f'  ⚠ {name}_queries.json not found — skipping (run 01_data_bm25.py first)')
        continue

    queries = load_json(queries_path)
    sample  = dict(list(queries.items())[:40])   # 40 per dataset = 200 total
    counts  = {'simple': 0, 'medium': 0, 'complex': 0}

    print(f'\n── {name} ({len(sample)} queries) ──')
    for i, (qid, query_text) in enumerate(sample.items()):
        label = label_query(query_text)
        labels[f'{name}__{qid}'] = {'query': query_text, 'label': label}
        counts[label] += 1
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(sample)}  so far: {counts}')

    print(f'  Done: {counts}')

save_json(labels, out_path)
print(f'\n✓ Saved {len(labels)} labels → {out_path}')

# Quick sanity check
from collections import Counter
all_labels = [v['label'] for v in labels.values()]
print(f'  Distribution: {dict(Counter(all_labels))}')
