import os
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi


BEIR_DATASETS = {
    'nfcorpus':   'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip',
    'scifact':    'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
    'trec-covid': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip',
}


def load_beir_dataset(name: str, data_dir: str = 'datasets') -> tuple:
    """Download and load a BEIR dataset. Returns (corpus, queries, qrels)."""
    assert name in BEIR_DATASETS, f"Unknown dataset: {name}. Choose from {list(BEIR_DATASETS)}"
    url = BEIR_DATASETS[name]
    data_path = util.download_and_unzip(url, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split='test')
    print(f"✓ {name}: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


def load_bright_dataset(subset: str) -> tuple:
    """Load a BRIGHT subset from HuggingFace. subset: 'biology' or 'economics'.
    Returns (corpus, queries, qrels) in BEIR format.
    corpus: {doc_id: {'text': content_str}}
    queries: {query_id: query_str}
    qrels: {query_id: {doc_id: 1}}
    """
    from datasets import load_dataset
    assert subset in ('biology', 'economics'), f"Use 'biology' or 'economics', got: {subset}"

    # Load queries and corpus as separate configs
    examples = load_dataset('xlangai/BRIGHT', 'examples')[subset]
    doc_list = load_dataset('xlangai/BRIGHT', 'documents')[subset]

    # Build corpus: {doc_id: {'text': content}}
    corpus = {dp['id']: {'text': dp['content']} for dp in doc_list}

    # Build queries and qrels
    queries, qrels = {}, {}
    for e in examples:
        qid = str(e['id'])
        queries[qid] = e['query']
        # gold_ids are string doc IDs matching corpus keys
        qrels[qid] = {str(did): 1 for did in e['gold_ids']}

    print(f"✓ BRIGHT-{subset}: {len(corpus)} docs, {len(queries)} queries")
    return corpus, queries, qrels


def save_json(obj: dict, path: str) -> None:
    """Save a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)
    print(f"✓ Saved: {path}")


def load_json(path: str) -> dict:
    """Load a dictionary from a JSON file."""
    with open(path) as f:
        return json.load(f)


def build_bm25_index(corpus: dict) -> tuple:
    """Build BM25 index over corpus. Returns (bm25, corpus_ids)."""
    corpus_ids = list(corpus.keys())
    tokenized = [corpus[did]['text'].lower().split() for did in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    print(f"✓ BM25 index built: {len(corpus_ids)} docs")
    return bm25, corpus_ids


def retrieve_bm25_top_k(bm25, corpus_ids: list, queries: dict, k: int = 100) -> dict:
    """Retrieve top-k docs per query. Returns {qid: {did: score}}."""
    import numpy as np
    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        tokenized_q = query_text.lower().split()
        scores = bm25.get_scores(tokenized_q)
        top_k_idx = scores.argsort()[-k:][::-1]
        results[qid] = {corpus_ids[j]: float(scores[j]) for j in top_k_idx}
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries")
    print(f"✓ BM25 retrieval done: {len(results)} queries")
    return results
