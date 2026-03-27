import os
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader


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
    """Load a BRIGHT subset from HuggingFace. subset: 'biology' or 'economics'."""
    from datasets import load_dataset
    assert subset in ('biology', 'economics'), f"Use 'biology' or 'economics', got: {subset}"

    ds = load_dataset('xlangai/BRIGHT', subset, split='test')

    # Build corpus, queries, qrels in BEIR format
    corpus, queries, qrels = {}, {}, {}
    for row in ds:
        qid = str(row['id'])
        queries[qid] = row['query']
        qrels[qid] = {}
        for doc in row['gold_ids']:
            did = str(doc)
            corpus[did] = {'text': row['documents'][row['gold_ids'].index(doc)]}
            qrels[qid][did] = 1
        # Add negative docs
        for i, doc_text in enumerate(row['documents']):
            did = f"{qid}_doc_{i}"
            if did not in corpus:
                corpus[did] = {'text': doc_text}

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
