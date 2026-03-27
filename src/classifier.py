import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

ENCODER_MODEL = 'BAAI/bge-small-en-v1.5'
LABEL_MAP = {'simple': 0, 'medium': 1, 'complex': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_labels(path: str = 'data/query_labels.json') -> tuple:
    """Load labeled queries. Returns (queries, labels) lists."""
    with open(path) as f:
        data = json.load(f)
    queries = [v['query'] for v in data.values() if v['label'] is not None]
    labels  = [v['label']  for v in data.values() if v['label'] is not None]
    assert len(queries) >= 100, f"Need ≥100 labels, got {len(queries)}"
    print(f"✓ Loaded {len(queries)} labeled queries")
    from collections import Counter
    print(f"  Distribution: {dict(Counter(labels))}")
    return queries, labels

def train_classifier(queries: list, labels: list) -> tuple:
    """Embed queries with bge-small, train logistic regression. Returns (encoder, clf)."""
    encoder = SentenceTransformer(ENCODER_MODEL)
    X = encoder.encode(queries, batch_size=32, show_progress_bar=True)
    y = np.array([LABEL_MAP[l] for l in labels])
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"✓ Classifier trained — train accuracy: {train_acc:.3f}")
    return encoder, clf

def predict_complexity(queries_dict: dict, encoder, clf) -> dict:
    """Predict complexity for all queries. Returns {qid: label_str}."""
    qids  = list(queries_dict.keys())
    texts = [queries_dict[qid] for qid in qids]
    X     = encoder.encode(texts, batch_size=32, show_progress_bar=False)
    preds = clf.predict(X)
    return {qid: INV_LABEL_MAP[p] for qid, p in zip(qids, preds)}
