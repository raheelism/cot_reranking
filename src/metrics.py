import math
from typing import Dict

def dcg_at_k(relevances: list, k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

def ndcg_at_k(ranked_doc_ids: list, qrel: Dict[str, int], k: int = 10) -> float:
    """NDCG@k for a single query. ranked_doc_ids: ordered list of doc ids."""
    ideal = sorted(qrel.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    predicted_rels = [qrel.get(did, 0) for did in ranked_doc_ids[:k]]
    return dcg_at_k(predicted_rels, k) / idcg

def compute_per_query_ndcg(results: Dict[str, Dict[str, float]],
                            qrels: Dict[str, Dict[str, int]],
                            k: int = 10) -> Dict[str, float]:
    """Compute NDCG@k for every query in results. Returns {qid: ndcg_score}."""
    per_query = {}
    for qid in results:
        if qid not in qrels:
            continue
        ranked = sorted(results[qid], key=results[qid].get, reverse=True)
        per_query[qid] = ndcg_at_k(ranked, qrels[qid], k)
    return per_query

def mean_ndcg(per_query: Dict[str, float]) -> float:
    """Mean NDCG@10 across all queries."""
    vals = list(per_query.values())
    return sum(vals) / len(vals) if vals else 0.0
