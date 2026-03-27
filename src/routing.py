from typing import Dict


def selective_route(direct_results: Dict[str, Dict[str, float]],
                    reason_results: Dict[str, Dict[str, float]],
                    complexity: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Route each query to Direct or Reason based on complexity.
    simple + medium → Direct-Point; complex → Reason-Point.
    Returns merged results dict in same format as input.
    """
    if set(direct_results) != set(reason_results):
        missing = set(direct_results).symmetric_difference(set(reason_results))
        print(f"Warning: {len(missing)} qids present in only one result dict")

    routed = {}
    route_counts = {'direct': 0, 'reason': 0}

    for qid in set(direct_results) | set(reason_results):
        label = complexity.get(qid, 'medium')
        if label in ('simple', 'medium'):
            routed[qid] = direct_results.get(qid, {})
            route_counts['direct'] += 1
        else:
            routed[qid] = reason_results.get(qid, {})
            route_counts['reason'] += 1

    total = sum(route_counts.values())
    print(f"Routing: {route_counts['direct']}/{total} → Direct, "
          f"{route_counts['reason']}/{total} → Reason")
    return routed
