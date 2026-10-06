import math


def ranking_metrics(ranked_ids, judgments, k=5):
    """Unjudged documents are an error, never implicitly irrelevant.

    Recall denominator includes only relevant documents in the judged pool.
    This is pooled recall, not a claim of exhaustive corpus recall.
    """
    if k < 1:
        raise ValueError("k must be positive")
    top = list(ranked_ids[:k])
    if len(top) != len(set(top)):
        raise ValueError("Duplicate results would inflate metrics")
    if any(i not in judgments for i in top):
        raise ValueError("Complete relevance judgments for every retrieved document")
    if any(isinstance(g, bool) or not isinstance(g, int) or g not in (0, 1, 2) for g in judgments.values()):
        raise ValueError("Relevance grades must be integers 0, 1, or 2")
    relevant = sum(g > 0 for g in judgments.values())
    if relevant == 0:
        raise ValueError("A query needs at least one relevant judged document")
    dcg = sum((2 ** judgments[i] - 1) / math.log2(n + 2) for n, i in enumerate(top))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(n + 2) for n, g in enumerate(ideal))
    return {"pooled_recall": sum(judgments[i] > 0 for i in top) / relevant, "ndcg": dcg / idcg}
