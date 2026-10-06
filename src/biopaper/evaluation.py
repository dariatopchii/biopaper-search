"""Check that pooled evaluation is fully labeled before calculating recall."""


def require_complete_pool(selected_queries, pool, judgments, fingerprint):
    current = [r for r in pool if r['corpus_hash'] == fingerprint]
    missing = []
    for query in selected_queries:
        ids = {r['paper_id'] for r in current if r['query_id'] == query['id']}
        if not ids:
            raise ValueError(f"No review pool for {query['id']}. Create the pool first.")
        remaining = ids - judgments.get(query['id'], {}).keys()
        if remaining:
            missing.append(f"{query['id']}: {len(remaining)} unreviewed")
    if missing:
        raise ValueError('Finish relevance review before evaluation: ' + '; '.join(missing))
