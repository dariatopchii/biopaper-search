"""Query-disjoint learning curves; refuses partial labels and unseen top results."""
import argparse
import json
import random
import statistics
from biopaper.data import ROOT, load_jsonl, load_papers, write_jsonl
from biopaper.evaluation import require_complete_pool
from biopaper.metrics import ranking_metrics
from biopaper.preferences import fit_ranker, predict_scores, annotation_snapshot
from biopaper.retrieval import SearchEngine


def experiment(engine, final_test=False):
    queries = json.loads((ROOT / 'evaluation/queries.json').read_text(encoding='utf-8'))
    dev = [q for q in queries if q['split'] == 'dev']
    # Fixed before viewing labels: first 15 development queries train; last 5 validate.
    training, validation = dev[:15], dev[15:]
    target = [q for q in queries if q['split'] == 'test'] if final_test else validation
    if len(training) < 15 or len(validation) < 5 or not target:
        raise ValueError('This protocol requires 20 development queries and a held-out test set')
    judgments = {}
    labels = load_jsonl(ROOT / 'evaluation/judgments.jsonl')
    for row in labels:
        if row['corpus_hash'] == engine.fingerprint:
            judgments.setdefault(row['query_id'], {})[row['paper_id']] = row['grade']
    require_complete_pool(training + target, load_jsonl(ROOT / 'evaluation/pool.jsonl'), judgments, engine.fingerprint)
    features = {q['id']: engine.preference_features(q['text']) for q in training + target}
    index = {p['id']: i for i, p in enumerate(engine.papers)}
    groups = []
    for q in training:
        grades = judgments[q['id']]
        ids = sorted(grades)
        groups.append({'query_id': q['id'], 'features': features[q['id']][[index[i] for i in ids]], 'grades': [grades[i] for i in ids]})
    curves, per_query, unseen = [], [], set()
    # Final test uses all training queries once; learning curves are validation-only.
    for seed in ([0] if final_test else [0, 1, 2]):
        order = groups.copy()
        random.Random(seed).shuffle(order)
        for count in ([15] if final_test else [2, 5, 10, 15]):
            model = fit_ranker(order[:count])
            scores = []
            for q in target:
                values = predict_scores(model, features[q['id']])
                ranked = sorted(range(len(values)), key=lambda i: (-values[i], engine.papers[i]['id']))[:5]
                ids = [engine.papers[i]['id'] for i in ranked]
                missing = [i for i in ids if i not in judgments[q['id']]]
                if missing:
                    unseen.update((q['id'], i) for i in missing)
                    continue
                metric = ranking_metrics(ids, judgments[q['id']], 5)
                scores.append(metric['ndcg'])
                per_query.append({'seed': seed, 'training_queries': count, 'query_id': q['id'], 'method': 'learned_pairwise', **metric})
            if len(scores) == len(target):
                curves.append({'seed': seed, 'training_queries': count, 'induced_pairs': model['pair_count'], 'mean_ndcg_at_5': statistics.mean(scores)})
    if unseen:
        pool_path = ROOT / 'evaluation/pool.jsonl'
        pool = load_jsonl(pool_path)
        existing = {(r['query_id'], r['paper_id'], r['corpus_hash']) for r in pool}
        pool.extend({'query_id': q, 'paper_id': p, 'corpus_hash': engine.fingerprint}
                    for q, p in sorted(unseen) if (q, p, engine.fingerprint) not in existing)
        write_jsonl(pool_path, pool)
        raise ValueError(f'Added {len(unseen)} previously unjudged results to Relevance review. Label them, then rerun. No experiment report was saved.')
    baseline = []
    for q in target:
        for method in ['bm25', 'dense', 'rerank']:
            results, _ = engine.search(q['text'], method, 5, 30)
            baseline.append({'query_id': q['id'], 'method': method, **ranking_metrics([p['id'] for p in results], judgments[q['id']], 5)})
    report = {'corpus_hash': engine.fingerprint, 'annotation_snapshot': annotation_snapshot(queries, labels, engine.fingerprint), 'split': 'test' if final_test else 'validation',
              'training_query_ids': [q['id'] for q in training], 'evaluation_query_ids': [q['id'] for q in target],
              'curves': curves, 'per_query': per_query, 'baselines': baseline,
              'limitations': 'Pilot: 15 train / 5 validation / 10 test queries. Induced preferences, one annotator, incomplete corpus relevance coverage. Curve seeds measure training subset sensitivity, not confidence intervals.'}
    folder = ROOT / 'evaluation/reports'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ('preferences_test.json' if final_test else 'preferences_validation.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    model['corpus_hash'] = engine.fingerprint
    (folder / 'preference_model.json').write_text(json.dumps(model, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-test', action='store_true', help='Only after protocol/settings are frozen')
    args = parser.parse_args()
    print(json.dumps(experiment(SearchEngine(load_papers()), args.final_test), indent=2))
