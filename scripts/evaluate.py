import argparse
import json
import random
import statistics
import time
from biopaper.data import ROOT, load_jsonl, load_papers, write_jsonl
from biopaper.metrics import ranking_metrics
from biopaper.evaluation import require_complete_pool
from biopaper.retrieval import SearchEngine

METHODS = ['bm25', 'dense', 'rerank']


def queries():
    return json.loads((ROOT / 'evaluation/queries.json').read_text(encoding='utf-8'))


def make_pool(engine, depth, candidate_k):
    rows = []
    for query in queries():
        union = {}
        for method in METHODS:
            results, _ = engine.search(query['text'], method, depth, candidate_k)
            for paper in results:
                union[paper['id']] = {"query_id": query['id'], "paper_id": paper['id'], "corpus_hash": engine.fingerprint}
        ids = sorted(union)
        random.Random(query['id']).shuffle(ids)
        rows.extend(union[i] for i in ids)
    write_jsonl(ROOT / 'evaluation/pool.jsonl', rows)
    print(f'Saved {len(rows)} unique query-document pairs. Label them in the app.')


def evaluate(engine, split, k, candidate_k):
    labels = load_jsonl(ROOT / 'evaluation/judgments.jsonl')
    judgments = {}
    for row in labels:
        if row['corpus_hash'] != engine.fingerprint:
            continue
        judgments.setdefault(row['query_id'], {})[row['paper_id']] = row['grade']
    selected = [q for q in queries() if q['split'] == split]
    if not selected:
        raise ValueError('No queries in this split')
    require_complete_pool(selected, load_jsonl(ROOT / 'evaluation/pool.jsonl'), judgments, engine.fingerprint)
    # Warm all models so reported timings exclude downloads and index construction.
    for method in METHODS:
        engine.search(selected[0]['text'], method, k, candidate_k)
    rows = []
    for q in selected:
        for method in METHODS:
            durations = []
            for _ in range(3):
                result, elapsed = engine.search(q['text'], method, k, candidate_k)
                durations.append(elapsed)
            metrics = ranking_metrics([r['id'] for r in result], judgments.get(q['id'], {}), k)
            rows.append({"query_id": q['id'], "method": method, **metrics, "latency_ms_median": statistics.median(durations)})
    summary = []
    for method in METHODS:
        values = [r for r in rows if r['method'] == method]
        summary.append({"method": method, "mean_ndcg": statistics.mean(r['ndcg'] for r in values), "mean_pooled_recall": statistics.mean(r['pooled_recall'] for r in values), "median_latency_ms": statistics.median(r['latency_ms_median'] for r in values)})
    report = {"split": split, "k": k, "candidate_k": candidate_k, "corpus_hash": engine.fingerprint, "models": [engine.embedder_name, engine.reranker_name], "query_count": len(selected), "summary": summary, "per_query": rows, "limitations": "Single annotator; incomplete relevance pool; not exhaustive recall; generic English models; three warmed timing repeats."}
    path = ROOT / 'evaluation/reports' / (split + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['pool', 'evaluate'])
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--candidate-k', type=int, default=30)
    a = p.parse_args()
    engine = SearchEngine(load_papers())
    if a.action == 'pool':
        make_pool(engine, a.depth, a.candidate_k)
    else:
        evaluate(engine, a.split, a.k, a.candidate_k)
