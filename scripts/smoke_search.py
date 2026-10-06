"""Exercise real models and corpus; record rankings and latency, not quality claims."""
import json
from biopaper.data import ROOT, load_papers
from biopaper.retrieval import SearchEngine

engine = SearchEngine(load_papers())
query = 'How can machine learning predict protein structure?'
output = {'query': query, 'corpus_hash': engine.fingerprint, 'results': {}}
for method in ['bm25', 'dense', 'rerank']:
    results, elapsed = engine.search(query, method)
    assert results and len({r['id'] for r in results}) == len(results)
    output['results'][method] = {'latency_ms_first_run': elapsed, 'papers': [{'id': r['id'], 'title': r['title'], 'score': r['score']} for r in results]}
    print(method, round(elapsed), 'ms:', results[0]['title'])
(ROOT / 'artifacts').mkdir(exist_ok=True)
(ROOT / 'artifacts/smoke.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
