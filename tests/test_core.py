import math
import tempfile
import unittest
from pathlib import Path
from biopaper.data import clean_text, load_papers, write_jsonl, corpus_hash
from biopaper.metrics import ranking_metrics
from biopaper.retrieval import SearchEngine
from biopaper.evaluation import require_complete_pool

PAPERS = [
    {'id': 'a', 'title': 'Protein folding', 'abstract': 'Predict protein structure using neural networks.'},
    {'id': 'b', 'title': 'Cell classification', 'abstract': 'Classify cell types in single cell RNA data.'},
    {'id': 'c', 'title': 'Drug toxicity', 'abstract': 'Predict drug toxicity from chemical structure.'},
]


class RetrievalTests(unittest.TestCase):
    def test_keyword_relevance_and_out_of_vocabulary(self):
        engine = SearchEngine(PAPERS)
        result, elapsed = engine.search('protein folding')
        self.assertEqual(result[0]['id'], 'a')
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(engine.search('zzzzzz')[0], [])

    def test_invalid_queries_and_limits(self):
        engine = SearchEngine(PAPERS)
        for query in ['', '   ', '?!', 'a' * 1001]:
            with self.assertRaises(ValueError):
                engine.search(query)
        with self.assertRaises(ValueError):
            engine.search('protein', k=5, candidate_k=2)

    def test_corpus_deduplication_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'papers.jsonl'
            write_jsonl(path, PAPERS + [PAPERS[0]])
            self.assertEqual(len(load_papers(path)), 3)
        self.assertNotEqual(corpus_hash(PAPERS), corpus_hash(PAPERS[::-1]))
        self.assertEqual(clean_text('<b>Protein</b> &amp; RNA'), 'Protein & RNA')


class MetricTests(unittest.TestCase):
    def test_pool_requires_labels_beyond_top_results(self):
        pool = [{'query_id': 'q', 'paper_id': i, 'corpus_hash': 'current'} for i in ['a', 'b']]
        with self.assertRaisesRegex(ValueError, '1 unreviewed'):
            require_complete_pool([{'id': 'q'}], pool, {'q': {'a': 2}}, 'current')
        require_complete_pool([{'id': 'q'}], pool, {'q': {'a': 2, 'b': 0}}, 'current')

    def test_old_corpus_pool_is_not_used(self):
        with self.assertRaisesRegex(ValueError, 'No review pool'):
            require_complete_pool([{'id': 'q'}], [{'query_id': 'q', 'paper_id': 'a', 'corpus_hash': 'old'}], {'q': {'a': 2}}, 'current')

    def test_perfect_and_reversed_ranking(self):
        labels = {'a': 2, 'b': 1, 'c': 0}
        self.assertEqual(ranking_metrics(['a', 'b'], labels, 2), {'pooled_recall': 1.0, 'ndcg': 1.0})
        result = ranking_metrics(['c', 'b'], labels, 2)
        self.assertEqual(result['pooled_recall'], 0.5)
        expected = (1 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(result['ndcg'], expected)

    def test_unjudged_duplicate_and_no_relevant(self):
        for ids, labels in [(['x'], {'a': 2}), (['a', 'a'], {'a': 2}), (['a'], {'a': 0}), (['a'], {'a': 3})]:
            with self.assertRaises(ValueError):
                ranking_metrics(ids, labels)


if __name__ == '__main__':
    unittest.main()
