import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from biopaper.preferences import fit_ranker, predict_scores
from biopaper.data import write_jsonl, load_jsonl
from scripts.preference_experiment import experiment


class PreferenceTests(unittest.TestCase):
    def groups(self):
        return [{'query_id': q, 'features': [[0, 0, 0], [1, 1, 1], [2, 2, 2]], 'grades': [0, 1, 2]} for q in ['train_a', 'train_b']]

    def test_learns_preference_direction_and_records_training_queries(self):
        model = fit_ranker(self.groups())
        scores = predict_scores(model, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.assertTrue(np.all(np.diff(scores) > 0))
        self.assertEqual(model['pair_count'], 6)
        self.assertEqual(model['training_query_ids'], ['train_a', 'train_b'])

    def test_ties_do_not_invent_preferences(self):
        groups = self.groups()
        groups[0]['grades'] = [1, 1, 1]
        with self.assertRaises(ValueError):
            fit_ranker(groups)

    def test_scaling_uses_only_supplied_training_features(self):
        model = fit_ranker(self.groups())
        before = model['scale'].copy()
        predict_scores(model, [[100000, 100000, 100000]])
        self.assertEqual(model['scale'], before)
        with self.assertRaises(ValueError):
            fit_ranker(self.groups()[:1])


class ExperimentTests(unittest.TestCase):
    def test_query_disjoint_pipeline_and_unjudged_pool_expansion(self):
        class FakeEngine:
            fingerprint = 'fixture'
            papers = [{'id': name} for name in ['a', 'b', 'c']]

            def preference_features(self, query):
                return np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

            def search(self, query, method, k, candidate_k):
                return list(reversed(self.papers)), 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evaluation').mkdir()
            qs = [{'id': f'q{i:02}', 'text': f'Question {i}', 'split': 'dev' if i <= 20 else 'test'} for i in range(1, 31)]
            (root / 'evaluation/queries.json').write_text(json.dumps(qs), encoding='utf-8')
            labels = [{'query_id': q['id'], 'paper_id': p, 'corpus_hash': 'fixture', 'grade': grade} for q in qs for p, grade in [('a', 0), ('b', 1), ('c', 2)]]
            pool = [{k: v for k, v in r.items() if k != 'grade'} for r in labels]
            write_jsonl(root / 'evaluation/judgments.jsonl', labels)
            write_jsonl(root / 'evaluation/pool.jsonl', pool)
            with patch('scripts.preference_experiment.ROOT', root):
                report = experiment(FakeEngine())
                self.assertEqual(len(report['curves']), 12)
                self.assertEqual(len(report['baselines']), 15)
                self.assertFalse(set(report['training_query_ids']) & set(report['evaluation_query_ids']))
                test = experiment(FakeEngine(), final_test=True)
                self.assertEqual(test['evaluation_query_ids'], [q['id'] for q in qs[20:]])
                self.assertEqual(len(test['curves']), 1)
                # A learned result outside the initial pool must require explicit review.
                labels = [r for r in labels if not (r['query_id'] == 'q16' and r['paper_id'] == 'c')]
                pool = [r for r in pool if not (r['query_id'] == 'q16' and r['paper_id'] == 'c')]
                write_jsonl(root / 'evaluation/judgments.jsonl', labels)
                write_jsonl(root / 'evaluation/pool.jsonl', pool)
                with self.assertRaisesRegex(ValueError, 'Added 1 previously unjudged'):
                    experiment(FakeEngine())
                expanded = load_jsonl(root / 'evaluation/pool.jsonl')
                self.assertTrue(any(r['query_id'] == 'q16' and r['paper_id'] == 'c' for r in expanded))
