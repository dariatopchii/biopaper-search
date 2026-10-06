import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from biopaper.data import ROOT, corpus_hash, write_jsonl, load_jsonl


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        papers = [
            {'id': 'a', 'title': 'Protein structure', 'abstract': 'Predicting protein structures using neural networks.', 'year': '2025', 'authors': 'Example', 'url': 'https://example.org/a'},
            {'id': 'b', 'title': 'Cell types', 'abstract': 'Classifying cell types from RNA sequencing.', 'year': '2025', 'authors': 'Example', 'url': 'https://example.org/b'},
        ]
        write_jsonl(self.root / 'data/papers.jsonl', papers)
        write_jsonl(self.root / 'evaluation/pool.jsonl', [{'query_id': 'q', 'paper_id': i, 'corpus_hash': corpus_hash(papers)} for i in ['a', 'b']])
        (self.root / 'evaluation/queries.json').write_text(json.dumps([{'id': 'q', 'text': 'Protein structure', 'split': 'dev'}]), encoding='utf-8')
        self.patch = patch('biopaper.data.ROOT', self.root)
        self.patch.start()
        self.app_path = str(ROOT / 'app.py')

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_explicit_rating_is_required(self):
        app = AppTest.from_file(self.app_path).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.radio[0].value)
        app.button[1].click().run()
        self.assertTrue(app.warning)
        self.assertFalse((self.root / 'evaluation/judgments.jsonl').exists())
        self.assertTrue(app.button[2].disabled)

    def test_save_advances_without_inventing_next_rating(self):
        app = AppTest.from_file(self.app_path).run()
        app.radio[0].set_value(2)
        app.button[1].click().run()
        self.assertEqual(len(app.exception), 0)
        saved = load_jsonl(self.root / 'evaluation/judgments.jsonl')
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['grade'], 2)
        self.assertIsNone(app.radio[0].value)
        app.radio[0].set_value(0)
        app.button[1].click().run()
        self.assertTrue(app.success)
        self.assertFalse(app.button[1].disabled)
        self.assertEqual(len(load_jsonl(self.root / 'evaluation/judgments.jsonl')), 2)
