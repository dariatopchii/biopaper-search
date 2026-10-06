"""Linear pairwise ranking trained only on explicitly graded query-document pairs."""
import numpy as np
import hashlib
import json

FEATURES = ['log_bm25', 'dense_similarity', 'title_query_overlap']


def annotation_snapshot(queries, labels, fingerprint):
    relevant = sorted((r['query_id'], r['paper_id'], r['grade']) for r in labels if r['corpus_hash'] == fingerprint)
    return hashlib.sha256(json.dumps([queries, relevant], sort_keys=True).encode()).hexdigest()


def fit_ranker(groups, epochs=600, learning_rate=0.1, regularization=0.01):
    """Each group contains features and grades for ONE training query.

    Strict grade inequalities induce preferences; equal grades provide no pair.
    Each query contributes equal total weight, irrespective of pool size.
    """
    if len(groups) < 2:
        raise ValueError('Use at least two training queries with preferences')
    arrays = [np.asarray(g['features'], dtype=float) for g in groups]
    all_x = np.vstack(arrays)
    if all_x.ndim != 2 or all_x.shape[1] != len(FEATURES) or not np.isfinite(all_x).all():
        raise ValueError('Invalid training features')
    scale = all_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    differences, weights = [], []
    for group, x in zip(groups, arrays):
        grades = group['grades']
        if len(grades) != len(x) or any(type(g) is not int or g not in (0, 1, 2) for g in grades):
            raise ValueError('Expected one integer relevance grade per document')
        pairs = [(x[i] - x[j]) / scale for i in range(len(x)) for j in range(len(x)) if grades[i] > grades[j]]
        if not pairs:
            raise ValueError('Each training query needs at least two different relevance grades')
        differences.extend(pairs)
        weights.extend([1 / (len(groups) * len(pairs))] * len(pairs))
    delta, sample_weights = np.asarray(differences), np.asarray(weights)
    w = np.zeros(len(FEATURES))
    for _ in range(epochs):
        margin = np.clip(delta @ w, -40, 40)
        gradient = -(delta.T @ (sample_weights / (1 + np.exp(margin)))) + regularization * w
        w -= learning_rate * gradient
    return {'feature_names': FEATURES, 'weights': w.tolist(), 'scale': scale.tolist(),
            'training_query_ids': [g['query_id'] for g in groups], 'pair_count': len(delta),
            'epochs': epochs, 'learning_rate': learning_rate, 'regularization': regularization,
            'preference_source': 'strict inequalities induced from human 0/1/2 relevance grades'}


def predict_scores(model, features):
    if model['feature_names'] != FEATURES:
        raise ValueError('Incompatible ranker features')
    x = np.asarray(features, dtype=float)
    return (x / np.asarray(model['scale'])) @ np.asarray(model['weights'])
