"""BM25 baseline and chunked dense retrieval with optional cross-encoder reranking."""
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from .data import ROOT, corpus_hash

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, papers, k1=1.5, b=0.75):
        self.papers, self.k1, self.b = papers, k1, b
        self.counts = [Counter(tokens(p["title"] + " " + p["abstract"])) for p in papers]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(len(papers), 1) or 1
        df = Counter(t for c in self.counts for t in c)
        self.idf = {t: math.log(1 + (len(papers) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query, k=5):
        scores = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            for word in set(tokens(query)):
                tf = counts[word]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                score += self.idf.get(word, 0) * tf * (self.k1 + 1) / denom
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda x: (-x[1], self.papers[x[0]]["id"]))[:k]


class SearchEngine:
    def __init__(self, papers, cache_dir=None, embedder=EMBEDDER, reranker=RERANKER):
        self.papers = papers
        self.bm25 = BM25(papers)
        self.embedder_name, self.reranker_name = embedder, reranker
        self.cache_dir = Path(cache_dir or ROOT / "artifacts")
        self.fingerprint = corpus_hash(papers)
        self._encoder = self._cross = self._vectors = None

    def _dense_index(self):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        if self._encoder is None:
            try:
                self._encoder = SentenceTransformer(self.embedder_name, device="cpu", local_files_only=True)
            except OSError:
                self._encoder = SentenceTransformer(self.embedder_name, device="cpu")
        if self._vectors is not None:
            return
        # Token-based overlapping chunks avoid silently dropping the end of an abstract.
        chunks, owners = [], []
        limit = min(int(self._encoder.max_seq_length), 256) - 2
        stride = max(1, limit - 40)
        for i, paper in enumerate(self.papers):
            ids = self._encoder.tokenizer.encode(paper["title"] + ". " + paper["abstract"], add_special_tokens=False, verbose=False)
            for start in range(0, len(ids), stride):
                chunks.append(self._encoder.tokenizer.decode(ids[start:start + limit], skip_special_tokens=True))
                owners.append(i)
                if start + limit >= len(ids):
                    break
        self._chunks, self._owners = chunks, np.asarray(owners)
        key = hashlib.sha256((self.fingerprint + self.embedder_name + "token-chunks-v1").encode()).hexdigest()[:24]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / (key + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                self._vectors = saved["vectors"]
            if self._vectors.shape[0] != len(chunks):
                self._vectors = None
        if self._vectors is None:
            self._vectors = self._encoder.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            np.savez_compressed(target, vectors=self._vectors)

    def search(self, query, method="bm25", k=5, candidate_k=30):
        query = query.strip()
        if not query or not tokens(query):
            raise ValueError("Enter a query containing letters or numbers")
        if len(query) > 1000:
            raise ValueError("Keep queries under 1,000 characters")
        if method not in {"bm25", "dense", "rerank"}:
            raise ValueError("Unknown search method")
        if k < 1 or candidate_k < k:
            raise ValueError("Require candidate_k >= k >= 1")
        started = time.perf_counter()
        if method == "bm25":
            ranked = self.bm25.search(query, k)
        else:
            import numpy as np
            self._dense_index()
            q = self._encoder.encode([query], normalize_embeddings=True)[0]
            similarities = self._vectors @ q
            best = {}
            for chunk, score in enumerate(similarities):
                owner = int(self._owners[chunk])
                if owner not in best or score > best[owner][0]:
                    best[owner] = (float(score), chunk)
            candidates = sorted(best, key=lambda i: (-best[i][0], self.papers[i]["id"]))[:candidate_k]
            ranked = [(i, best[i][0]) for i in candidates]
            if method == "rerank":
                from sentence_transformers import CrossEncoder
                if self._cross is None:
                    try:
                        self._cross = CrossEncoder(self.reranker_name, device="cpu", max_length=512, local_files_only=True)
                    except OSError:
                        self._cross = CrossEncoder(self.reranker_name, device="cpu", max_length=512)
                # Rerank the matched passage, not a truncated full abstract.
                pairs = [(query, self.papers[i]["title"] + ". " + self._chunks[best[i][1]]) for i in candidates]
                scores = self._cross.predict(pairs, show_progress_bar=False)
                ranked = sorted([(i, float(s)) for i, s in zip(candidates, scores)], key=lambda x: (-x[1], self.papers[x[0]]["id"]))
            ranked = ranked[:k]
        return [{**self.papers[i], "score": float(score), "rank": n + 1} for n, (i, score) in enumerate(ranked)], (time.perf_counter() - started) * 1000

    def preference_features(self, query):
        """Features for every document; no human labels enter feature extraction."""
        import numpy as np
        dense, _ = self.search(query, 'dense', len(self.papers), len(self.papers))
        similarities = {p['id']: p['score'] for p in dense}
        keywords = dict(self.bm25.search(query, len(self.papers)))
        query_words = set(tokens(query))
        return np.asarray([[math.log1p(keywords.get(i, 0)), similarities[p['id']],
                            len(query_words & set(tokens(p['title']))) / max(1, len(query_words))]
                           for i, p in enumerate(self.papers)])
