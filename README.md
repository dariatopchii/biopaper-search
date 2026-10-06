# BioPaper Search

[![Tests](https://github.com/dariatopchii/biopaper-search/actions/workflows/tests.yml/badge.svg)](https://github.com/dariatopchii/biopaper-search/actions/workflows/tests.yml)

A biomedical literature search and preference-learning experiment by Daria Topchii, an Informatics master's student at LMU Munich with a background in molecular biotechnology.

**Research question:** can a lightweight ranker learn from a small number of human relevance judgments and improve biomedical search over keyword retrieval and pretrained semantic models?

**Status:** the retrieval app and preference-learning pipeline are implemented. Human annotations and quality results are pending. No improvement is claimed yet.

[Experiment protocol](docs/EXPERIMENT.md) · [Roadmap](docs/ROADMAP.md) · [Progress log](docs/PROGRESS.md) · [Tracked tasks](https://github.com/dariatopchii/biopaper-search/issues) · [GitHub milestones](https://github.com/dariatopchii/biopaper-search/milestones)

## Idea and approach

Researchers describe biological problems using different terminology. Keyword matches can miss related concepts; semantic matches can find topical articles that do not actually answer the question. This project compares those failures and tests whether explicit relevance feedback can teach a small, interpretable ranker to combine the signals better.

```mermaid
flowchart LR
    A[Europe PMC abstracts] --> B[Keyword and semantic retrieval]
    B --> C[Blinded human review]
    C --> D[Within-query induced preferences]
    D --> E[Linear pairwise ranker]
    E --> F[Separate validation questions]
    F --> G[Learning curves and error analysis]
    G --> H[Frozen settings and held-out test]
```

## What works

- Attributed article abstracts downloaded from Europe PMC, with a corpus fingerprint and acquisition manifest.
- BM25 keyword retrieval, normalized MiniLM embeddings, and cross-encoder reranking of 30 candidates.
- Token-based overlapping abstract chunks; document retrieval uses the maximum matching passage score.
- Side-by-side Streamlit search, blinded-method relevance review, and experiment tables.
- Persistent dense index, deterministic ties, deduplicated papers, and tests for metric correctness and failure cases.
- Pairwise preference learning, train-only feature scaling, and validation learning curves on query-disjoint splits.

## Track the goal

The [roadmap](docs/ROADMAP.md) defines completion criteria; the [progress log](docs/PROGRESS.md) records actual evidence and next steps. Use the repository's Experiment issue template for each new hypothesis and link its resulting pull request. Run tests on every push. Completed software milestones and pending research results are tracked separately.

For preference learning, use **Experiment results → Run preference learning experiment** after development annotations are complete. See the [protocol](docs/EXPERIMENT.md) before running the held-out test.

## Start on Windows

Use Python 3.11 or newer. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts/fetch_papers.py --limit 300
.venv\Scripts\python.exe -m streamlit run app.py
```

The first semantic search downloads two public Hugging Face models. Both run on CPU; a GPU and paid API are unnecessary. Model weights and corpus stay local. Keep this app bound to localhost: it is a single-user prototype, not an authenticated public service.

## Evaluate instead of guessing

1. Freeze a corpus snapshot. Start with 300 papers; expand to 2,000 after the workflow is verified.
2. Review the 30 supplied research questions for usefulness BEFORE judging. Twenty are development queries; ten are held-out test queries. Do not tune against test scores.
3. Create the union of top 10 results from all methods:

```powershell
.venv\Scripts\python.exe scripts/evaluate.py pool
```

4. In **Relevance review**, label every query-paper pair: 0 irrelevant, 1 partly useful, 2 directly relevant. Method names and scores are hidden. One annotator is a limitation; a second independent annotator would strengthen the benchmark.
5. Run development evaluation, tune if needed, rebuild the pool for new candidates, and label any newly retrieved papers. Freeze settings, then run test evaluation once:

```powershell
.venv\Scripts\python.exe scripts/evaluate.py evaluate --split dev
.venv\Scripts\python.exe scripts/evaluate.py evaluate --split test
```

The evaluator refuses unjudged top results and queries with no judged relevant documents. It reports nDCG@5, **pooled Recall@5**, and warmed median latency from three repetitions. Pooled recall is not exhaustive corpus recall. Corpus hashes prevent mixing labels or reports from different datasets. Review results and write error analysis; do not invent an improvement if the cross-encoder loses.

The entire selected relevance pool must be labeled before evaluation, including papers outside the final top five: otherwise the pooled recall denominator would be incomplete. In the app, **Relevance review** shows overall progress and advances through unreviewed papers after each explicit rating. **Experiment results → Run evaluation** runs the same evaluation workflow without a terminal once the selected split is ready. Questions with no relevant papers require a review of corpus coverage; the software will not invent positive labels.

## Reproducibility and limitations

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

Keep corpus_manifest.json and evaluation settings with experiment outputs. Dependencies are bounded in pyproject.toml; the lock file captures the tested environment. Model aliases may change upstream: pin Hugging Face model revisions before a final published benchmark. Results on a small corpus and 30 questions do not establish general biomedical retrieval performance. The generic embedding and reranking models were not selected for biomedical specialization; compare a domain model later. Long-document max-passage scoring can favor longer abstracts; examine this in error analysis. The cross-encoder sees the highest-scoring passage, so evidence elsewhere may be missed.

To recreate the tested package versions, install `requirements-lock.txt` first, then install this project with `pip install -e . --no-deps`. The lock excludes the machine-specific editable project path.

## Software validation

Tested locally on 6 October 2026: **13 automated tests passed**, covering ranking metrics, review behavior, preference learning, query-disjoint learning curves, and unjudged-result pool expansion. Real-corpus feature extraction produced 300 × 3 finite values. The experiment pipeline tests use synthetic fixtures; biomedical quality evaluation still requires human annotations.

All three retrieval methods returned results on a 300-abstract corpus and passed an offline smoke check. Streamlit's application test completed a three-method search with 15 result cards and no application exceptions. The initial blinded pool contains 601 distinct query-paper pairs for 30 questions. These checks verify functioning software, not retrieval quality; no benchmark improvement has been fabricated.

The query uses open-access articles, but open access alone does not grant uniform redistribution rights. Download data locally; check each article's licence before publishing abstracts. Corpus, model files, judgments, and reports are ignored by Git. Always display source attribution. This tool retrieves literature and does not generate medical recommendations.

## A credible portfolio deliverable

After human evaluation: add a results table, five failure examples, a brief architecture diagram, and a 60-second screen recording. Explain the tradeoff between quality and latency. Only then put numerical results into your CV.

## Sources

- [Europe PMC REST API](https://europepmc.org/RestfulWebService)
- [Sentence Transformers semantic similarity](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
- [Embedding model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Reranker model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
