# Experiment protocol — version 1

## Research question

Can a lightweight ranker trained on a small number of human preferences improve biomedical literature search over keyword search, semantic retrieval, and pretrained reranking?

This pilot connects biomedical knowledge, information retrieval, preference learning, and software testing. An improvement is a hypothesis, not an existing result.

## Fixed design

| Item | Decision |
| --- | --- |
| Corpus | 300 Europe PMC abstracts, frozen by SHA-256 fingerprint |
| Queries | 30 English research questions |
| Training | q01–q15; relevance grades induce within-query preference pairs |
| Validation | q16–q20; learning curves and model decisions |
| Final test | q21–q30; only after settings are frozen |
| Baselines | BM25; MiniLM dense retrieval; dense top-30 + cross-encoder |
| Learned model | Linear score; pairwise logistic loss with L2 regularization |
| Features | log(1 + BM25), maximum passage cosine similarity, title query-token overlap |
| Training budget | 2, 5, 10, 15 queries; nested subsets for each seed 0, 1, 2 |
| Primary metric | nDCG@5 on the same judged validation pool |
| Secondary metric | Pooled Recall@5; baseline warmed search latency |

For a preferred document A over B, minimize `log(1 + exp(-(score(A) - score(B))))`. Feature scales use training documents only. Each training query has equal total weight. Grades 2 > 1 > 0 induce pairs; tied grades produce none. These are **induced preferences**, not directly collected pairwise choices. Training queries must contain at least two distinct grades.

The learned model ranks the whole corpus. A learned top-five result outside the initial baseline pool is added to blinded review automatically. The experiment refuses to save metrics until those papers have explicit human grades. Baseline scores are recomputed on the expanded pool so denominators remain comparable. Do not label unjudged documents as irrelevant.

## Run

1. Freeze the corpus and query wording before labeling.
2. Create the pool and grade the complete development set in the app.
3. Run **Experiment results → Run preference learning experiment**.
4. If it adds new papers, review them and rerun.
5. Compare curves and baselines on q16–q20. Inspect failures, document every change, and rerun validation only.
6. Freeze settings, grade the test pool, then run:

```powershell
.venv\Scripts\python.exe -m scripts.preference_experiment --final-test
```

The final model still trains on q01–q15; validation labels do not enter training. Initial evaluation of all development queries is diagnostic; it is not an unbiased score for the trained model.

## Error analysis to fill after evaluation

Record query, expected evidence, top result, grade, and likely failure cause. Examine terminology mismatch, broad topical similarity without answering the question, missing biological context, and missing corpus coverage. Do not choose only favorable examples.

## Limits and follow-up

Thirty questions and five validation questions constitute a pilot. Query topics are manually chosen and are not representative of every biomedical search. Some biological topics overlap across splits. Three subset seeds measure sensitivity to training examples, not statistical confidence. Preference pairs from one query are dependent, so pair count is not the count of independent annotations. The primary learning curve is measured in annotated **queries**, with induced pair counts shown separately.

Before a larger claim: expand and group-split queries by topic, add a second annotator, assess agreement, pin model revisions, measure learned-search latency, add a feature ablation, and evaluate a biomedical embedding model. Record unfavorable outcomes too.
