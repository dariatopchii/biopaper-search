# Progress log

Update this file when a milestone changes. Record evidence, remaining uncertainty, and the next concrete step.

## 2026-10-06 — Baseline prototype

Implemented BM25, dense retrieval, cross-encoder reranking, and a local Streamlit interface. Downloaded 300 real abstracts. Created an initial blinded review pool of 601 query-document pairs for 30 questions. Offline retrieval smoke checks succeeded. Human quality evaluation remains pending.

## 2026-10-06 — Preference experiment and repository structure

Added a linear pairwise ranker with three interpretable features, a 15/5/10 query split, validation learning curves for three subset seeds, and a final-test entry point. Newly retrieved unjudged papers are queued for human review before metrics can be saved. Added an experiment protocol, milestone checklist, issue templates, and automated tests.

**Current bottleneck:** human relevance annotations. Neither model quality gains nor final benchmark scores have been established.

**Validation:** 13 automated tests passed locally. Real-corpus feature extraction produced 300 × 3 finite feature values. Synthetic experiment tests cover learning curves, query-disjoint evaluation, and automatic addition of unjudged learned results. These are software checks, not biomedical benchmark results.

**Next step:** review the question set and label development-query papers. Run validation, inspect error cases, then freeze the final protocol.

## 2026-10-06 — Public repository

Published the code and documentation at [dariatopchii/biopaper-search](https://github.com/dariatopchii/biopaper-search). Four milestones track the research stages; four issues track the next work packages. Automated tests run on pushes and pull requests. Downloaded abstracts, model caches, local judgments, and experiment reports remain outside Git.

- [Collect development annotations](https://github.com/dariatopchii/biopaper-search/issues/1)
- [Evaluate learning curves and failures](https://github.com/dariatopchii/biopaper-search/issues/2)
- [Publish held-out evidence](https://github.com/dariatopchii/biopaper-search/issues/3)
- [Design deeper follow-up](https://github.com/dariatopchii/biopaper-search/issues/4)
