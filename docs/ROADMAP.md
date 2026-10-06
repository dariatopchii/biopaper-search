# Roadmap and definition of done

Goal: produce a reproducible answer to whether small amounts of human preference data improve biomedical search, and explain the quality/runtime tradeoff.

## M1 — Working retrieval prototype

- [x] Acquire attributed Europe PMC abstracts with a corpus fingerprint.
- [x] Implement three retrieval baselines and a local comparison app.
- [x] Add explicit human relevance review without default labels.
- [x] Validate metrics and core review behavior.

## M2 — Preference-learning experiment

- [x] Implement a train-only-scaled pairwise linear model.
- [x] Separate train, validation, and final test questions.
- [x] Implement learning curves and expansion of unjudged result pools.
- [ ] Review query suitability and complete human development annotations.
- [ ] Run validation learning curves and compare every baseline.
- [ ] Inspect at least five failures and document causes.

## M3 — Reproducible evidence

- [ ] Pin pretrained model revisions and freeze experiment settings.
- [ ] Complete held-out annotations and run final test.
- [ ] Publish an aggregate results table with limitations and hardware details.
- [ ] Add a short demo recording and explain the architecture.

## M4 — Deeper follow-up, after the pilot

- [ ] Expand corpus and query set; separate related query topics across splits.
- [ ] Compare a biomedical embedding model.
- [ ] Ablate each learned feature and measure learned inference latency.
- [ ] Add a second annotator and measure agreement.

Success means a defensible experiment with reproducible evidence. A well-explained negative result also meets the research goal.
