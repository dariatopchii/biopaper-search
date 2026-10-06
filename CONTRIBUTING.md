# Working on the project

Track research changes with an Experiment issue. State a hypothesis and evaluation plan before changing model settings. Use the roadmap for milestone status and update the progress log with observed evidence.

Run `python -m unittest discover -s tests -v` before opening a pull request. Unit tests use synthetic fixtures; they do not measure retrieval quality.

Do not commit model caches, downloaded abstracts, local annotations, credentials, or machine-specific paths. Publish aggregate benchmark evidence only after checking data rights and documenting the corpus and annotation protocol. Pin model revisions before the final benchmark.

Keep held-out queries out of training and model selection. A new query split is a new experiment, not a silent replacement of the old result.
