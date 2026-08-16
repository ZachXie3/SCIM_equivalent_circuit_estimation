"""Weighted-clustering + NN R2/R3 analysis (see ``cluster-plan.md``).

- ``cluster_model.py``    — data prep, feature encoding, the weighted
                            soft-clustering model (NN learns per-feature
                            weights), and the hard-assignment lookup.
- ``cluster_analysis.py`` — end-to-end runner: prep -> split -> train ->
                            baselines -> report -> plots -> lookup.
"""
