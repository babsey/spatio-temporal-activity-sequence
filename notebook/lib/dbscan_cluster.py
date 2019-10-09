# -*- coding: utf-8 -*-
#
# dbscan_cluster.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
from sklearn.cluster import DBSCAN

__all__ = [
    'detect',
]


def detect(data, eps=1, min_samples=10, core_samples_mask=False):

    X = np.vstack(data).T
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    if core_samples_mask:
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        return n_clusters_, labels, core_samples_mask
    else:
        return n_clusters_, labels
