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


def detect_wrap(ts, gids, nrow, ncol, td=1, eps=2, niteration=10):

    row1 = gids % nrow
    col1 = gids // ncol

    center = nrow * ncol / 2 + ncol / 2
    npop = nrow * ncol
    gids2 = (gids + center) % npop
    row2 = gids2 % nrow
    col2 = gids2 // ncol

    nclusters1, clusters1 = detect([ts / td, row1, col1], eps=eps, min_samples=20)
    nclusters2, clusters2 = detect([ts / td, row2, col2], eps=eps, min_samples=20)
    clusters = np.copy(clusters1)

    for ii in range(niteration):
        clusters_pairwise = np.unique(list(zip(clusters2,clusters)), axis=0)
        clu2,clu1 = clusters_pairwise.T
        clu2_set = np.unique(clu2, return_counts=True)
        clu2_set = clu2_set[0][clu2_set[1] > 1][1:]
        for cluId in clu2_set:
            clu = clu1[clu2 == cluId]
            clu = clu[clu>-1]
            idx = np.in1d(clusters, clu)
            clusters[idx] = clu[0]

        nclusters, clusters = np.unique(clusters + 1, return_inverse=True)
        return len(nclusters), clusters - 1
