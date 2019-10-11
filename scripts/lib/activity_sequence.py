# -*- coding: utf-8 -*-
#
# activity_sequence.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import lib.dbscan_cluster as dbscan


__all__ = [
    'detect',
    'identify_vectors',
]


def detect(ts, row, col, labels, steps=1, width=50):
    """
    Returns data points of centroids in individual sequences.

    ts - time data
    row, col - sender data in rows and cols
    labels - index of clusters for each spike (-1 = unclustered)
    width - temporal width for centroids

    Returns
    -------
    sequences - a list of centroids points for sequences
    """

    vidx = []
    vts = []
    vrow = []
    vcol = []

    for label in set(labels):
        if label == -1:
            continue

        idx = labels == label
        t, r, c = ts[idx], row[idx], col[idx]

        vt = np.arange(np.floor(t.min()), np.ceil(
            t.max()) - width - steps, steps)
        for vti in vt:
            tw = (t >= vti) * (t < vti + width)
            if tw.any():
                vts.extend([vti])
                vidx.extend([label])
                vrow.extend([np.nanmean(r[tw])])
                vcol.extend([np.nanmean(c[tw])])

    return np.array([vts, vrow, vcol, vidx])


def identify_vectors(ts, gids, nrow, ncol, steps=10, width=25, td=20, eps=3):

    row = gids % nrow
    col = gids // ncol

    clusters = dbscan.detect([ts / td, row, col], eps=eps, min_samples=20)
    sequences = detect(
        ts, row, col, clusters[1], steps=steps, width=width)

    sequences = np.vstack([sequences, np.empty(len(sequences[0])), np.empty(len(sequences[0]))])
    sequences[4].fill(np.nan)
    sequences[5].fill(np.nan)

    for ii in np.unique(sequences[3]):
        idx = sequences[3] == ii
        sequence = sequences[:, idx]
        dx, dy = np.diff(sequence[[1, 2]], axis=1)
        angle = np.arctan2(dy, dx)
        angle = np.append(np.nan, angle)
        size = [np.nan] + [np.linalg.norm(p) for p in zip(dx, dy)]
        sequences[4][idx] = angle
        sequences[5][idx] = size

    return clusters, sequences
