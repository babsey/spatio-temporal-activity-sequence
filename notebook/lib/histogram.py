# -*- coding: utf-8 -*-
#
# histogram.py
#
# Copyright 2019 Sebastian Spreizer
# The MIT License


import numpy as np

def histogram(data, bins, noverlap=0, mode='iter'):
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    data : array_like
        Input data. The histogram is computed over the flattened array.
    bins : sequence of scalars
        It defines the bin edges, including the rightmost edge.
    noverlap : int, optional
        Number of bins to overlap.
    mode : str, optional
        A mode to compute overlapping histogram.


    Returns
    -------
    hist : array
        The values of the histogram.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    """

    hist, bin_edges = np.histogram(data, bins=bins)
    if noverlap == 0: return hist, bin_edges

    if mode == 'iter':
        h = [np.sum(hist[idx:idx+noverlap]) for idx in range(len(bins)-noverlap)]
    elif mode == 'map':
        h = map(lambda idx: np.sum(hist[idx:noverlap+idx]), range(len(bins)-noverlap))
    elif mode == 'convolve':
        h = np.convolve(hist,np.ones(noverlap),'valid')

    hist = np.array(h, dtype=float)
    return hist, bin_edges[:-noverlap+1]
