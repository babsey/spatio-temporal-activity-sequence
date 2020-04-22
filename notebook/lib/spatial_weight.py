# -*- coding: utf-8 -*-
#
# spatial_weight.py
#
# Copyright 2019 Sebastian Spreizer
# The MIT License

import numpy as np

def cosine_weights(source, targets, nrow, phi0, w0=1., dw=.1):
    s = nrow * nrow // 2 + nrow // 2
    t = (targets - source + s) % np.power(nrow, 2)
    dy = (t % nrow) - (s % nrow)
    dx = (t // nrow) - (s // nrow)
    phi = np.arctan2(dy, dx)
    weights = w0 * (1. + dw * np.cos(-phi0 + phi))
    return weights
