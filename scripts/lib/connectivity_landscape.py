# -*- coding: utf-8 -*-
#
# connectivity_landscape.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import noise

__all__ = [
    'symmetric',
    'homogeneous',
    'random',
    'tiled',
    'Perlin',
    'Perlin_uniform',
    'move'
]


def symmetric(nrow, specs={}):
    return

def homogeneous(nrow, specs={}):
    dir_idx = specs.get('phi', 4)

    npop = np.power(nrow, 2)
    landscape = np.ones(npop, dtype=int) * dir_idx
    return landscape


def random(nrow, specs={}):
    seed = specs.get('seed', 0)

    np.random.seed(seed)
    npop = np.power(nrow, 2)
    landscape = np.random.randint(8, size=npop)
    return landscape

def tiled(nrow, specs={}):
    seed = specs.get('seed', 0)
    tile_size = specs.get('tile_size', 10)

    np.random.seed(seed)
    ncol_dir = nrow / tile_size
    didx = np.random.randint(0, 8, size=[ncol_dir, ncol_dir])
    landscape = np.repeat(np.repeat(didx, tile_size, 0), tile_size, 1)
    return landscape.ravel()


def Perlin(nrow, specs={}):
    size = specs.get('size', 5)
    assert(size > 0)

    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size) for j in y] for i in x]
    m = n - np.min(n)
    m /= np.max(m)
    landscape = np.array(np.round(m * 7), dtype=int)
    return landscape.ravel()


def Perlin_uniform(nrow, specs={}):
    size = specs.get('size', 5)
    assert(size > 0)

    nrow = 100
    size = 4
    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size) for j in y] for i in x]
    m = np.concatenate(n)
    sorted_idx = np.argsort(m)
    max_val = nrow * 2
    idx = len(m) // max_val
    for ii, val in enumerate(range(max_val)):
        m[sorted_idx[ii * idx:(ii + 1) * idx]] = val
    landscape = (m - nrow) / nrow
    return landscape


def move(nrow):
    return np.array([1, nrow + 1, nrow, nrow - 1, -1, -nrow - 1, -nrow, -nrow + 1])
