# -*- coding: utf-8 -*-
#
# connectivity_landscape.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import noise


def homogeneous(nrow, phi=4):
    npop = np.power(nrow, 2)
    landscape = np.ones(npop, dtype=int) * phi
    return landscape


def random(nrow, seed=0):
    np.random.seed(seed)
    npop = np.power(nrow, 2)
    landscape = np.random.randint(8, size=npop)
    return landscape


def tiled(nrow, size=10, seed=0):
    np.random.seed(seed)
    ncol_dir = nrow /size
    didx = np.random.randint(0, 8, size=[ncol_dir, ncol_dir])
    landscape = np.repeat(np.repeat(didx, size, 0), size, 1)
    return landscape.ravel()


def Perlin(nrow, size=5, base=0):
    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size, base=base)
         for j in y] for i in x]
    m = n - np.min(n)
    m /= m.max()
    return m.ravel()


def Perlin_uniform(nrow, *args, **kwargs):
    m = Perlin(nrow, *args, **kwargs)
    a = np.argsort(m)
    b = np.power(nrow, 2) // 8;
    for j,i in enumerate(np.linspace(0,1,8)):
        m[a[j * b:(j + 1) * b]] = i
    return m


def move(nrow):
    return np.array([1, nrow + 1, nrow, nrow - 1, -1, -nrow - 1, -nrow, -nrow + 1])
