# -*- coding: utf-8 -*-
#
# plot3d.py
#
# Copyright 2019 Sebastian Spreizer
# The MIT License

import pylab as pl
from mpl_toolkits.mplot3d import axes3d


def scatter(x, y, z):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, '.', markersize=1)
    return fig, ax
