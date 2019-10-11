# -*- coding: utf-8 -*-
#
# circular_colormap.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors


def gen_circular_cmap():

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = pl.cm.viridis(np.linspace(0., 1, 128))
    colors2 = pl.cm.inferno_r(np.linspace(0., 1, 128))

    # combine them and build a new colormap
    colors = np.vstack((colors1[5:][::-1], colors2[12:99][::-1]))
    circular_cmap = mcolors.LinearSegmentedColormap.from_list('circular_cmap', colors)
    return circular_cmap


if __name__ == '__main__':
    circular_cmap = gen_circular_cmap()
    data = np.random.rand(10,10) * 2 - 1

    pl.figure()
    pl.pcolor(data, cmap=circular_cmap)
    pl.colorbar()
    pl.show()
