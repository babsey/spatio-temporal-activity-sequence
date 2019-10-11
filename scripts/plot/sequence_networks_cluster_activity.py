# -*- coding: utf-8 -*-
#
# plot_sequence_networks_cluster_activity.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib as mpl
import pylab as pl

from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.activity_sequence as seq
import lib.dbscan_cluster as dbscan
import lib.protocol as protocol
set_fontsize()


def plot_clustered_activity(axes, simulation, nrow, ncol, ts_bins, eps, td, center, hanning_width):

    title = ' '.join(simulation.split('_')[1:])
    gids, ts = protocol.get_or_simulate(simulation)

    npop = nrow * ncol
    idx = ((gids - 1) < (nrow * ncol)) * (ts > wuptime)
    gids, ts = gids[idx], ts[idx]

    nclusters, clusters = dbscan.detect_wrap(ts, gids - 1, nrow, ncol, td, eps)
    clustersize = np.bincount(clusters + 1)

    ax = axes[0]
    ax.clear()
    steps = 10
    for i in range(nclusters):
        if clustersize[i] < 100:
            continue
        idx = (clusters == i)
        ax.plot(ts[idx][::steps] / 1000., gids[idx][::steps], '.', ms=2)

    ax.text(0.04, 0.95, title,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, fontsize=8,
            bbox={'facecolor': 'white', 'pad': 3})

    yfmt = mpl.ticker.ScalarFormatter()
    yfmt.set_powerlimits((0, 2))
    ax.yaxis.set_major_formatter(yfmt)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.locator_params(nbins=3)

    ax.set_ylabel('Neuron')

    c = np.hanning(hanning_width)
    cx, cy = center
    g = np.unique(gids) - 1
    n = 10
    gx = g[((cx + n) > g // nrow) * (g // nrow > (cx - n))
           * ((cy + n) > g % nrow) * (g % nrow > (cy - n))]
    idx = np.in1d(gids - 1, gx)

    tsc, gidsc = ts[idx], gids[idx]
    h = np.histogram2d(
        tsc, gidsc - 1, bins=[ts_bins, range(nrow * ncol + 1)])[0]
    gidx = np.sum(h, 0) != 0

    h = h[:, gidx]
    hh = np.array([np.convolve(hi, c, mode='valid') for hi in h.T]).T

    spike_sort = np.argsort(np.argmax(hh, 0))
    hh1 = hh[:, spike_sort]

    g = np.unique(gids)
    np.random.shuffle(g)
    idx = np.in1d(gids, g[:400])

    tsc, gidsc = ts[idx], gids[idx]
    h = np.histogram2d(
        tsc, gidsc - 1, bins=[ts_bins, range(nrow * ncol + 1)])[0]
    gidx = np.sum(h, 0) != 0

    h = h[:, gidx]
    hh = np.array([np.convolve(hi, c, mode='valid') for hi in h.T]).T

    spike_sort = np.argsort(np.argmax(hh, 0))
    hh2 = hh[:, spike_sort]

    ax = axes[1]
    ax.clear()
    ax.matshow(hh1.T, aspect='auto', origin='bottom', cmap='binary', extent=[0, ts_bins[-1] / 1000., 0, hh1.shape[1]])
    ax.yaxis.set_major_formatter(yfmt)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.locator_params(nbins=3)

    ax = axes[2]
    ax.clear()
    ax.matshow(hh2.T, aspect='auto', origin='bottom', cmap='binary', extent=[0, ts_bins[-1] / 1000., 0, hh1.shape[1]])
    ax.yaxis.set_major_formatter(yfmt)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.locator_params(nbins=3)


wuptime = 500.

simulationI = 'sequence_I_networks'
simulationEI = 'sequence_EI_networks'

fig = pl.figure(dpi=300)
fig.set_size_inches(6.4, 4.8)
fig.subplots_adjust(hspace=.5)
fig.suptitle('Sequential activity of the Perlin networks')

axI1 = pl.subplot2grid((2, 4), (0, 0), colspan=2)
axI2 = pl.subplot2grid((2, 4), (0, 2))
axI3 = pl.subplot2grid((2, 4), (0, 3))
axEI1 = pl.subplot2grid((2, 4), (1, 0), colspan=2)
axEI2 = pl.subplot2grid((2, 4), (1, 2))
axEI3 = pl.subplot2grid((2, 4), (1, 3))

plot_clustered_activity([axI1, axI2, axI3], simulationI, 100, 100, np.arange(1000., 4000.1, 10.), 3, 20, [50, 50], 10)
plot_clustered_activity([axEI1, axEI2, axEI3], simulationEI, 120, 120,
                        np.arange(2000., 3000.1, 10.), 3, 3, [60, 60], 10)

axI1.set_xlim([1, 4])
axEI1.set_xlim([2, 3])

axEI1.set_xlabel('Time [s]')
axEI2.set_xlabel('Time [s]')

x = -.15
panel_label(axI1, 'a', x=-.2)
panel_label(axI2, 'b', x=x)
panel_label(axI3, 'c', x=x)
panel_label(axEI1, 'd', x=-.2)
panel_label(axEI2, 'e', x=x)
panel_label(axEI3, 'f', x=x)

filename = 'sequence_networks_cluster_activity'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
