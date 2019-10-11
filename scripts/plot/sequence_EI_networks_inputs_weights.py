# -*- coding: utf-8 -*-
#
# plot_sequence_EI_networks_inputs_weights.py
#
# Copyright 2019 Sebastian Spreizer
# The MIT License

import numpy as np
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

from lib.dbscan_cluster import detect_wrap
from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.ax_spines as ax_spines
import lib.protocol as protocol
set_fontsize()


simulation = 'sequence_EI_networks'
params = {}

landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
    {'mode': 'random'},
    {'mode': 'Perlin', 'specs': {'size': 4}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 3}},
]

nrow = ncol = 120
npop = nrow * ncol
recstart = 500.

in_spike_count = []
in_cluster_count = []
in_cluster_mean_lifespan = []

params['landscape'] = landscapes[-1]
for mean in np.arange(0., 501., 50.):
    for std in np.arange(0., 501., 50.):
        print('Mean: %s, std: %s' % (mean, std))
        params['noiseE'] = {'mean': mean, 'std': std}
        params['noiseI'] = {'mean': mean, 'std': std}
        gids, ts = protocol.get_or_simulate(simulation, params)

        if len(ts) == 0:
            in_spike_count.append(0)
            in_cluster_count.append(0)
            in_cluster_mean_lifespan.append(0)
            continue

        idx = (gids - 1 < nrow * ncol) * ts > recstart
        ts, gids = ts[idx], gids[idx]
        in_spike_count.append(len(ts))
        nclusters, clusters = detect_wrap(ts, gids, nrow, ncol, td=4, eps=4)
        in_cluster_count.append(nclusters - 1)

        if nclusters == 1:
            in_cluster_mean_lifespan.append(0)
        else:
            lifespan = []
            for cidx in range(nclusters - 1):
                t = ts[clusters == cidx]
                dt = np.max(t) - np.min(t)
                lifespan.append(dt)
            in_cluster_mean_lifespan.append(np.mean(lifespan))

params = {}
re_spike_count = []
re_cluster_count = []
re_cluster_mean_lifespan = []

params['landscape'] = landscapes[-1]
for Ji in [5., 10., 15., 20., 25.]:
    for g in [6, 7, 8, 9, 10]:
        print('Ji: %s, g: %s' % (Ji, g))
        params['Ji'] = Ji
        params['g'] = g
        gids, ts = protocol.get_or_simulate(simulation, params)

        if len(ts) == 0:
            re_cluster_count.append(0)
            re_spike_count.append(0)
            re_cluster_mean_lifespan.append(0)
            continue

        nrow = ncol = 120
        npop = nrow * ncol
        idx = (gids - 1 < nrow * ncol) * ts > 500.
        ts, gids = ts[idx], gids[idx]

        re_spike_count.append(len(ts))
        nclusters, clusters = detect_wrap(ts, gids, nrow, ncol, td=4, eps=4)
        re_cluster_count.append(nclusters)

        if nclusters == 1:
            re_cluster_mean_lifespan.append(0)
        else:
            lifespan = []
            for cidx in range(nclusters - 1):
                t = ts[clusters == cidx]
                dt = np.max(t) - np.min(t)
                lifespan.append(dt)
            re_cluster_mean_lifespan.append(np.mean(lifespan))


def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


colors = pl.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = pl.subplots(2, 3, dpi=300)
fig.subplots_adjust(wspace=.7)

im0 = ax[0, 0].imshow(np.array(in_spike_count).reshape(11, 11) / 5. / 120 / 120, origin='bottom')
im1 = ax[0, 1].imshow(np.array(in_cluster_count).reshape(11, 11) / 5., origin='bottom')
im2 = ax[0, 2].imshow(np.array(in_cluster_mean_lifespan).reshape(11, 11), origin='bottom')



for i in range(3):
    ax[0, i].set_xticks([0,0,5,10])
    ax[0, i].set_yticks([0,0,5,10])
    ax[0, i].set_xticklabels([0] + range(0, 501, 250))
    ax[0, i].set_yticklabels([0] + range(0, 501, 250))

ax[0, 0].set_xlabel('Input standard deviation (pA)')
ax[0, 0].set_ylabel('Input mean (pA)')

divider = make_axes_locatable(ax[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar0 = pl.colorbar(im0, cax=cax)
cbar0.set_ticks([0,7,14])
cbar0.set_label('Firing rate [spikes/s]', fontsize=6)
divider = make_axes_locatable(ax[0, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = pl.colorbar(im1, cax=cax)
cbar1.set_ticks([0,50,100])
cbar1.set_label('Number of clusters per seconds', fontsize=6)
divider = make_axes_locatable(ax[0, 2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = pl.colorbar(im2, cax=cax)
cbar2.set_ticks([0,0,40,80])
cbar2.set_label('Mean lifespan of cluster [ms]', fontsize=6)

panel_label(ax[0, 0], 'a', x=-.5)
panel_label(ax[0, 1], 'b', x=-.3)
panel_label(ax[0, 2], 'c', x=-.3)


for i in range(3):
    circle = patches.Circle((2, 7), .5, linewidth=1, edgecolor=colors[1], facecolor='none')
    ax[0, i].add_patch(circle)


im0 = ax[1, 0].imshow(np.array(re_spike_count).reshape(5, 5) / 5. / 120 / 120, origin='bottom')
im1 = ax[1, 1].imshow(np.array(re_cluster_count).reshape(5, 5) / 5., origin='bottom')
im2 = ax[1, 2].imshow(np.array(re_cluster_mean_lifespan).reshape(5, 5), origin='bottom')

for i in range(3):
    ax[1, i].set_xticks([0,2,4])
    ax[1, i].set_xticklabels([6,8,10])
    ax[1, i].set_yticks([0,2,4])
    ax[1, i].set_yticklabels([5,15,25])

ax[1, 0].set_xlabel('g')
ax[1, 0].set_ylabel('Synaptic weight (pA)')
divider = make_axes_locatable(ax[1, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = pl.colorbar(im1, cax=cax)
cbar1.set_ticks([0,20,90,160])
cbar1.set_label('Number of clusters per seconds', fontsize=6)
divider = make_axes_locatable(ax[1, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar0 = pl.colorbar(im0, cax=cax)
cbar0.set_ticks([0,5,20,35])
cbar0.set_label('Firing rate [spikes/sec]', fontsize=6)
divider = make_axes_locatable(ax[1, 2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = pl.colorbar(im0, cax=cax)
cbar2.set_ticks([0,5,20,35])
cbar2.set_label('Mean lifespan of cluster [ms]', fontsize=6)

panel_label(ax[1, 0], 'd', x=-.5)
panel_label(ax[1, 1], 'e', x=-.3)
panel_label(ax[1, 2], 'f', x=-.3)

for i in range(3):
    circle = patches.Circle((2, 1), .5, linewidth=1, edgecolor=colors[1], facecolor='none')
    ax[1, i].add_patch(circle)


filename = 'sequence_EI_networks_inputs_weights'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')
