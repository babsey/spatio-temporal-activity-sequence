# -*- coding: utf-8 -*-
#
# plot_sequence_EI_networks_stim_inputs.py
#
# Copyright 2019 Sebastian Spreizer
# The MIT License

import numpy as np
import pylab as pl
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

from lib.dbscan_cluster import detect_wrap
from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.ax_spines as ax_spines
import lib.protocol as protocol
set_fontsize()

import pickle
def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



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

bg_cluster_count = []

params['landscape'] = landscapes[-1]
for mean in np.arange(0., 501., 50.):
    for std in np.arange(0., 501., 50.):
        print('Mean: %s, std: %s' % (mean, std))
        params['noiseE'] = {'mean': mean, 'std': std}
        params['noiseI'] = {'mean': mean, 'std': std}
        gids, ts = protocol.get_or_simulate(simulation, params)

        if len(ts) == 0:
            bg_cluster_count.append(0)
            continue

        idx = (gids - 1 < nrow * ncol) * ts > recstart
        ts, gids = ts[idx], gids[idx]
        nclusters, clusters = detect_wrap(ts, gids, nrow, ncol, td=4, eps=4)
        bg_cluster_count.append(nclusters - 1)



simulation = 'sequence_EI_networks_stim'
params = protocol.get_parameters(simulation)

##########################################

nrowE, ncolE = params['nrowE'], params['ncolE']
npopE = nrowE * ncolE

nrow = ncol = 120
npop = nrow * ncol
recstart = 500.
offsetE = 1
nseries = params['stim']['nseries']
simtime = nseries * 500.

stds = np.arange(0., 501., 50.)
means = np.arange(0., 501., 50.)

####################################

stim_spike_starttime = []
stim_cluster_count = []
stim_cluster_mean_lifespan = []

for i, mean in enumerate(means):
    for i, std in enumerate(stds):
        print('Mean: %s, std: %s' % (mean, std))
        params = {
            "noiseE": {
                "mean": mean,
                "std": std
            },
            "noiseI": {
                "mean": mean,
                "std": std
            },
        }

        gids, ts = protocol.get_or_simulate(simulation, params)

        if len(ts) == 0:
            stim_spike_starttime.append(np.nan)
            stim_cluster_count.append(0)
            stim_cluster_mean_lifespan.append(0)
            continue

        idx = (gids - 1 < nrow * ncol) * ts > recstart
        ts, gids = ts[idx], gids[idx]
        nclusters, clusters = detect_wrap(ts, gids, nrow, ncol, td=4, eps=4)

        if nclusters == 1:
            stim_cluster_mean_lifespan.append(0)
            stim_spike_starttime.append(np.nan)
        else:
            lifespan = []
            spikes = []
            starttime = []
            for cid in range(nclusters - 1):
                cidx = clusters == cid
                if not np.any(cidx):
                    continue
                is_evoked = np.any(gids[cidx] // nrow == (nrow / 2)) and np.any(gids[cidx] % ncol == (ncol / 2)) and np.any(
                    ((ts[cidx] - recstart) % 500. < 50.)) and not np.any(((ts[cidx] - recstart) % 500. > 450.))
                if is_evoked:
                    spikes.append(len(ts[cidx]))
                    starttime.append(np.min(ts[cidx]))
                    dt = np.max(ts[cidx]) - np.min(ts[cidx])
                    lifespan.append(dt)
            stim_spike_starttime.append(np.nanmean(
                (np.array(starttime) - recstart) % 500.))
            stim_cluster_mean_lifespan.append(np.nanmean(lifespan))
            stim_cluster_count.append(len(lifespan))





colors = pl.rcParams['axes.prop_cycle'].by_key()['color']

bg_cluster_count = in_cluster_count
n = 11
step = 5

fig, ax = pl.subplots(2, 4, sharex=True, sharey=True, dpi=300)
fig.subplots_adjust(wspace=.7)

im0 = ax[1, 0].imshow(np.array(bg_cluster_count).reshape(n, n) / 5., origin='bottom', vmin=0, vmax=100)
im1 = ax[1, 1].imshow(np.array(stim_cluster_count).reshape(n, n) / 20., origin='bottom', vmin=0, vmax=1)
im2 = ax[1, 2].imshow(np.array(stim_spike_starttime).reshape(n, n), origin='bottom', vmin=0, vmax=30)
im3 = ax[1, 3].imshow(np.array(stim_cluster_mean_lifespan).reshape(n, n), origin='bottom', vmin=0, vmax=100)

ax[1, 0].locator_params(nbins=3)
ax[1, 0].set_xticklabels([0] + np.linspace(0, 500, 3).astype(int).tolist())
ax[1, 0].set_yticklabels([0] + np.linspace(0, 500, 3).astype(int).tolist())
ax[1, 0].set_xlabel('Input standard deviation (pA)')
ax[1, 0].set_ylabel('Input mean (pA)')

divider = make_axes_locatable(ax[1, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar0 = pl.colorbar(im0, cax=cax, extend='max')
cbar0.set_ticks([0, 50, 100])
cbar0.set_label('Number of clusters per seconds', fontsize=5)
divider = make_axes_locatable(ax[1, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = pl.colorbar(im1, cax=cax)
cbar1.set_ticks([0, .5, 1.])
cbar1.set_label('Probability of evoked clusters', fontsize=5)
divider = make_axes_locatable(ax[1, 2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = pl.colorbar(im2, cax=cax, extend='max')
cbar2.set_ticks([0, 10, 20, 30])
cbar2.set_label('Reaction time [ms]', fontsize=5)
divider = make_axes_locatable(ax[1, 3])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar3 = pl.colorbar(im3, cax=cax, extend='max')
cbar3.set_ticks([0, 50, 100])
cbar3.set_label('Lifespan of cluster [ms]', fontsize=5)

panel_label(ax[1, 0], 'c', x=-.8)
panel_label(ax[1, 1], 'd', x=-.2)
panel_label(ax[1, 2], 'e', x=-.2)
panel_label(ax[1, 3], 'f', x=-.2)

for i in range(4):
    ax_spines.set_default(ax[1, i])

for i in range(4):
    ax[0, i].remove()


# axA = fig.add_subplot(2, 2, 1)
axA = pl.subplot2grid((2,4), (0,0), rowspan=1, colspan=2)
panel_label(axA, 'a', x=-.27)
for i in range(4):
    circle = patches.Circle((5, 3), .5, linewidth=1, edgecolor=colors[3], facecolor='none')
    ax[1, i].add_patch(circle)
ax[1, 1].text(5, 3, "a", {'color': 'black', 'ha': 'center', 'va': 'center', 'size': 5})

params = {
    "noiseE": {
        "mean": 150.,
        "std": 250.
    },
    "noiseI": {
        "mean": 150.,
        "std": 250.
    },
}

gids, ts = protocol.get_or_simulate(simulation, params)
idx = (gids - 1 < nrow * ncol) * ts > recstart
ts, gids = ts[idx], gids[idx]
nclusters, clusters = detect_wrap(ts, gids, nrow, ncol, td=4, eps=4)

idx = (ts > 1900) * (ts < 2400)
axA.plot(ts[idx], gids[idx], 'k.', alpha=.1, ms=2)

for cid in range(nclusters - 1):
    cidx = clusters == cid
    if not np.any(cidx):
        continue
    is_evoked = np.any(gids[cidx] // nrow == (nrow / 2)) and np.any(gids[cidx] % ncol == (ncol / 2)) and np.any(
        ((ts[cidx] - recstart) % 500. < 50.)) and not np.any(((ts[cidx] - recstart) % 500. > 450.))
    # ((ts[cidx] - recstart) >2000)) and not np.any(((ts[cidx] - recstart) < 2050))
    if is_evoked:
        axA.plot(ts[cidx], gids[cidx], '.', color=colors[3], ms=2)


axA.set_xlabel('Time [ms]')
axA.set_ylabel('Neuron ID')
axA.locator_params(nbins=3)
ax_spines.set_default(axA)
axA.set_xlim(1980, 2350)
axA.set_ylim(-100, 120 * 120 + 101)

axB = pl.subplot2grid((2,4), (0,2), rowspan=1, colspan=2)
for i in range(4):
    circle = patches.Circle((7, 5), .5, linewidth=1, edgecolor=colors[1], facecolor='none')
    ax[1, i].add_patch(circle)
ax[1, 1].text(7, 5, "b", {'color': 'white', 'ha': 'center', 'va': 'center', 'size': 5})

params = {
    "noiseE": {
        "mean": 250.,
        "std": 350.
    },
    "noiseI": {
        "mean": 250.,
        "std": 350.
    },
}

gids, ts = protocol.get_or_simulate(simulation, params)
idx = (gids - 1 < nrow * ncol) * ts > recstart
ts, gids = ts[idx], gids[idx]
nclusters, clusters = detect_wrap(ts, gids, nrow, ncol, td=4, eps=4)


idx = (ts > 1900) * (ts < 2400)
axB.plot(ts[idx][::step], gids[idx][::step], 'k.', alpha=.1, ms=2)

for cid in range(nclusters - 1):
    cidx = clusters == cid
    if not np.any(cidx):
        continue
    is_evoked = np.any(gids[cidx] // nrow == (nrow / 2)) and np.any(gids[cidx] % ncol == (ncol / 2)) and np.any(
        ((ts[cidx] - recstart) % 500. < 50.)) and not np.any(((ts[cidx] - recstart) % 500. > 450.))
    # ((ts[cidx] - recstart) >2000)) and not np.any(((ts[cidx] - recstart) < 2050))
    if is_evoked:
        axB.plot(ts[cidx][::step], gids[cidx][::step], '.', color=colors[1], ms=2)

panel_label(axB, 'b', x=-.2)
axB.set_xlabel('Time [ms]')
axB.locator_params(nbins=3)
ax_spines.set_default(axB)
axB.set_xlim(1980, 2380)
axB.set_ylim(-100, 120 * 120 + 101)


filename = 'sequence_EI_networks_stim_inputs'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')


pl.show()
