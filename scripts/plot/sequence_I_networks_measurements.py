# -*- coding: utf-8 -*-
#
# plot_sequence_I_networks_measurements.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import scipy.io as sio
import scipy.ndimage as snd
import matplotlib as mpl
import matplotlib.cm as cm
import pylab as pl

from lib.circular_colormap import gen_circular_cmap
from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.activity_sequence as seq
import lib.ax_spines as ax_spines
import lib.protocol as protocol
set_fontsize()


def center_of_mass(w, idx):
    return snd.measurements.center_of_mass(np.roll(w, -idx + center).reshape(nrow, -1))


def plot_indegree_map(ax, W, cmap=cm.viridis):
    nrow, ncol = params['nrow'], params['ncol']
    ax.axis('off')
    indegree = np.sum(W, 0)
    im = ax.matshow(indegree.reshape(nrow, ncol), cmap=cmap,
                    origin='bottom', vmin=800, vmax=1200)
    ax.set_aspect(1., adjustable='box')
    return im, indegree


def plot_rate_map(ax, gids, cmap=cm.viridis):
    ax.axis('off')
    spike_rates = []
    for jj in range(2):
        spike_count = np.histogram(
            gids[jj], bins=range(npop + 1))[0].astype(float)
        spike_rate = spike_count / float(params['simtime']) * 2 * 1000.
        spike_rates.append(spike_rate)
    im = ax.matshow(np.log2(np.mean(spike_rates, 0)).reshape(nrow, ncol), cmap=cmap,
                    origin='bottom', vmin=0., vmax=np.log2(rate_max))
    ax.set_aspect(1., adjustable='box')
    return im, spike_rates


def plot_direction_quiver(ax, sequences, cmap=cmap):
    ax.axis('off')
    t, x, y, c, a, s = sequences
    ii = np.invert(np.isnan(a) * np.isnan(s)) * t >= 500.
    X, Y = x[ii], y[ii]
    U = np.cos(a[ii])
    V = np.sin(a[ii])
    n = 20
    ax.quiver(X[::n], Y[::n], U[::n] * s[ii][::n], V[::n] * s[ii][::n], a[ii][::n] % (2 * np.pi), cmap=cmap,
              pivot='middle', scale=20, headwidth=5, headlength=5, width=0.005, headaxislength=5, clim=[0, 2 * np.pi])
    ax.set_aspect(1., adjustable='box')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    return a[ii * t < 5500.] % (2 * np.pi), a[ii * t >= 5500.] % (2 * np.pi)


def plot_histogram(axes, S, indegree, rate, direction, idx):
    phi_bins = np.linspace(0, 2 * np.pi, 36)
    indegree_bins = range(800, 1201, 5)
    rate_bins = np.arange(0, rate_max + 1, 2.0)
    direction_bins = np.linspace(0, 2 * np.pi, 36)
    orientation = 'horizontal'

    ax = axes[0]
    ll = S[~np.isnan(S)]
    ax.hist(ll, bins=phi_bins, density=True, histtype='step', orientation=orientation, lw=1)
    ax_spines.set_default(ax)
    # ax.set_xlim(40, 12000.)
    ax.set_yticks(np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2))
    ax.set_yticklabels(['$0$', '$0.5{\pi}$', '$\pi$', '$1.5\pi$', '$2\pi$'])

    ax = axes[1]
    ax.hist(indegree, bins=indegree_bins, density=True, histtype='step', orientation=orientation, lw=1)
    ax_spines.set_default(ax)

    ax = axes[2]
    for jj in range(2):
        r = rate[jj]
        rr = r[~np.isnan(r)]
        ax.hist(rr, bins=rate_bins, histtype='step', density=True,
                orientation=orientation, color=paired_colors[1 - jj::2][idx], lw=1)
    ax_spines.set_default(ax)
    ax.set_xscale('log')

    ax = axes[3]
    ax.set_yticks(np.arange(0, 8, 2) + .5)
    for jj in range(2):
        d = direction[jj]
        dd = d[~np.isnan(d)]
        ax.hist(dd, bins=direction_bins, histtype='step', density=True,
                orientation=orientation, color=paired_colors[1 - jj::2][idx], lw=1)
    ax_spines.set_default(ax)

    ax.set_yticks(np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2))
    ax.set_yticklabels(['$0$', '$0.5{\pi}$', '$\pi$', '$1.5\pi$', '$2\pi$'])


landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'random'},
    # {'mode': 'Perlin', 'specs': {'perlin_max': 6}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 3}},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
]

simulation = 'sequence_I_networks'

td = 20
eps = 3
rate_max = 64.0

params = protocol.get_parameters(simulation).as_dict()

nrow, ncol = params['nrow'], params['ncol']
npop = nrow * ncol
center = int(ncol * nrow / 2 + nrow / 2)

paired_colors = cm.tab20.colors
cmap = gen_circular_cmap()

fig, axes = pl.subplots(4, 6, dpi=300)
fig.set_size_inches(6.4, 4.8)
fig.subplots_adjust(right=.8, left=.05, bottom=.08)
fig.suptitle('Measurements of the spiking activity in I networks')

for idx, landscape in enumerate(landscapes):
    params['landscape'] = landscape

    label = landscape['mode'].split('_')[0].capitalize()
    ax = axes[0, idx]
    ax.set_title(label, fontsize=6)
    ax.axis('off')
    W = sio.loadmat('Data/I_networks_%s_1.mat' % landscape['mode'])['W']
    C = np.array([center_of_mass(w.reshape(100, 100), widx) for (widx, w) in enumerate(W)])
    S = (np.array([np.arctan2(tx - nrow / 2, ty - ncol / 2) for (tx, ty) in C])) % (2 * np.pi)
    im = ax.matshow(S.reshape(nrow, ncol), cmap=cmap, origin='bottom', vmin=0, vmax=(2 * np.pi))

    if idx == 0:
        ax = axes[0, -3]
        axbar = axes[0, -2]
        box = ax.get_position()
        barbox = axbar.get_position()
        axbar.set_position([barbox.x0, box.y0, 0.01, box.height])
        norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
        cbar = mpl.colorbar.ColorbarBase(axbar, cmap=cmap, norm=norm)
        cbar.set_ticks(np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2))
        cbar.set_ticklabels(
            ['$0$', '$0.5{\pi}$', '$\pi$', '$1.5\pi$', '$2\pi$'])
        cbar.set_label('$\phi$')

    im, indegree = plot_indegree_map(axes[1, idx], W, cmap=cm.viridis)
    if idx == 0:
        ax = axes[1, -3]
        axbar = axes[1, -2]
        box = ax.get_position()
        barbox = axbar.get_position()
        axbar.set_position([barbox.x0, box.y0, 0.01, box.height])
        norm = mpl.colors.Normalize(vmin=800, vmax=1200)
        cbar = mpl.colorbar.ColorbarBase(axbar, cmap=cm.viridis, norm=norm)
        cbar.set_ticks(np.arange(800, 1201, 200))
        cbar.set_ticklabels(['0.8k', '1k', '1.2k'])
        cbar.set_label('Indegree')

    gids, ts = protocol.get_or_simulate(simulation, params)

    rate, direction = [], []

    sidx = ((gids - 1) < (npop)) * (ts > (500.))
    gids, ts = gids[sidx], ts[sidx]
    tidx = ts < 5500.

    im, rate = plot_rate_map(axes[2, idx], [gids[tidx] - 1, gids[~tidx] - 1], cmap=cm.viridis)
    if idx == 0:
        ax = axes[2, -3]
        axbar = axes[2, -2]
        box = ax.get_position()
        barbox = axbar.get_position()
        axbar.set_position([barbox.x0, box.y0, 0.01, box.height])
        norm = mpl.colors.Normalize(vmin=0, vmax=np.log2(rate_max))
        cbar = mpl.colorbar.ColorbarBase(axbar, cmap=cm.viridis, norm=norm, ticks=range(0, 7, 2))
        cbar.set_ticklabels([1, 4, 16, 64])
        cbar.set_label('Firing rate [Hz]')

    clusters, sequences = seq.identify_vectors(
        ts, gids, nrow, ncol, td=td, eps=eps)
    ax = axes[3, idx]
    direction = plot_direction_quiver(ax, sequences, cmap=cmap)
    if idx == 0:
        ax = axes[3, -3]
        axbar = axes[3, -2]
        box = ax.get_position()
        barbox = axbar.get_position()
        axbar.set_position([barbox.x0, box.y0, 0.01, box.height])
        norm = mpl.colors.Normalize(vmin=0, vmax=2 * np.pi)
        cbar = mpl.colorbar.ColorbarBase(axbar, cmap=cmap, norm=norm)
        cbar.set_ticks(np.arange(0, 2 * np.pi + np.pi / 2, np.pi / 2))
        cbar.set_ticklabels(['$0$', '$0.5{\pi}$', '$\pi$', '$1.5\pi$', '$2\pi$'])
        cbar.set_label('$\phi$')

    plot_histogram(axes[:, -1], S, indegree, rate, direction, idx)

ax = axes[0, -1]
ax.legend(['Symmetric', 'Random', 'Perlin', 'Homogeneous'], bbox_to_anchor=(1.2, 1.00))

ax = axes[-1, -1]
ax.set_xlabel('Probability')

rows = ['a', 'b', 'c', 'd']
cols = ['i', 'ii', 'iii', 'iv']
cols = ['1', '2', '3', '4']
for i in range(4):
    for j in range(4):
        panel_label(axes[i, j], rows[i] + cols[j], x=-.2)
    panel_label(axes[i, -1], rows[i] + '5', x=-.5)

filename = 'sequence_I_networks_measurements'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
