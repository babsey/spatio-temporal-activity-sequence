# -*- coding: utf-8 -*-
#
# plot_sequence_I_networks_perlin_size.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib as mpl
import pylab as pl

from lib.circular_colormap import gen_circular_cmap
from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.activity_sequence as seq
import lib.ax_spines as ax_spines
import lib.connectivity_landscape as cl
import lib.protocol as protocol
set_fontsize()


def plot_landscape_map(ax, landscape):
    ax.matshow(landscape.reshape(nrow, ncol), vmin=0, vmax=7, origin='bottom', cmap=cmap)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_axis_off()
    return ax


def plot_rate_map(ax, gids):
    x, y = gids % ncol, gids // nrow
    h = np.histogram2d(x, y, bins=[range(ncol + 1), range(nrow + 1)])[0]
    ax.matshow(np.log10(h.T / 10.), origin='bottom', vmin=0, vmax=np.log10(50), cmap=cmap)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_axis_off()


def plot_direction_changing(das, label):
    bins = np.linspace(-np.pi, np.pi, 51)
    hd = np.histogram(das, bins=bins)[0]
    ax_da.plot(bins[:-1], hd, label=label, lw=1)


def plot_speed(speed, label):
    bins = np.linspace(0, 3, 51)
    hs = np.histogram(speed, bins=bins)[0]
    ax_s.plot(bins[:-1] / 25. * 1000., hs / float(np.sum(hs)), label=label, lw=1)


def plot(idx, params, label):
    landscape = cl.__dict__[params['landscape']['mode']](
        nrow, params['landscape'].get('specs', {}))
    gids, ts = protocol.get_or_simulate(simulation, params)
    ii = ts > recstart

    clusters, sequences = seq.identify_vectors(ts[ii], gids[ii] - 1, nrow, ncol, steps=steps, width=width, td=td, eps=eps)

    das = []
    t, x, y, c, a, s = sequences
    for cid in np.unique(c):
        cidx = c == cid
        da = np.diff(a[cidx])
        das.extend(da[~np.isnan(da)])

    nclu = []
    for i in range(0, 9000, 100):
        q = (t >= (i + 500)) * (t < (i + 1500))
        nclu.append((np.unique(c[q]).size))
    nclusters.append(nclu)

    ax1 = pl.subplot2grid((gridy, gridx), (0, idx))
    plot_landscape_map(ax1, landscape)
    ax1.set_title(label)

    ax2 = pl.subplot2grid((gridy, gridx), (1, idx))
    plot_rate_map(ax2, gids[ii])

    plot_speed(s[~np.isnan(s)], label.split('\n')[-1])
    plot_direction_changing(das, label.split('\n')[-1])

    return ax1, ax2


simulation = 'sequence_I_networks'
params = protocol.get_parameters(simulation).as_dict()

nrow, ncol = params['nrow'], params['ncol']
npop = nrow * ncol

landscapes = [
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 3}},
    {'mode': 'random'},
]

steps = 10
width = 25
td = 20
eps = 4

recstart = 500.

perlin_sizes = [3, 5, 10, 20, 50]
gridx = len(perlin_sizes) + 2
gridy = 5

nclusters = []

fig = pl.figure(dpi=300)
ax_c = pl.subplot2grid((10, 9), (5, 0), rowspan=3, colspan=2)
ax_s = pl.subplot2grid((10, 9), (5, 3), rowspan=3, colspan=2)
ax_da = pl.subplot2grid((10, 9), (5, 5), rowspan=3, colspan=3, projection='polar')

cmap = gen_circular_cmap()
params['landscape'] = landscapes[0]
ax1, ax2 = plot(0, params, 'Homogeneous\n$\infty$')

for idx, size in enumerate(perlin_sizes):
    params['landscape'] = landscapes[1]
    params['landscape']['specs']['size'] = size
    plot(idx + 1, params, 'Perlin\n%i' % (100 // size))

params['landscape'] = landscapes[-1]
plot(idx + 2, params, 'Random\n1')

box = ax_c.boxplot(nclusters, 0, '', widths = 0.6)

colors = pl.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(7):
    box['boxes'][i].set_color(colors[i])
    box['medians'][i].set_color(colors[i])
    box['whiskers'][i * 2].set_color(colors[i])
    box['whiskers'][i * 2 + 1].set_color(colors[i])
    box['caps'][i * 2].set_color(colors[i])
    box['caps'][i * 2 + 1].set_color(colors[i])

ax_spines.set_default(ax_c)
ax_da.legend(bbox_to_anchor=(2.1, 1.2), title='Perlin scale')

ax_c.set_xticks([1, 4, 7])
ax_c.set_xticklabels(['$\infty$', 10, 1])
ax_c.set_xlabel('Perlin scale')
ax_c.set_ylim(20, 70)
ax_c.set_ylabel('Number of sequences')

ax_da.set_theta_offset(np.pi / 2)
# ax_da.set_ylabel('Count')
ax_da.set_xlabel('Direction changing')
ax_spines.set_default(ax_da)
ax_da.locator_params(nbins=4)
ax_da.set_xticks([0, np.pi / 2, np.pi, np.pi * 3 / 2])
ax_da.set_xticklabels([0, '$\pi/2$', '$\pi$', '$-\pi/2$'])
ax_da.set_rticks([500, 1000, 1500])
ax_da.set_yticklabels([])
ax_da.set_rlabel_position(np.pi)

ax_s.set_ylabel('Probability')
ax_s.set_xlabel('Velocity [grid points / sec]')
ax_spines.set_default(ax_s)
ax_s.locator_params(nbins=3)

panel_label(ax1, 'a', x=-.84)
panel_label(ax_c, 'b', x=-.5)
panel_label(ax_s, 'c', x=-.5)
panel_label(ax_da, 'd', x=-.25)

filename = 'sequence_I_networks_perlin_size'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
