# -*- coding: utf-8 -*-
#
# plot_sequence_networks_connections.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib as mpl
import pylab as pl
import scipy.io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.ax_spines as ax_spines
import lib.connection_matrix as cm
import lib.protocol as protocol
set_fontsize()


landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
    {'mode': 'random'},
    # {'mode': 'Perlin', 'specs': {'size': 4}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 4}},
]

fig, axes = pl.subplots(2, 2, figsize=(4.8, 4.8), dpi=300)
fig.subplots_adjust(wspace=.2, hspace=.6, bottom=.1, left=.15, right=.9)

simulation = 'sequence_I_networks'

title = ' '.join(simulation.split('_')[1:])
params = protocol.get_parameters(simulation).as_dict()
ax = axes[0, 0]
for landscape in landscapes:
    params['landscape'] = landscape
    label = landscape['mode'].split('_')[0].capitalize()

    W = cm.I_networks(**params)
    h = np.array([np.histogram(Wi, bins=range(50))[0] for Wi in W])
    m = np.mean(h, axis=0)

    ax.plot(m, label=label, lw=1)
ax.set_xlim(0, 11)
ax.set_xlabel('Multiple connections')
ax.set_ylabel('Number of connections')
ax.set_title(title)
ax.set_yscale("log")

ax_spines.set_default(ax)


simulation = 'sequence_EI_networks'
title = ' '.join(simulation.split('_')[1:])
params = protocol.get_parameters(simulation).as_dict()
ax = axes[0, 1]
for landscape in landscapes:
    params['landscape'] = landscape
    label = landscape['mode'].split('_')[0].capitalize()

    W = cm.EI_networks(**params)
    h = np.array([np.histogram(Wi, bins=range(50))[0] for Wi in W[0]])
    m = np.mean(h, axis=0)

    ax.plot(m, label=label, lw=1)
ax.set_xlim(0, 20)
ax.set_title(title)
ax.set_yscale("log")
ax.legend(bbox_to_anchor=(.6, 1), fontsize=6)

ax_spines.set_default(ax)
axes[0, 1].set_yticklabels([])


nrowI = 100
centerI = int(nrowI * nrowI / 2 + nrowI / 2)
W_sym = sio.loadmat('Data/I_networks_symmetric_1.mat')['W']
Q_sym = np.array([np.roll(x, centerI - i) for i, x in enumerate(W_sym)])
x_sym = np.mean(Q_sym, 0)

W_homo = sio.loadmat('Data/I_networks_homogeneous_1.mat')['W']
Q_homo = np.array([np.roll(x, centerI - i) for i, x in enumerate(W_homo)])
x_homo = np.mean(Q_homo, 0)

dxI = (x_sym - x_homo) / x_sym


nrowE = 120
centerE = int(nrowE * nrowE / 2 + nrowE / 2)
W_sym = sio.loadmat('Data/EI_networks_symmetric_1.mat')['W_EE']
Q_sym = np.array([np.roll(x, centerE - i) for i, x in enumerate(W_sym)])
x_sym = np.mean(Q_sym, 0)

W_homo = sio.loadmat('Data/EI_networks_homogeneous_1.mat')['W_EE']
Q_homo = np.array([np.roll(x, centerE - i) for i, x in enumerate(W_homo)])
x_homo = np.mean(Q_homo, 0)

dxEI = (x_sym - x_homo) / x_sym


vmin, vmax = -1., 1.
im = []
im.append(axes[1, 0].matshow(dxI.reshape(nrowI, -1), vmin=vmin, vmax=vmax,
                             extent=[-50, 50 - 1, -50, 50 - 1], cmap='RdBu', origin='bottom'))
im.append(axes[1, 1].matshow(dxEI.reshape(nrowE, -1), vmin=vmin, vmax=vmax,
                             extent=[-60, 60 - 1, -60, 60 - 1], cmap='RdBu', origin='bottom'))
titles = ['I', 'EI']

lim = -30, 30

for i in range(2):
    ax = axes[1, i]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=.2)
    cbar = pl.colorbar(im[i], cax=cax)
    if i == 0:
        cax.remove()

    ax.xaxis.set_ticks_position('bottom')
    ax_spines.set_default(ax)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    ax.set_xticks(lim)
    ax.set_yticks(lim)

    ax.set_title(titles[i] + ' networks')

cbar.set_label('Change in the connection', fontsize=6)
cbar.set_ticks([-1, 0, 1])


panel_label(axes[0, 0], 'a', x=-.35)
panel_label(axes[1, 0], 'b', x=-.42)

filename = 'sequence_networks_connections'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
