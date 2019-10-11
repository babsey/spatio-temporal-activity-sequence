# -*- coding: utf-8 -*-
#
# plot_sequence_networks_schematics.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import pylab as pl
import matplotlib as mpl
from matplotlib.patches import Circle

from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.ax_spines as ax_spines
import lib.connection_matrix as cm
import lib.connectivity_landscape as cl
import lib.lcrn_network as lcrn
import lib.protocol as protocol
set_fontsize()


n = 9
src = 10 // 2 * 10
x = y = np.arange(5, 100, 10)
X, Y = np.meshgrid(x, y)
X, Y = X.ravel(), Y.ravel()

colors = pl.rcParams['axes.prop_cycle'].by_key()['color']
c0 = colors[0]
c1 = colors[2]
cs = colors[1]  # 'black'

subplot_kw = {'adjustable': 'box', 'aspect': 1.}
fig, axes = pl.subplots(2, 3, sharex=True, sharey=True, subplot_kw=subplot_kw, dpi=300, figsize=(5.6,5.6*.66))
fig.suptitle('Schematics of the asymmetric networks')


ax = axes[0, 0]
ax.plot(X, Y, 'o', ms=2, color='grey')
ax.plot(src, src, 'o', ms=4, color=cs)

ax.add_patch(Circle((src, src), 20, alpha=.3, color=c0))
ax.add_patch(Circle((src + 10, src + 10), 20, alpha=.3, color=c1))
ax.arrow(src + 2, src + 2, 4, 4, head_width=2, head_length=3, fc='k', ec='k')
ax_spines.set_default(ax)
panel_label(ax, 'a')

np.random.seed(0)

nrow = 100
ncol = 100
ncon = 10000
t = lcrn.lcrn_gamma_targets(5050, nrow, ncol, nrow, nrow, ncon, 4, 3)[0]

tx0 = t / nrow
ty0 = t % ncol

tx1 = tx0 + 5
ty1 = ty0 + 5

bins = np.arange(0, 101, 2)
htx0 = np.histogram(tx0, bins=bins)[0] / 50.
hty0 = np.histogram(ty0, bins=bins)[0] / 50.

htx1 = np.histogram(tx1, bins=bins)[0] / 50.
hty1 = np.histogram(ty1, bins=bins)[0] / 50.

ax = axes[0, 1]
ax.text(0.05, 0.95, 'Gamma distribution',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, fontsize=8,
        bbox={'facecolor': 'white', 'pad': 3})
ax.plot(tx0[::50], ty0[::50], '.', color=c0, ms=2)
ax.plot(tx1[::50], ty1[::50], '.', color=c1, ms=2)
ax.plot(50, 50, 'o', ms=4, color=cs)

ax.step(bins[:-1], htx0, color=c0,lw=1)
ax.step(hty0, bins[:-1], color=c0,lw=1)
ax.step(bins[:-1], htx1, color=c1,lw=1)
ax.step(hty1, bins[:-1], color=c1,lw=1)

ax.xaxis.set_ticks([0, 50, 100])
ax.yaxis.set_ticks([0, 50, 100])
ax.hlines(50, -0, 100, lw=1, color='grey')
ax.vlines(50, -0, 100, lw=1, color='grey')
ax_spines.set_default(ax)


n = 10000
px0 = np.random.normal(50, 8, n)
py0 = np.random.normal(50, 8, n)
px1 = np.random.normal(50 + 5, 8, n)
py1 = np.random.normal(50 + 5, 8, n)

hx0 = np.histogram(px0, bins=bins)[0] / 50.
hy0 = np.histogram(py0, bins=bins)[0] / 50.

hx1 = np.histogram(px1, bins=bins)[0] / 50.
hy1 = np.histogram(py1, bins=bins)[0] / 50.

ax = axes[0, 2]
ax.text(0.05, 0.95, 'Gaussian distribution',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, fontsize=8,
        bbox={'facecolor': 'white', 'pad': 3})
ax.plot(px0[::50], py0[::50], '.', color=c0, ms=2)
ax.plot(px1[::50], py1[::50], '.', color=c1, ms=2)
ax.plot(50, 50, 'o', ms=4, color=cs)

ax.step(bins[:-1], hx0, color=c0,lw=1)
ax.step(hy0, bins[:-1], color=c0,lw=1)
ax.step(bins[:-1], hx1, color=c1,lw=1)
ax.step(hy1, bins[:-1], color=c1,lw=1)

ax.xaxis.set_ticks([0, 50, 100])
ax.yaxis.set_ticks([0, 50, 100])
ax.hlines(50, -0, 100, lw=1, color='grey')
ax.vlines(50, -0, 100, lw=1, color='grey')
ax_spines.set_default(ax)


landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'random'},
    # {'mode': 'Perlin', 'specs': {'size': 4}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 4}},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
]

simulation = 'sequence_I_networks'
params = protocol.get_parameters(simulation)

nrow, ncol = params['nrow'], params['ncol']
npop = nrow * ncol
ncon = params['ncon']


size = 8
ax = axes[1, 0]
ax.text(0.05, 0.95, 'Random',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, fontsize=8,
        bbox={'facecolor': 'white', 'pad': 3})
size = 8
D = cl.random(size, specs={'seed': 9555})
U = np.cos(D / 8. * 2 * np.pi)
V = np.sin(D / 8. * 2 * np.pi)
ax.quiver(X, Y, U, V, pivot='middle')
ax_spines.set_default(ax)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = axes[1, 1]
ax.text(0.05, 0.95, 'Perlin',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, fontsize=8,
        bbox={'facecolor': 'white', 'pad': 3})
D = cl.Perlin_uniform(100, specs={'size': 8}).reshape(100, 100)[:8, :8].ravel()
U = np.cos(D / 8. * 2 * np.pi)
V = np.sin(D / 8. * 2 * np.pi)
ax.quiver(X, Y, U, V, pivot='middle')
ax_spines.set_default(ax)

ax = axes[1, 2]
ax.text(0.05, 0.95, 'Homogeneous',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes, fontsize=8,
        bbox={'facecolor': 'white', 'pad': 3})
D = cl.homogeneous(size, specs={'phi': 3})
U = np.cos(D / 8. * 2 * np.pi)
V = np.sin(D / 8. * 2 * np.pi)
ax.quiver(X, Y, U, V, pivot='middle')
ax_spines.set_default(ax)

panel_label(axes[0, 0], 'a')
panel_label(axes[1, 0], 'b')

filename = 'sequence_networks_schematics'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
