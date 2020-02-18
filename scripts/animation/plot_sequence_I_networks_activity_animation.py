# -*- coding: utf-8 -*-
#
# plot_sequence_I_networks_activity_antimation.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import pylab as pl
from matplotlib import animation

import lib.protocol as protocol
from lib.dbscan_cluster import detect_wrap
import lib.ax_spines as ax_spines
from lib.circular_colormap import gen_circular_cmap


landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'random'},
    # {'mode': 'Perlin', 'specs': {'size': 3}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 3}},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
]

simulation, nrow, ncol = 'sequence_I_networks', 100, 100
npop = nrow * ncol

params = protocol.get_parameters(simulation).as_dict()

tmax = 2000.
dt = 100.
ts_bins = np.arange(1000., tmax, 10.)
steps = len(ts_bins)

td = 30
eps = 3

t,d,c = [],[],[]

for idx, landscape in enumerate(landscapes):
    params['landscape'] = landscape
    gids, ts = protocol.get_or_simulate(simulation, params)
    gids -= 1
    ts, gids = ts[gids < npop], gids[gids < npop]
    tidx = (1000. <= ts) * (ts < tmax+dt)
    gidx = (gids // nrow < 100) * (gids % nrow < 100)
    ts, gids = ts[gidx * tidx], gids[gidx * tidx]
    nclusters, clusters = detect_wrap(ts, gids, nrow, ncol, td=td, eps=eps)

    t.append(ts)
    d.append(np.array([gids % nrow, gids // nrow]).T)
    c.append(clusters.astype(float)/(nclusters-1))

steps = ts_bins
width = 50.

fig, axes = pl.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle('Activity sequence in I networks')
axes = np.concatenate(axes)

ax = axes[0]
ax.set_xlim(0,100)
ax.set_ylim(0,100)

ax = axes[2]
ax.set_xlabel('x')
ax.set_ylabel('y')

cmap = gen_circular_cmap()

s = []
for idx in range(len(landscapes)):
    ax = axes[idx]
    landscape = landscapes[idx]
    label = landscapes[idx]['mode'].split('_')[0].capitalize()
    ax.text(0.05, 1.05, label,
       verticalalignment='top', horizontalalignment='left',
       transform=ax.transAxes, fontsize=9,
       bbox={'facecolor': 'white', 'pad': 3})

    scat = ax.scatter([], [], s=1, marker='s')
    s.append(scat)

def animate(ii):
    for idx in range(len(landscapes)):
        tidx = (t[idx]>steps[ii]) * (t[idx]<=steps[ii]+width)
        scat = s[idx]
        scat.set_offsets(d[idx][tidx])
        scat.set_array(c[idx][tidx])
        scat.set_clim(0,1)
        scat.set_cmap(cmap)
    return s

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=len(steps))
anim.save('sequence_I_networks_activity.mp4', fps=12, extra_args=['-vcodec', 'libx264'])

# pl.show()
