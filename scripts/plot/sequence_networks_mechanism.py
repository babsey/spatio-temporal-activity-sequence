# -*- coding: utf-8 -*-
#
# plot_sequence_networks_mechanism.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License


import numpy as np
import scipy.io as sio
import matplotlib as mpl
import pylab as pl

from lib.circular_colormap import gen_circular_cmap
from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.ax_spines as ax_spines
set_fontsize()


def mean_wrap(d, size):
    if np.std(d) < (size / 10):
        return np.mean(d)
    else:
        return (np.mean((np.array(d) - size / 2) % size) + size / 2) % size


def diff_wrap(d, size):
    if np.std(np.diff(d)) < (size / 10):
        return np.abs(np.diff(d[[0, -1]]))
    else:
        return np.abs(np.diff((np.array(d[[0, -1]]) - size / 2) % size))


def get_targets(W, nx, ny, start_n_id):

    distance = []
    size = []
    size_dev = []
    targets = np.zeros([nrow, ncol]) * np.nan

    for ii in range(nx):
        for jj in range(ny):
            post_ids = start_n_id[ii, jj]
            tgts = []
            tgts_dev = []
            centroids = []
            for nn in range(steps):
                sel_id = W[:, post_ids]

                all_targets = []
                for a1 in range(len(post_ids)):
                    ref_ids = np.where(sel_id[:, a1] > 0)[0]
                    all_targets.extend(ref_ids)

                neuron_hist = np.histogram(all_targets, bins=ed_deg)[0]

                num_shared_targets = np.unique(neuron_hist)[::-1]
                post_ids = []
                for ntargets in num_shared_targets:
                    most_correlated_targets = np.where(neuron_hist == ntargets)[0]
                    if len(post_ids) + len(most_correlated_targets) <= no_neuron:
                        post_ids.extend(most_correlated_targets)
                    else:
                        np.random.shuffle(most_correlated_targets)
                        post_ids.extend(most_correlated_targets[:no_neuron - len(post_ids)])

                if len(post_ids) == 0:
                    break
                post_ids = np.array(post_ids)
                targets[post_ids % nrow, post_ids // ncol] = nn
                tgts.extend(post_ids)
                tgts_dev.append(len(np.unique(tgts)))

                px, py = post_ids // nrow, post_ids % ncol
                mx = mean_wrap(px, nrow)
                my = mean_wrap(py, nrow)
                centroids.append((mx, my))

            cx, cy = np.array(centroids).T
            dx = diff_wrap(cx, nrow)
            dy = diff_wrap(cy, nrow)
            d = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
            distance.append(d[0])
            size.append(len(np.unique(tgts)))
            size_dev.append(tgts_dev)

    return targets, size, distance, size_dev


landscapes = ['symmetric', 'random', 'Perlin_uniform', 'homogeneous']
steps = 50
cmap = gen_circular_cmap()
colors = [cmap(1. * i / steps) for i in range(steps)]

nx = 8
no_neuron = nx * nx

no_clusters = 100

####################################
# I networks
####################################

nrow = ncol = 100
npop = nrow * ncol
pop = np.arange(npop)
x = pop % nrow
y = pop // ncol
center = ncol * nrow / 2 + nrow / 2

x_loc = [25, 75]
y_loc = [25, 75]
start_n_id = np.zeros([len(x_loc), len(y_loc), no_neuron], dtype=int)
for ii in range(len(x_loc)):
    for jj in range(len(y_loc)):
        ini_id = nrow * x_loc[ii] + y_loc[jj]
        post_id = []
        for ww in range(nx):
            new_ids = np.arange(ini_id, ini_id + nx) + nrow * ww
            post_id.extend(new_ids)

        start_n_id[ii, jj] = post_id

ed_deg = range(npop + 1)

targetsI = []
for lidx, landscape in enumerate(landscapes):
    W = sio.loadmat('Data/I_networks_%s_1.mat' % landscape)['W']
    targets, size, distance, size_dev = get_targets(W, len(x_loc), len(y_loc), start_n_id)
    targetsI.append(targets)

start_n_id = []
for ini_id in np.random.randint(0, nrow * ncol, no_clusters):
    post_id = []
    for ww in range(nx):
        new_ids = np.arange(ini_id, ini_id + nx) + nrow * ww
        post_id.extend(new_ids)

    post_id = np.array(post_id)
    new_x, new_y = post_id % nrow, post_id // nrow
    new_y = new_y % nrow
    post_id = new_x + nrow * new_y
    start_n_id.append(post_id)

start_n_id = np.array(start_n_id)
start_n_id = start_n_id.reshape(10, 10, -1)

distanceI = []
sizeI = []
sizeI_dev = []
for lidx, landscape in enumerate(landscapes):
    W = sio.loadmat('Data/I_networks_%s_1.mat' % landscape)['W']
    targets, size, distance, size_dev = get_targets(W, 10, 10, start_n_id)
    distanceI.append(distance)
    sizeI.append(size)
    sizeI_dev.append(size_dev)


####################################
# EI networks
####################################

nrow = ncol = 120
npop = nrow * ncol
pop = np.arange(npop)
x = pop % nrow
y = pop // ncol
center = ncol * nrow / 2 + nrow / 2

x_loc = [20, 60, 100]
y_loc = [20, 60, 100]
start_n_id = np.zeros([len(x_loc), len(y_loc), no_neuron], dtype=int)

targetsEI = []
distanceEI = []
sizeEI = []
sizeEI_dev = []

for ii in range(len(x_loc)):
    for jj in range(len(y_loc)):
        ini_id = nrow * x_loc[ii] + y_loc[jj]
        post_id = []
        for ww in range(nx):
            new_ids = np.arange(ini_id, ini_id + nx) + nrow * ww
            post_id.extend(new_ids)

        start_n_id[ii, jj] = post_id

ed_deg = range(npop + 1)
for lidx, landscape in enumerate(landscapes):
    W = sio.loadmat('Data/EI_networks_%s_1.mat' % landscape)['W_EE']
    targets, size, distance, size_dev = get_targets(W, len(x_loc), len(y_loc), start_n_id)
    targetsEI.append(targets)


start_n_id = []
for ini_id in np.random.randint(0, nrow * ncol, no_clusters):
    post_id = []
    for ww in range(nx):
        new_ids = np.arange(ini_id, ini_id + nx) + nrow * ww
        post_id.extend(new_ids)

    post_id = np.array(post_id)
    new_x, new_y = post_id % nrow, post_id // nrow
    new_y = new_y % nrow
    post_id = new_x + nrow * new_y
    start_n_id.append(post_id)

start_n_id = np.array(start_n_id)
start_n_id = start_n_id.reshape(10, 10, -1)

distanceEI = []
sizeEI = []
sizeEI_dev = []
for lidx, landscape in enumerate(landscapes):
    W = sio.loadmat('Data/EI_networks_%s_1.mat' % landscape)['W_EE']
    targets, size, distance, size_dev = get_targets(W, 10, 10, start_n_id)
    distanceEI.append(distance)
    sizeEI.append(size)
    sizeEI_dev.append(size_dev)


#########################
# Plotting
#########################

fig, axes = pl.subplots(3, len(landscapes), dpi=300)
fig.subplots_adjust(wspace=.1, hspace=.4, bottom=.05, left=.1, right=.9, top=.95)

for i in range(4):
    axes[0, i].remove()

ax1 = pl.subplot2grid((3, 9), (0, 0), colspan=3, fig=fig)
ax2 = pl.subplot2grid((3, 9), (0, 4), colspan=3, fig=fig)
ms = [4, 4, 4, 2]
n = 1000

dots_colors = pl.rcParams['axes.prop_cycle'].by_key()['color']
for idx, landscape in enumerate(landscapes[::-1]):
    W = sio.loadmat('Data/I_networks_%s_1.mat' % landscape)['W']
    w, v = np.linalg.eig(-1. * W[:n, :n])
    ax1.plot(w.real, w.imag, '.', label=landscape.split(
        '_')[0].capitalize(), color=dots_colors[3 - idx], ms=2)

landscapes = ['symmetric', 'random', 'Perlin_uniform', 'homogeneous']

for ii in range(4):
    landscape = landscapes[ii]
    print(landscape)
    ax2.plot(distanceI[3 - ii], sizeI[3 - ii], '.', color=dots_colors[3 - ii], ms=2)
    ax2.plot(distanceEI[3 - ii], sizeEI[3 - ii], 'x', color=dots_colors[3 - ii], ms=3)

    ax = axes[1, ii]
    ax.matshow(targetsI[ii], cmap=cmap, origin='bottom')
    ax.xaxis.tick_bottom()

    ax_spines.set_default(ax)
    ax.locator_params(nbins=3)
    if ii != 0:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    label = landscape.split('_')[0].capitalize()
    ax.text(0.08, 0.92, label, verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, fontsize=8, bbox={'facecolor': 'white', 'pad': 3})

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = axes[2, ii]
    ax.matshow(targetsEI[ii], cmap=cmap, origin='bottom')

    ax.set_xticks([0, nrow / 2, nrow])
    ax.set_yticks([0, nrow / 2, nrow])
    ax.xaxis.tick_bottom()

    ax_spines.set_default(ax)
    ax.locator_params(nbins=4)

    if ii != 0:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)

handles, labels = ax1.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1],
           loc='upper right', bbox_to_anchor=(1.9, 1), fontsize=8)

ax1.set_title('')
ax1.set_xlabel('Real part')
ax1.set_ylabel('Imaginary part')
ax_spines.set_default(ax1)

ax2.set_xlabel('Effective length')
ax2.set_ylabel('Number of neurons')
ax2.set_ylim([0, 1600])
ax_spines.set_default(ax2)
ax2.locator_params(nbins=5)

panel_label(ax1, 'a', x=-.35)
panel_label(ax2, 'b', x=-.35)
panel_label(axes[1, 0], 'c', x=-.55)
panel_label(axes[2, 0], 'd', x=-.55)

filename = 'sequence_networks_mechanism'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')
