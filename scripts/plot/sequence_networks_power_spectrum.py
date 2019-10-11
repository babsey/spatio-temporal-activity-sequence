# -*- coding: utf-8 -*-
#
# plot_sequence_networks_power_spectrum.py
#
# Copyright 2018 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib as mpl
import pylab as pl
from scipy import signal

from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.ax_spines as ax_spines
import lib.protocol as protocol
set_fontsize()


def power_spectrum(data):
    spike_count = np.histogram(data, ts_bins)[0]
    ps_epoch = []
    for ii in range(20):
        epoch = spike_count[ii*10:ii*10 + 40]
        z_score = (epoch - np.mean(epoch)) / np.std(epoch)
        x, y = signal.welch(z_score, fs=fs, nfft=nfft)
        ps_epoch.append(y)
    return x, np.nanmean(ps_epoch, 0)


def neurons_of_interest(gids, xlim, ylim):
    return (gids % nrow >= xlim[0]) * (gids // nrow >= ylim[0]) * (gids % ncol < xlim[1]) * (gids // ncol < ylim[1])


np.random.seed(0)

landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'random'},
    {'mode': 'Perlin_uniform', 'specs': {'size': 3}},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
]

wuptime = 500.
simtime = 5000.
dt = 5.
ts_bins = np.arange(wuptime, wuptime + simtime, dt)
fs = int(1000 / dt)
nfft = 2**12

neuron_box_pts = 10


# I networks

simulation = 'sequence_I_networks'
params = protocol.get_parameters(simulation).as_dict()

nrow, ncol = params['nrow'], params['ncol']
npop = nrow * ncol

ps_I = []
for axIdx, landscape in enumerate(landscapes):
    params['landscape'] = landscape
    gids, ts = protocol.get_or_simulate(simulation, params)
    sidx = ((gids - 1) < (npop)) * (ts > (wuptime))
    gids, ts = gids[sidx], ts[sidx]
    pop = np.unique(gids)

    # Total
    x, y = power_spectrum(ts)
    ps_total = y

    # Random
    ps = []
    for i in range(20):
        np.random.shuffle(pop)
        idx = np.in1d(gids, pop[:np.power(neuron_box_pts, 2)])
        x, y = power_spectrum(ts[idx])
        ps.append(y)
    ps_random = np.nanmean(ps, 0)

    # Local
    ps = []
    for i in range(20):
        xRand = np.random.randint(0, 100 - neuron_box_pts)
        yRand = np.random.randint(0, 100 - neuron_box_pts)
        idx = neurons_of_interest(gids - 1, [xRand, xRand + neuron_box_pts], [yRand, yRand + neuron_box_pts])
        x, y = power_spectrum(ts[idx])
        ps.append(y)
    ps_local = np.nanmean(ps, 0)


    ps_I.append([ps_total, ps_random, ps_local])


# EI networks

simulation = 'sequence_EI_networks'
params = protocol.get_parameters(simulation).as_dict()

nrow, ncol = params['nrowE'], params['ncolE']
npop = nrow * ncol

ps_EI = []
for axIdx, landscape in enumerate(landscapes):
    params['landscape'] = landscape
    gids, ts = protocol.get_or_simulate(simulation, params)
    sidx = ((gids - 1) < (npop)) * (ts > (wuptime))
    gids, ts = gids[sidx], ts[sidx]
    pop = np.unique(gids)

    # Total
    x, y = power_spectrum(ts)
    ps_total = y

    # Random
    ps = []
    for i in range(20):
        np.random.shuffle(pop)
        idx = np.in1d(gids, pop[:np.power(neuron_box_pts, 2)])
        x, y = power_spectrum(ts[idx])
        ps.append(y)
    ps_random = np.nanmean(ps, 0)

    # Local
    ps = []
    for i in range(20):
        xRand = np.random.randint(0, 100 - neuron_box_pts)
        yRand = np.random.randint(0, 100 - neuron_box_pts)
        idx = neurons_of_interest(gids - 1, [xRand, xRand + neuron_box_pts], [yRand, yRand + neuron_box_pts])
        x, y = power_spectrum(ts[idx])
        ps.append(y)
    ps_local = np.nanmean(ps, 0)


    ps_EI.append([ps_total, ps_random, ps_local])

landscapes = [l['mode'] for l in landscapes]




fig, axes = pl.subplots(2, 5, sharex=True, dpi=300)
fig.subplots_adjust(wspace=.2, left=.1, right=1., top=.8, bottom=.2)
fig.set_size_inches(6.4, 3.8)
fig.suptitle('Power spectra of network activity')

for axIdx, landscape in enumerate(landscapes):
    ax = axes[0, axIdx]
    ax.set_title(landscape.split('_')[0].capitalize())

    for ii in range(3):
        y = 10*np.log10(ps_I[axIdx][ii])
        ax.plot(x, y, zorder=3-ii, lw=1)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_spines.set_default(ax)

    ax = axes[1, axIdx]
    for ii in range(3):
        y = 10*np.log10(ps_EI[axIdx][ii])
        ax.plot(x, y, zorder=3-ii, lw=1)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_spines.set_default(ax)


axes[0, 0].text(0.07, 0.94, 'I networks',
                verticalalignment='top', horizontalalignment='left',
                transform=axes[0, 0].transAxes, fontsize=8,
                bbox={'facecolor': 'white', 'pad': 3})

axes[1, 0].text(0.07, 0.06, 'EI networks',
                verticalalignment='bottom', horizontalalignment='left',
                transform=axes[1, 0].transAxes, fontsize=8,
                bbox={'facecolor': 'white', 'pad': 3})

ax = axes[1, 0]
ax.set_xlim(0,100)
ax.set_xscale('symlog')
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))

for i in range(4):
    if i > 0:
        axes[0,i].yaxis.set_ticklabels([])
        axes[1,i].yaxis.set_ticklabels([])
    axes[1, i].set_xlabel('Frequency [Hz]')

for i in range(2):
    axes[i, 0].set_ylabel('Power [dB]')

axes[0, -1].set_visible(False)
axes[1, -1].set_visible(False)
axes[0, -2].legend(['Total', 'Random', 'Local'], loc='upper right', bbox_to_anchor=(2, 1))

panel_label(axes[0, 0], 'a', x=-.5)
panel_label(axes[1, 0], 'b', x=-.5)

filename = 'sequence_networks_power_spectrum'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
