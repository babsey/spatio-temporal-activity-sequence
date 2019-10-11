# -*- coding: utf-8 -*-
#
# plot_sequence_EI_networks_spectrogram.py
#
# Copyright 2019 Sebastian Spreizer
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


wuptime = 500.
simtime = 5000.
dt = 5.
ts_bins = np.arange(wuptime, wuptime + simtime, dt)
fs = int(1000. / dt)
nfft = 2**12

simulation = 'sequence_EI_networks'

params = protocol.get_parameters(simulation).as_dict()
nrow, ncol = params['nrowE'], params['ncolE']
npop = nrow * ncol

params['landscape'] = {'mode': 'Perlin_uniform', 'specs': {'size': 3}}


params = protocol.get_parameters(simulation).as_dict()
gids, ts = protocol.get_or_simulate(simulation, params)
sidx = ((gids - 1) < (npop)) * (ts > (wuptime))
gids, ts = gids[sidx], ts[sidx]

spike_count = np.histogram(ts, ts_bins)[0]
ps_epoch = []
for ii in range(50):
    epoch = spike_count[ii*10:ii*10 + 40]
    z_score = (epoch - np.mean(epoch)) / np.std(epoch)
    x, y = signal.welch(z_score, fs=fs, nfft=nfft)
    ps_epoch.append(y)


params['Ji'] = 20.
gids, ts = protocol.get_or_simulate(simulation, params)
sidx = ((gids - 1) < (npop)) * (ts > (wuptime))
gids, ts = gids[sidx], ts[sidx]

spike_count = np.histogram(ts, ts_bins)[0]
ps_epoch1 = []
for ii in range(50):
    epoch = spike_count[ii*10:ii*10 + 40]
    z_score = (epoch - np.mean(epoch)) / np.std(epoch)
    x, y = signal.welch(z_score, fs=fs, nfft=nfft)
    ps_epoch1.append(y)



fig,axes = pl.subplots(2,2, dpi=300)
fig.suptitle('Spectrogram')

y = ps_epoch
# y = 10*np.log10(y)
ax = axes[0,0]
ax.matshow(y, aspect='auto')
ax_spines.set_default(ax)
ax.xaxis.set_ticklabels([])
ax.set_ylabel('Epoch')
ax.set_title('Ji: 10pA')

ax = axes[1,0]
for ps in ps_epoch:
    y = ps
    # y = 10*np.log10(y)
    ax.plot(x, y, color='black', lw=1, alpha=.2)
y = np.nanmean(ps_epoch,0)
# y= 10*np.log10(y)
ax.plot(x, y, color='red')
ax.set_xlim(0,100)
ax_spines.set_default(ax)
ax.set_ylabel('Power')
ax.set_xlabel('Frequency [Hz]')

y = ps_epoch1
# y = 10*np.log10(y)
ax = axes[0,1]
ax.set_title('Ji: 20pA')
ax.matshow(y, aspect='auto')
ax_spines.set_default(ax)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

ax = axes[1,1]
for ps in ps_epoch1:
    y = ps
    # y = 10*np.log10(y)
    ax.plot(x, y, color='black', lw=1, alpha=.2)
y = np.nanmean(ps_epoch1,0)
# y= 10*np.log10(y)
ax.plot(x, y, color='red', lw=1)
ax.set_xlim(0,100)
ax_spines.set_default(ax)
ax.set_xlabel('Frequency [Hz]')

axes[1,0].set_xticks([0,50,100])
axes[1,1].set_xticks([0,50,100])

axes[1,0].set_yticks([0.,0.05,.1])
axes[1,1].set_yticks([0.,0.05,.1])


panel_label(axes[0,0], 'a', x=-.3)
panel_label(axes[0,1], 'b', x=-.15)
panel_label(axes[1,0], 'c', x=-.3)
panel_label(axes[1,1], 'd', x=-.15)

filename = 'sequence_EI_networks_spectrogram'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
