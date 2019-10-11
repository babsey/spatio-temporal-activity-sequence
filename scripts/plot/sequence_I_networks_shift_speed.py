# -*- coding: utf-8 -*-
#
# plot_sequence_I_networks_shift_speed.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib as mpl
import pylab as pl

from lib.panel_label import panel_label
from lib.ploscb_formatting import set_fontsize
import lib.activity_sequence as seq
import lib.ax_spines as ax_spines
import lib.protocol as protocol
set_fontsize()


simulation = 'sequence_I_networks'
params = protocol.get_parameters(simulation).as_dict()

nrow, ncol = params['nrow'], params['ncol']
npop = nrow * ncol

steps = 10
width = 25
td = 20
eps = 3

landscapes = [
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 3}},
]

fig, axes = pl.subplots(len(landscapes), sharex=True, sharey=True, figsize=(4.8, 3.6), dpi=300)

for idx, landscape in enumerate(landscapes):

    params['landscape'] = landscape
    ax = axes[idx]
    ax.set_title(landscape['mode'].split('_')[0].capitalize())

    for shift in range(3):
        params['shift'] = shift
        gids, ts = protocol.get_or_simulate(simulation, params)

        clusters, sequences = seq.identify_vectors(ts, gids - 1, nrow, ncol, steps=steps, width=width, td=td, eps=eps)
        t, x, y, c, a, s = sequences

        speed = s[~np.isnan(s)]
        bins = np.linspace(0, 4, 51)
        hs = np.histogram(speed, bins=bins)[0]
        ax.plot(bins[:-1] / 25. * 1000., hs / float(np.sum(hs)), label='%s' % shift)
        ax_spines.set_default(ax)
        ax.locator_params(nbins=4)


ax = axes[-1]
ax.set_ylabel('Probability')
ax.set_xlabel('Velocity [grid points / sec]')
ax.legend(title='Shift [grid points]')

panel_label(axes[0], 'a', x=-.15)
panel_label(axes[1], 'b', x=-.15)

filename = 'sequence_I_networks_shift_speed'
fig.savefig(filename + '.png', format='png', dpi=300)
fig.savefig(filename + '.pdf', format='pdf')

pl.show()
