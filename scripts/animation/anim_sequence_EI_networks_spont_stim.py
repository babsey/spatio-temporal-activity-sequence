# -*- coding: utf-8 -*-
#
# anim_sequence_EI_networks_spont_stim.py
#
# Copyright 2019 Sebastian Spreizer
# The MIT License

import numpy as np
import pylab as pl
import lib.protocol as protocol
import lib.animation_image as ai
import datetime

landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
    {'mode': 'random'},
    {'mode': 'Perlin', 'specs': {'size': 4}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 4}},
]


simulation = 'sequence_EI_networks'
params = protocol.get_parameters(simulation).as_dict()
params.update({'landscape': landscapes[-1]})

gids, ts = protocol.get_or_simulate(simulation, params)

nrow = ncol = params['nrowE']
npop = nrow * ncol
offset = 1

idx = gids - offset < npop
gids, ts = gids[idx], ts[idx]

ts_bins = np.arange(500., 2500., 10.)
h = np.histogram2d(ts, gids - offset, bins=[ts_bins, range(npop + 1)])[0]
hh = h.reshape(-1, nrow, ncol)

simulation = 'sequence_EI_networks_stim'
params = protocol.get_parameters(simulation).as_dict()
params.update({'landscale': landscapes[-1]})

gids, ts = protocol.get_or_simulate(simulation, params)

nrow = ncol = params['nrowE']
npop = nrow * ncol
offset = 1

idx = gids - offset < npop
gids, ts = gids[idx], ts[idx]

ts_bins = np.arange(500., 2500., 10.)
h = np.histogram2d(ts, gids - offset, bins=[ts_bins, range(npop + 1)])[0]
hh_stim = h.reshape(-1, nrow, ncol)

fig, ax = pl.subplots(1,2)
a = ai.multiple_animate_image(fig, ax, [hh,hh_stim], vmin=0, vmax=np.max([hh,hh_stim]), cmap='binary')
ax[0].set_title('Spontaneous')
ax[1].set_title('Evoked')
date = datetime.datetime.now()
a.save('sequence_EI_networks_spont_stim-%s.mp4' %(date), fps=12, extra_args=['-vcodec', 'libx264'])
pl.show()
