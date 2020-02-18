# -*- coding: utf-8 -*-
#
# anim_sequence_I_networks.py
#
# Copyright 2017 Sebastian Spreizer
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
    {'mode': 'Perlin', 'specs': {'size': 3}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 3}},
]


simulation = 'sequence_I_networks'
params = protocol.get_parameters(simulation).as_dict()
params.update({'landscape': landscapes[-1]})

gids, ts = protocol.get_or_simulate(simulation, params)

nrow = ncol = params['nrow']
npop = nrow * ncol
offset = 1

idx = gids - offset < npop
gids, ts = gids[idx], ts[idx]

ts_bins = np.arange(1000., 6000., 50.)
h = np.histogram2d(ts, gids - offset, bins=[ts_bins, range(npop + 1)])[0]
hh = h.reshape(-1, nrow, ncol)

fig, ax = pl.subplots(1)
a = ai.animate_image(ax, hh, vmin=0, vmax=np.max(hh))
date = datetime.datetime.now()
a.save('%s-%s.mp4' %(simulation, date), extra_args=['-vcodec', 'libx264'])
pl.show()
