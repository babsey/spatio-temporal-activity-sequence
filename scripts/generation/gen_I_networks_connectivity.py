# -*- coding: utf-8 -*-
#
# gen_I_networks_connectivity.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import scipy.io as sio

import lib.connection_matrix as cm
import lib.protocol as protocol


landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'random', 'specs': {'seed': 0}},
    {'mode': 'Perlin', 'specs': {'size': 4}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 4}},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
]

simulation = 'sequence_I_networks'
params = protocol.get_parameters(simulation).as_dict()

for landscape in landscapes:
    params['landscape'] = landscape
    params['seed'] = params['shift']
    print(landscape['mode'],params['shift'])
    W = cm.I_networks(**params)
    mdict = {'W': W.astype(int)}
    sio.savemat('Data/I_networks_%s_%s.mat' %(params['landscape']['mode'],params['shift']), mdict, do_compression=True)

# params['landscape'] = landscapes[0]
# W = cm.I_networks(**params)
# mdict = {'W': W.astype(int)}
# sio.savemat('conmat/I_networks_%s_ref.mat' %(params['landscape']['mode']), mdict, do_compression=True)
