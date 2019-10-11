# -*- coding: utf-8 -*-
#
# sequence_I_networks.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

"""
Script for NEST simulation of purely inhibitory network model to produce activity sequences.
"""

import sys
import numpy as np
import nest
import pylab as pl

import lib.lcrn_network as lcrn
import lib.connectivity_landscape as cl

###################### Build parameters from file #############################

from sumatra.parameters import build_parameters
parameter_file = sys.argv[1]
output_file = parameter_file.split('/')[-1].split('.')[0]
p = build_parameters(parameter_file)
p.__dict__.update(p.as_dict())
print(p.pretty())

###############################################################################


"""
Set Kernel Status
"""

np.random.seed(p.seed)
nest.ResetKernel()
nest.SetKernelStatus({
    'local_num_threads': p.local_num_threads,
    'resolution': p.resolution,
    'data_path': './Data',
    'overwrite_files': True,
})

"""
Create nodes
"""

# Network size
npop = p.nrow * p.ncol                 # amount of neurons in population

# Create neurons
neuron = p.neuron
neuron['n'] = npop
pop = nest.Create(**neuron)

# Create input devices
noise = nest.Create('noise_generator')

# Create recording devices
sd = nest.Create('spike_detector', params={
    'start':        p.recstart,
    'to_file':      True,
    'label':        output_file,
})


"""
Connect nodes
"""

landscape = cl.__dict__[p.landscape['mode']](
    p.nrow, p.landscape.get('specs', {}))
move = cl.move(p.nrow)
offset = pop[0]

# Connect neurons to neurons
for ii in range(npop):
    source = ii, p.nrow, p.ncol, p.nrow, p.ncol, p.ncon, p.kappa, p.theta
    targets, delay = lcrn.lcrn_gamma_targets(*source)
    if landscape is not None:          # asymmetry
        targets = (targets + p.shift * move[landscape[ii] % len(move)]) % npop
    # no selfconnections
    targets = targets[targets != ii]
    nest.Connect([pop[ii]], (targets + offset).tolist(),
                 syn_spec={'weight': p.Ji})

# Connect noise input device to all neurons
nest.Connect(noise, pop)

# Connect spike detector to population of all neurons
nest.Connect(pop, sd)


"""
Warming up
"""

# Preparation for simulation
nest.SetStatus(noise, params={'std': 500.})
nest.Simulate(p.wuptime / 2)
nest.SetStatus(noise, params=p.noise)
nest.Simulate(p.wuptime / 2)

"""
Start simulation
"""

# Run simulation
nest.Simulate(p.simtime)
