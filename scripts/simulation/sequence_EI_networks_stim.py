# -*- coding: utf-8 -*-
#
# sequence_EI_networks_stim.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

"""
Script for NEST simulation of EI network models to produce activity sequences.
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

npopE = p.nrowE * p.ncolE
npopI = p.nrowI * p.ncolI

# Create neurons
neuronE = p.neuron
neuronE['n'] = npopE
popE = nest.Create(**neuronE)

neuronI = p.neuron
neuronI['n'] = npopI
popI = nest.Create(**neuronI)

pop = popE + popI

# Create input devices
noiseE = nest.Create('noise_generator')
noiseI = nest.Create('noise_generator')
noise = noiseE + noiseI

stim = nest.Create('dc_generator')

# Create recording devices
sd = nest.Create('spike_detector', params={
    'start':        p.recstart,
    'to_file':      True,
    'label':        output_file,
})


"""
Connect nodes
"""

landscape = cl.__dict__[p.landscape['mode']](p.nrowE, p.landscape.get('specs', {}))
move = cl.move(p.nrowE)
offsetE = popE[0]
offsetI = popI[0]

for idx in range(npopE):
    kwargs = {'syn_spec': {'weight': p.Ji}}

    # E-> E
    source = idx, p.nrowE, p.ncolE, p.nrowE, p.ncolE, int(p.p * npopE), p.stdE, p.selfcon
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    if landscape is not None:        #  asymmetry
        targets = (targets +  p.shift * move[landscape[idx] % len(move)]) % npopE
    targets = targets[targets != idx]
    nest.Connect([popE[idx]], (targets + offsetE).tolist(), **kwargs)

    # E-> I
    source = idx, p.nrowE, p.ncolE, p.nrowI, p.ncolI, int(p.p * npopI), p.stdE / 2, p.selfcon
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    nest.Connect([popE[idx]], (targets + offsetI).tolist(), **kwargs)

for idx in range(npopI):
    kwargs = {'syn_spec': {'weight': p.g * -p.Ji}}

    # I-> E
    source = idx, p.nrowI, p.ncolI, p.nrowE, p.ncolE, int(p.p * npopE), p.stdI, p.selfcon
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    nest.Connect([popI[idx]], (targets + offsetE).tolist(), **kwargs)

    # I-> I
    source = idx, p.nrowI, p.ncolI, p.nrowI, p.ncolI, int(p.p * npopI), p.stdI / 2, p.selfcon
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    targets = targets[targets != idx]
    nest.Connect([popI[idx]], (targets + offsetI).tolist(), **kwargs)

# Connect noise input device to all neurons
nest.Connect(noiseE, popE)
nest.Connect(noiseI, popI)

# Stimulate centered area
centerE = (p.nrowE * (p.ncolE + 1)) / 2
source = centerE, p.nrowE, p.ncolE, p.nrowE, p.ncolE, p.stim['ncon'], p.stim['std']
targets, delay = lcrn.lcrn_gauss_targets(*source)
targets = np.unique(targets)
nest.Connect(stim, (targets + offsetE).tolist())

# Connect spike detector to population of all neurons
nest.Connect(pop, sd)


"""
Warming up
"""

nest.SetStatus(noise, params=p.noiseWUP)
nest.Simulate(p.wuptime/2)
nest.SetStatus(noiseE, params=p.noiseE)
nest.SetStatus(noiseI, params=p.noiseI)
nest.Simulate(p.wuptime/2)


"""
Start simulation
"""

for i in range(p.stim['nseries']):
    nest.SetStatus(
        stim, params={'amplitude': p.stim['amplitude']})
    nest.Simulate(p.stim['duration'])
    nest.SetStatus(stim, params={'amplitude': 0.})
    nest.Simulate(p.stim['pause'])
