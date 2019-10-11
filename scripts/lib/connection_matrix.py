# -*- coding: utf-8 -*-
#
# connection_matrix.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np

import lib.lcrn_network as lcrn
import lib.connectivity_landscape as cl


def I_networks(landscape, nrow, ncol, ncon, kappa, theta, shift=0, seed=0, **kwargs):
    np.random.seed(seed)
    npop = nrow * ncol

    if landscape['mode'] != 'symmetric':
        move = cl.move(nrow)
        ll = cl.__dict__[landscape['mode']](nrow, landscape.get('specs', {}))

    conmat = []
    for ii in range(npop):
        targets, delay = lcrn.lcrn_gamma_targets(
            ii, nrow, ncol, nrow, ncol, ncon, kappa, theta)
        if landscape['mode'] != 'symmetric':          # asymmetry
            targets = (targets + shift * move[ll[ii] % len(move)]) % npop
        targets = targets[targets != ii]            # no selfconnections
        hist_targets = np.histogram(targets, bins=range(npop + 1))[0]
        conmat.append(hist_targets)

    return np.array(conmat)


def EI_networks(landscape, nrowE, ncolE, nrowI, ncolI, p, stdE, stdI, shift=0, seed=0, **kwargs):
    np.random.seed(seed)
    npopE = nrowE * ncolE
    npopI = nrowI * ncolI

    if landscape['mode'] != 'symmetric':
        move = cl.move(nrowE)
        ll = cl.__dict__[landscape['mode']](nrowE, landscape.get('specs', {}))

    conmatEE, conmatEI, conmatIE, conmatII = [], [], [], []
    for idx in range(npopE):
        # E -> E
        source = idx, nrowE, ncolE, nrowE, ncolE, int(p * npopE), stdE, False
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        if landscape['mode'] != 'symmetric':  # asymmetry
            targets = (targets + shift * move[ll[idx] % len(move)]) % npopE
        targets = targets[targets != idx]
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatEE.append(hist_targets)

        # E -> I
        source = idx, nrowE, ncolE, nrowI, ncolI, int(p * npopI), stdI, False
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatEI.append(hist_targets)

    for idx in range(npopI):
        # I -> E
        source = idx, nrowI, ncolI, nrowE, ncolE, int(p * npopE), stdE, False
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatIE.append(hist_targets)

        # I -> I
        source = idx, nrowI, ncolI, nrowI, ncolI, int(p * npopI), stdI, False
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        targets = targets[targets != idx]
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatII.append(hist_targets)

    return np.array(conmatEE), np.array(conmatEI), np.array(conmatIE), np.array(conmatII)
