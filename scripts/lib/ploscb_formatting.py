# -*- coding: utf-8 -*-
#
# ploscb_formatting.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import matplotlib.pyplot as plt


def set_fontsize(sm=6, md=8, lg=10):

    plt.rc('font', size=sm)          # controls default text sizes
    plt.rc('axes', titlesize=md)     # fontsize of the axes title
    plt.rc('axes', labelsize=md)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sm)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sm)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sm)    # legend fontsize
    plt.rc('figure', titlesize=lg)  # fontsize of the figure title
