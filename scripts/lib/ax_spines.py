# -*- coding: utf-8 -*-
#
# ax_spines.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License


def set_default(ax):
    set_visible(ax, ['bottom', 'left'])

def set_visible(ax, sides):
    all_sides = ax.spines.keys()
    for side in all_sides:
        ax.spines[side].set_visible(side in sides)

def set_invisible(ax, sides):
    all_sides = ax.spines.keys()
    for side in all_sides:
        ax.spines[side].set_visible(side not in sides)
