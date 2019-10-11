# -*- coding: utf-8 -*-
#
# panel_label.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License


def panel_label(ax, label, x=-0.4, y=1.0):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')
