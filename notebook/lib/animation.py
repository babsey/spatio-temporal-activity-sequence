# -*- coding: utf-8 -*-
#
# animation.py
#
# Copyright 2019 Sebastian Spreizer
# The MIT License


from matplotlib import animation
from IPython.display import HTML
import numpy as np


def raster(fig, ax, dots, ts, x, y, ts_bins=[], dt=10.):

    def animate(ii):
        print(ii)
        idx = (ts >= ts_bins[ii]) * (ts < ts_bins[ii]+dt)
        dots.set_data(x[idx], y[idx])
        if len(ts_bins) > 0:
            ax.set_title('%s ms' % ts_bins[ii])
        else:
            ax.set_title('%s' % ii)
        return dots,

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=len(ts_bins), interval=50, blit=True)

    return anim


def scatter(fig, ax, scat, ts, x, y, c, ts_bins=[], dt=10.):

    def animate(ii):
        idx = (ts >= ts_bins[ii]) * (ts < ts_bins[ii] + dt)
        scat.set_offsets(np.c_[x[idx], y[idx]])
        scat.set_array(c[idx])
        if len(ts_bins) > 0:
            ax.set_title('%s ms' % ts_bins[ii])
        else:
            ax.set_title('%s' % ii)
        return scat,

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=len(ts_bins), interval=50, blit=True)

    return anim

def imshow(fig, ax, im, h, ts_bins=[]):
    if type(im) == list:
        frames = len(h[0])
    else:
        frames = len(h)

    def animate(ii):
        if type(im) == list:
            for idx in range(len(im)):
                im[idx].set_array(h[idx][ii])
        else:
            im.set_array(h[ii])
        if len(ts_bins) > 0:
            ax.set_title('%s ms' % ts_bins[ii])
        else:
            ax.set_title('%s' % ii)
        return im,

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)

    return anim
