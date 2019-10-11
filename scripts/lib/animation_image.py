# -*- coding: utf-8 -*-
#
# animation_image.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import pylab as pl

from matplotlib import animation
mywriter = animation.FFMpegWriter()

def animate_image(ax, h, *args, **kwargs):

    # First set up the figure, the axis, and the plot element we want to animate
    im = ax.imshow(h[0], *args, **kwargs)

    # initialization function: plot the background of each frame
    def init():
        im.set_array([])
        return im,

    def animate(ii):
        im.set_array(h[ii])
        ax.set_title('%s'%ii)
        pl.draw()
        return im,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(ax.get_figure(), animate, frames=len(h), interval=50, blit=True)

    return anim


def multiple_animate_image(axes, data, *args, **kwargs):

    # First set up the figure, the axis, and the plot element we want to animate
    im = []
    for ii in range(len(axes)):
        im.append(axes[ii].imshow(data[ii][0], *args, **kwargs))

    def animate(jj):
        for ii in range(len(im)):
            im[ii].set_array(data[ii][jj])
            axes[ii].set_title('%s'%jj)
        pl.draw()
        return im

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(axes[0].get_figure(), animate, frames=len(data[0]), interval=50, blit=True)

    return anim


#anim.save('animation_image.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
