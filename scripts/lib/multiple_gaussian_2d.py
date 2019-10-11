# -*- coding: utf-8 -*-
#
# multiple_gaussian_2d.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import scipy as sci

from scipy.optimize import curve_fit

def gaussian_2d((x,y), amp, mu_x,  mu_y, sig):
    val = amp * np.exp( -np.power(x - mu_x, 2.) /  (2. * np.power(sig, 2.)) ) * np.exp( -np.power(y - mu_y, 2.) /  (2. * np.power(sig, 2.)) )
    return val.ravel()

def multiple_gaussian_2d((x,y), popts, normalized=False):
    z = np.zeros((len(x)*len(y)))
    for popt in popts:
        ii = gaussian_2d((x,y),*popt)
        if normalized and ii.sum() != 0:
            ii /= float(ii.sum())
        z += ii
    return z

def multiple_gaussian_2d_fit((x,y), z, p0):
    popts,pcovs = [],[]
    for p in p0:
        try:
            popt,pcov = curve_fit(gaussian_2d, (x,y), z, p)
            popts.append(popt), pcovs.append(pcov)
        except:
            pass

    return np.array(popts),pcovs

def multiple_gaussian_2d_fit_search((x,y), z, p0, steps, correcting=False, max_std=10):
    amp,mu_x,mu_y,sig = p0

    xx,yy = x[::steps,::steps].ravel(), y[::steps,::steps].ravel()
    p0 = zip(len(xx)*[amp], xx, yy, len(xx)*[sig])
    popts,pcovs = multiple_gaussian_2d_fit((x,y),z,p0)

    if correcting:
        if np.sum(popts) != 0:
            popts = popts[popts[:,0] > 0.01]
            popts = popts[(popts[:,3] >= 1.) * (popts[:,3] <= max_std)]

        amp,mu_x,mu_y,sig = np.array(popts).T
        pos = np.array([np.round(mu_x),np.round(mu_y)], dtype=int).T
        pos = map(lambda x: x[0]+x[1]*1j,pos)
        pos_unique,idx = np.unique(pos, return_index=True)
        popts = np.array([amp[idx], mu_x[idx], mu_y[idx],np.abs(sig[idx])]).T

    return popts


if '__main__' == __name__:
    import pylab as pl

    # Create x and y indices
    x = np.linspace(0, 100, 101)
    y = np.linspace(0, 100, 101)
    x, y = np.meshgrid(x, y)

    #create data
    data = multiple_gaussian_2d((x, y), [[1,50,60,10],[.7,20,30,5]])

    # plot twoD_Gaussian data generated above
    pl.figure()
    pl.imshow(data.reshape(101, 101), origin='bottom')
    pl.colorbar()

    # add some noise to the data and try to fit the data generated beforehand
    initial_guess = [[1,50,60,10],[.7,20,30,5]]
    data_noisy = data + 0.2*np.random.normal(size=data.shape)

    popt, pcov = multiple_gaussian_2d_fit((x, y), data_noisy, p0=initial_guess)
    data_fitted = multiple_gaussian_2d((x, y), popt)

    fig, ax = pl.subplots(1, 1)
    ax.hold(True)
    ax.imshow(data_noisy.reshape(101, 101), cmap=pl.cm.jet, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, data_fitted.reshape(101, 101), 8, colors='w')

    pl.show()
