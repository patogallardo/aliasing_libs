import numpy as np
import scipy.signal

def removePolynomialFromTod(tod,  order=6):
    '''Fits a polynomial and removes it from the data.
    needs a tod of the type returned by openMCEFile, see openTodlib for more detail.'''
    [rows, cols, nsamples] = tod.data.shape
    fs = tod.fs
    dt = 1.0/fs
    t = np.arange(nsamples) * dt
    for i in range(rows):
        for j in range(cols):
            polynomial = np.polyfit(t, tod.data[i, j], deg=order)
            trend = np.poly1d(polynomial)
            tod.data[i, j, :] = tod.data[i, j, :] - trend(t)
    #return nothing

def getPeriodogramsNumpy(tod):
    f, pxx = scipy.signal.periodogram(tod.data, fs=tod.fs, axis=2,
                                      scaling='density')
    return f, pxx
