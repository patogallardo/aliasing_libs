'''Routines to extract average noise levels from a tod go here.'''
import aliasingV4.openTodLib as openTodLib
import aliasingV4.todAnalysis as todAnalysis
import aliasingV4.detectorTypes as detTypes
import numpy as np
import pandas as pd
import os

def getNoiseObservationFrom_df_line(dfline, freqRange=[10, 60]):
    '''Receives a filename db line and gets the noise observation.
    length must be a pandas series. you can get this with iloc[0]

    returns a noiseObservation object
    '''
    assert type(dfline) == pd.core.series.Series
    fname = dfline.fname
    iv = dfline.ivCurve
    mce = dfline.mce
    tmin = dfline.tmin
    tmax = dfline.tmax
    pctRn = dfline.pctRn

    no = noiseObservation(fname, mce, pctRn=pctRn, ivOutFile=iv,
                          tlims=[tmin, tmax], freqRange=freqRange)
    return no


class noiseObservation:
    '''This object stores one noise observation, from one mcefile name.
    This stores the MCEFile, tod, and frequency domain noise information.
    Receives fname and loads it in this container.
    periodogram info will be added as optional.
    fname: mce file filename
    pctRn: pctRn
    ivOutFile: iv.out filename
    tlims: limits in time to consider in time series'''
    def __init__(self, fname, mce, pctRn=None, ivOutFile=None,
                 tlims=None, freqRange=[20,60]):
        self.fname = fname
        self.mce = mce
        self.array = {'mce1': 'AR4', 'mce2':'AR5', 'mce3': 'AR6'}[mce]
        mcefile, tod = openTodLib.openMCEFile(fname, mce, tlims=tlims)
        self.mcefile = mcefile
        self.unit = 'dac'
        self.freqRange = freqRange
        if ivOutFile == 'None':
            ivOutFile = None
	if ivOutFile is not None:
            nrows = tod.data.shape[0]
            self.iv = openTodLib.IVrunfileContainer(ivOutFile, mce)
            responsivities = self.iv.responsivities[:nrows, :]
            tod.data = responsivities[:,:, np.newaxis] * tod.data * 1e12
            self.unit = 'pW'
        self.tod = tod
        self.t = tod.t
        self.fs = tod.fs
        self.nrows, self.ncols, self.nsamples = tod.data.shape
        self.getNoiseSpectrum()
        self.getAvgNoiseSpectrum(freqRange=freqRange)
        self.pctRn = pctRn
        self.detFreqs = self.getDetFreqs()
    def getNoiseSpectrum(self):
        '''Gets the ffts and stores them.'''
        f, pxx = todAnalysis.getPeriodogramsNumpy(self.tod)
        self.f = f
        self.pxx = pxx
    def getAvgNoiseSpectrum(self, freqRange = [20, 60]):
        '''Uses results from getNoiseSpectrum and computes one 
        average noise per detector in the indicated band'''
        fmin, fmax = freqRange
        self.freqRange = freqRange
        sel = (self.f > fmin) & (self.f < fmax)
        pxx = self.pxx
        avgNoise = np.mean(pxx[:,:,sel], axis=-1)
        stdNoise = np.std(pxx[:,:,sel], axis=-1)
        medianNoise = np.median(pxx[:,:,sel], axis=-1)
        self.medianNoise = medianNoise
        self.avgNoise = avgNoise
        self.stdNoise = stdNoise
    def getDetFreqs(self):
        '''writes detector frequencies from moby.'''
        detFreqs = np.zeros([self.nrows, self.ncols])
        for row in range(self.nrows):
            for col in range(self.ncols):
                detFreqs[row, col] = int(detTypes.get_det_freq(self.mce, row, col))
        return detFreqs


def writeAliasingStats(no_fast, no_slow):
    '''Writes out statistics for the fast and slow noise observation.
    there will be columns for the noise mean psd, noise median psd in band
    Array, nrows, pctRn, detFreq, etc...'''
    nrows = min(no_fast.nrows, no_slow.nrows)
    ncols = no_fast.ncols

    fast_avg_noise_psd = np.zeros(nrows * ncols)
    slow_avg_noise_psd = np.zeros(nrows * ncols)
    fast_median_noise_psd = np.zeros(nrows * ncols)
    slow_median_noise_psd = np.zeros(nrows * ncols)
    fast_std_noise = np.zeros(nrows * ncols)
    slow_std_noise = np.zeros(nrows * ncols)
    alias_fraction_avg = np.zeros(nrows * ncols)
    alias_fraction_median = np.zeros(nrows * ncols)

    array = [no_fast.array] * (nrows * ncols)
    pctRn = np.ones(nrows*ncols, dtype=int) * no_fast.pctRn
    nrows_slow = np.ones(nrows * ncols, dtype=int) * no_slow.nrows
    nrows_fast = np.ones(nrows * ncols, dtype=int) * nrows

    rows = np.zeros(nrows * ncols, dtype=int)
    cols = np.zeros(nrows * ncols, dtype=int)

    fs_slow = np.ones(nrows * ncols) * no_slow.fs
    fs_fast = np.ones(nrows * ncols) * no_fast.fs

    detFreq = np.zeros(nrows * ncols, dtype=int)
    for row in range(nrows):
        for col in range(32):
            indx = col + ncols * row
            rows[indx] = row 
            cols[indx] = col
            fast_avg_noise_psd[indx] = no_fast.avgNoise[row, col]
            slow_avg_noise_psd[indx] = no_slow.avgNoise[row, col]
            fast_median_noise_psd[indx] = no_fast.medianNoise[row, col]
            slow_median_noise_psd[indx] = no_slow.medianNoise[row, col]
            fast_std_noise[indx] = no_fast.stdNoise[row, col]
            slow_std_noise[indx] = no_slow.stdNoise[row, col]
            alias_fraction_avg[indx] = no_slow.avgNoise[row,col] / no_fast.avgNoise[row, col]
            alias_fraction_median[indx] = no_slow.medianNoise[row,col]/no_fast.medianNoise[row, col]
            detFreq[indx] = no_fast.detFreqs[row, col]
    df = pd.DataFrame({'row': rows,
                       'col': cols,
                       'fs_slow': fs_slow,
                       'fs_fast': fs_fast,
                       'fast_avg_noise_psd':fast_avg_noise_psd, 
                       'slow_avg_noise_psd': slow_avg_noise_psd, 
                       'fast_median_noise_psd': fast_median_noise_psd,
                       'slow_median_noise_psd': slow_median_noise_psd,
                       'fast_std_noise': fast_std_noise,
                       'slow_std_noise': slow_std_noise,
                       'alias_fraction_avg': alias_fraction_avg,
                       'alias_fraction_median': alias_fraction_median,
                       'array' : array,
                       'pctRn': pctRn,
                       'nrows_slow': nrows_slow,
                       'nrows_fast': nrows_fast,
                       'detFreq': detFreq})
    folderout = 'results/noiseStats'
    if not os.path.exists(folderout):
        os.mkdir(folderout)
    df.to_csv('%s/noiseStats_pctRn_%i_nrows_%i_nrows_%i.csv' %(folderout,
              pctRn[0], no_fast.nrows, no_slow.nrows))
    return df
