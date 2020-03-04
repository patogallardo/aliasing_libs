'''Receives a list of noise observations and plots them in different ways. '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
from matplotlib import gridspec
from aliasingV4 import detectorTypes
import progressbar
import matplotlib.cm as cm

def plot_inOnePlot_show_all_fast_psds(noiseObservationList,
                                   detFreq=None,
                                   plotAllNoiseCurves=False,
                                   plotAverageNoiseCurves=True,
                                   show=True,
                                   useBadDets=None):
    '''Receives a list of noise observations.
    smooth out the psds and plot them for all the detectors with 
    a high transparency.
    The hope is that the mean shows up after overlapping enough of these.
    noiseObservationList: noise instances for the array
    plotAllnoiseCurves: True if you want to plot everything in nos
    plotAverageNoiseCurves: True if you want to overlay the avg noise
    show: to show
    useBadDets: pandas dataframe containing a list of rows and cols to use'''

    if useBadDets is not None:
        badDets = useBadDets
        rows, cols = badDets.row, badDets.col

    if useBadDets is None:
        from itertools import product
        rows, cols = range(4), range(32)
        rows, cols = np.meshgrid(rows, cols)#all combinations
        rows, cols = rows.flatten(), cols.flatten()
    if selFreq is not None:
        sel = detecctorTypes.sel_byFreq(noiseObservationList[0].mce,
                                    rows, cols, detFreq)
        rows, cols = rows[sel], cols[sel]
    rowcols = zip(rows, cols)

    if plotAllNoiseCurves:
        for j, noiseObs in enumerate(noiseObservationList):
            assert noiseObs.pctRn is not None
            pctRn = noiseObs.pctRn
            print "plotting %i pctRn"%pctRn
            f = noiseObs.f
            for row, col in rowcols:
                print "row %i col %i" % (row, col)
                pxx = noiseObs.pxx[row, col]
                pxx_smooth = pd.Series(pxx).rolling(window=100).mean()
                plt.loglog(f, pxx_smooth, color='C%i'%j,
                           alpha=0.9)
                    #now plot just the color for the label
            plt.plot([], color='C%i' % j, label='pctRn: %i'%pctRn)

    if plotAverageNoiseCurves:
        for j, noiseObs in enumerate(noiseObservationList):
            pctRn = noiseObs.pctRn
            f = noiseObs.f
            pxx_mean = np.median(noiseObs.pxx[rows, cols, :], axis = (0))
            plt.loglog(f, pxx_mean, color='C%i'%j, alpha=0.6,
                       label='pctRn: %i' % pctRn)
            plt.title('Median Noise')
  
    plt.legend()
    assert noiseObs.unit == 'dac'
    plt.ylim([2e-5, 2])
    plt.xlim([0.5, 3e3])
    plt.xlabel('f [Hz]')
    plt.ylabel('psd %s^2/Hz$' % noiseObs.unit )
    if show:
        plt.show()
    else:
        outputDir = 'results/AvgFastNoiseCurves'
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        if plotAverageNoiseCurves:
            filename = 'AvgNoiseFastSamplingRate.png'
            if detFreq is not None:
                filename = "AvgNoiseFastSamplingRate_det_%i.png" % detFreq
        else:
            filename = 'NoiseCurvesFastSamplingRate.png'
            if detFreq is not None:
                filename = 'NoiseCurvesFast_det_%i.png' %detFreq
        plt.savefig(outputDir + '/%s'%filename, dpi=200)
        plt.close()

         


def plot_psds_fastSamplingRate_forAllBiasPoints(noiseObservationList, row, col, show=False):
    '''Receives a list of observations and plots spectra for all dets.
    For the same detector it superimposes the noise data and shows a smoothed out version of the 
    noise spectrum.'''
    for j, noiseObs in enumerate(noiseObservationList):
        assert noiseObs.pctRn is not None
        pctRn = noiseObs.pctRn
        f = noiseObs.f
        pxx = noiseObs.pxx[row, col, :]
        pxx_smooth = pd.Series(pxx).rolling(window=100).mean()
        plt.loglog(f, pxx_smooth, color = 'C%i'%j, alpha=0.8,
                   label='pctRn: %i' % pctRn)
    plt.title('row: %02i col: %02i' % (row, col))
    if noiseObs.unit == 'dac':
        plt.ylim([1e-5, 2])
    else:
        plt.ylim([1e-14, 1e-7])
    plt.xlabel('f [Hz]')
    plt.ylabel('psd [%s$^2$/Hz]' %noiseObs.unit)
    plt.legend(loc='lower left')
    plt.grid()
    if show:
        plt.show()
    else:
        outputDir = 'results/psd_fastSamplingRate_Vary_pctRn'
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        plt.savefig(outputDir + '/row%02i_col%02i.png' % (row, col), dpi=200)
        plt.close()


def plot_psds_fastSampRate_allBiasPoints(noiseObservationList):
    '''loops over plot_psds_fastSamplingRate_forAllBiasPoints and makes the plots
    for all dets one by one'''
    limitRow=4
    for row in range(limitRow):
        for col in range(32):
            plot_psds_fastSamplingRate_forAllBiasPoints(noiseObservationList, row, col, show=False)


def plot_psds_oneDet(noiseObservationList, row, col, show=False):
    '''Gets a list of noise observations.
    Loops across them and plots all detectors.'''

    for j, noiseObs in enumerate(noiseObservationList):
        assert noiseObs.pctRn is not None
        pctRn = noiseObs.pctRn

        f = noiseObs.f
        pxx = noiseObs.pxx[row, col, :]
        pxx_smooth = pd.Series(pxx).rolling(window=100).mean()
        label = 'fs:%3.1f $\\mu=%1.2e$'% (noiseObs.fs, noiseObs.avgNoise[row, col])
        plt.loglog(f, pxx, color='C%i'%j, alpha=0.6,
                   label=label)
        plt.axhline(noiseObs.avgNoise[row, col], color='C%i' %j)
        plt.plot(f, pxx_smooth, color='C%i'%j, alpha=0.8)
    plt.title('pctRn:%i row:%02i col:%02i '%(pctRn, row, col))
    plt.axvspan(noiseObs.freqRange[0], noiseObs.freqRange[1],
                alpha=0.1, color='black')
    if noiseObs.unit == 'dac':
        plt.ylim([1e-5, 100])
    else:
        plt.ylim([1e-7, 2e-4])
    plt.xlabel('f[Hz]')
    plt.ylabel('psd [%s/$\sqrt{Hz}$]' % noiseObs.unit)
    plt.legend()
    if show:
        plt.show()
    else:
        outputDir = 'results/psdVaryingSamplingFreq_%i_pctRn/' % pctRn
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        plt.savefig(outputDir + "pctRn_%2i_row_%02i_col_%02i.png"%(pctRn, row, col), dpi=200)
        plt.close()


def plot_psds_allDets(noiseObservationList, limitRow=4):
    '''Loops over plot_psds_oneDet for the whole array'''
    for row in range(limitRow):
        for col in range(32):
            plot_psds_oneDet(noiseObservationList, row, col, show=False)


def makeHist(noiseObservationList, badDets, indexToPlot=None, show=True):
    '''Normalizes with respect to the fastest sampling rate data.
    plots histogram of aliasing fraction for list of good detectors given in the 
    badDets object
    noiseObservationList: list with noise observation objects
    indexToPlot: what index in noiseObservationList to plot
    show: show or not the plot
    badDets: badDetector object, see cuts file'''
    fss = np.array([noiseObservationList[j].fs for j in range(len(noiseObservationList))])
    maxFsLoc = np.where(fss == fss.max())[0][0]

    if indexToPlot is None: #if None, plot the smallest sampling freq available
        indexToPlot = np.where(fss == fss.min())[0][0]
    noiseObs = noiseObservationList[indexToPlot]

    good_rows, good_cols = badDets.good_rows, badDets.good_cols
    avgNoise = noiseObs.avgNoise[good_rows, good_cols]
    norm = noiseObservationList[maxFsLoc].avgNoise[good_rows, good_cols]
    normed = avgNoise/norm

    #stats
    sel = (normed > 0.8) & (normed<1.5)
    if sel.sum() >= 1:
        p16, p50, p84 = np.percentile(normed[sel], [16,50,84])
        s_m = p50-p16
        s_p = p84-p50

    plt.hist(normed, bins=60, range=(0.8,2))
    plt.xlabel("aliasing fraction")
    plt.ylabel("count")
    figTitle = ("Aliasing fraction at fs:%3.1f Hz, %i pctRn" 
                %(noiseObs.fs, noiseObs.pctRn))
    plt.title(figTitle)
    if sel.sum() >= 1:
        plt.figtext(0.7,0.7, "$\mu=%1.2f_{-%1.2f}^{+%1.2f}$"%(p50, s_m, s_p))
    if show:
        plt.show()
    else:
        resultsDir = 'results/Histograms'
        if not os.path.exists(resultsDir):
            os.makedirs(resultsDir)
        fname_out = ("%s/AliasingFraction_%3.1fHz_%ipctRn.png" 
                     % (resultsDir, noiseObs.fs, noiseObs.pctRn))
        plt.savefig(fname_out, dpi=200)
        plt.close()


def mosaicPlotAllArray(no_lowfreq, no_highfreq):
    '''Do mosaic plot for the whole array.'''
    array = {'mce1':'AR4','mce2':'AR5','mce3':'AR6'}[no_lowfreq.mce]
    fs = no_lowfreq.fs
    pctRn = no_lowfreq.pctRn
    nrows = no_lowfreq.nrows

    fss = []
    rows = []
    cols = []
    fractions = []
    detFreqs = []
    arrays = []
    pctRns = []
    nrowss = []
    
    avgFasts, avgSlows, stdFasts, stdSlows = [],[],[],[]
    
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.AdaptiveETA()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=128)
    for row in range(4):
        for col in range(32):
            fraction, avgFast, avgSlow, stdFast, stdSlow, detFreq = mosaicPlot(no_lowfreq, no_highfreq, row, col, show=False)
            rows.append(row)
            cols.append(col)
            fractions.append(fraction)
            avgFasts.append(avgFast)
            avgSlows.append(avgSlow)
            stdFasts.append(stdFast)
            stdSlows.append(stdSlow)
            detFreqs.append(detFreq)
            nrowss.append(nrows)
            arrays.append(array)
            fss.append(fs)
            pctRns.append(pctRn)
            bar.update(row*32 + col +1)
    bar.finish()
    #save result
    df = pd.DataFrame({'row': rows, 
                       'col': cols, 
                       'alias_fraction':fractions,
                       'avgPowerFast':avgFasts,
                       'avgPowerSlow':avgSlows,
                       'stdPowerFast':stdFasts,
                       'stdPowerSlow':stdSlows,
                       'detFreq': detFreqs,
                       'array': arrays,
                       'fs': fss,
                       'pctRn': pctRns,
                       'nrows': nrowss})
    outputDir = 'results/aliasFractionsFromMosaicPlot'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    df.to_csv('%s/aliasFractions_pctRn_%i_fs_%i_fromMosaicPlot.csv'%(outputDir, no_lowfreq.pctRn, no_lowfreq.fs))
      


def mosaicPlot(no_lowfreq, no_highfreq, row, col, show=True,
               bandOfInterest = [10, 60]):
    '''For one noise observation (no), plots the time domain
    stream, ffts in both linear space and log space for the row, col
    in question.
    bandOfInterest: freqs in hz to average power.
    '''
    assert row < 4 and col < 32
    
    detectorFrequency = detectorTypes.get_det_freq(no_lowfreq.mce, row, col)

    slowColor = 'C0'
    fastColor = 'C1'
    fig = plt.figure(figsize=[15,8])
    gs = gridspec.GridSpec(nrows=3, ncols=6, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:5]) # time domain plot
    ax2 = fig.add_subplot(gs[1:3, 0:3]) #spectrum
    ax3 = fig.add_subplot(gs[1:3, 3:6]) #spectrum
    ax4 = fig.add_subplot(gs[0, 5])

    #histogram
    ax4.hist(no_highfreq.tod.data[row, col, :],
             label = 'fs: %3.1f' %no_highfreq.fs,
             color=fastColor, alpha=0.5, density=True)
    ax4.hist(no_lowfreq.tod.data[row, col, :],
             label = 'fs: %3.1f' %no_lowfreq.fs,
             color=slowColor, alpha=0.5, density=True)
    ax4.set_xlabel('[%s]' %no_lowfreq.unit)
    ax4.set_ylabel('pdf [1/$%s^2$]' %no_lowfreq.unit)
    ax4.legend()
    
    #time doamin
    ax1.set_title('mce: %s row: %i col: %i Type: %i' %(no_lowfreq.mce, row,
                                                           col, detectorFrequency))
    ax1.plot(no_lowfreq.t[-1] + no_highfreq.t, 
             no_highfreq.tod.data[row,col, :], 
             label='fs: %3.1f' %no_highfreq.fs,
             color=fastColor)
    ax1.plot(no_lowfreq.t,
             no_lowfreq.tod.data[row, col, :],
             label='fs: %3.1f' %no_lowfreq.fs,
             color=slowColor)
    ax1.set_xlabel('t[s]')
    ax1.set_ylabel('amplitude [%s]' %no_lowfreq.unit)
    ax1.legend()
    #frequency domain log scale
    ax2.plot(no_highfreq.f, no_highfreq.pxx[row, col],
             color=fastColor)
    ax2.plot(no_lowfreq.f, no_lowfreq.pxx[row, col],
             color=slowColor)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylim([1e-14, 1e-5])
    ax2.set_xlabel('f[Hz]')
    ax2.set_ylabel('psd [pW$^2$/Hz]')
    ax2.axvspan(bandOfInterest[0], bandOfInterest[1],
                alpha=0.3, color='black')

    #compute alias fraction
    sel_inband_fast = np.logical_and(no_highfreq.f>bandOfInterest[0],
                                     no_highfreq.f<bandOfInterest[1])
    sel_inband_slow = np.logical_and(no_lowfreq.f>bandOfInterest[0],
                                     no_lowfreq.f<bandOfInterest[1])
    avg_powerFast = (no_highfreq.pxx[row, col,sel_inband_fast]).mean()
    avg_powerSlow = (no_lowfreq.pxx[row, col, sel_inband_slow]).mean()
    std_powerFast = (no_highfreq.pxx[row, col, sel_inband_fast]).std()
    std_powerSlow = (no_lowfreq.pxx[row, col, sel_inband_slow]).std()
    NSamples_Fast = sel_inband_fast.sum()
    NSamples_Slow = sel_inband_slow.sum()

    #freq domain linear scale
    ax3.plot(no_highfreq.f, no_highfreq.pxx[row, col],
             color=fastColor, alpha=0.7,
             label='fs:%1.1f' % no_highfreq.fs)
    ax3.plot(no_lowfreq.f, no_lowfreq.pxx[row, col],
             color=slowColor, alpha=0.7,
             label='fs:%1.1f' % no_lowfreq.fs)
    avg_fastError = std_powerFast/np.sqrt(NSamples_Fast)
    avg_slowError = std_powerSlow/np.sqrt(NSamples_Slow)
    ax3.axhline(avg_powerFast, color=fastColor, ls='--',
                label='avg: %1.2e $\\pm$ %1.2e (%1.2f)'%(avg_powerFast,
                avg_fastError, avg_fastError/avg_powerFast))
    ax3.axhline(avg_powerSlow, color=slowColor, ls='--',
                label='avg: %1.2e $\\pm$ %1.2e (%1.2f)'%(avg_powerSlow,
                avg_slowError, avg_slowError/avg_powerSlow))
    ax3.legend(loc='upper right')
    aliasingFraction = avg_powerSlow/avg_powerFast
    ax3.set_title('Alias Fraction: %1.3f' %(aliasingFraction))
    ax3.axvspan(bandOfInterest[0], bandOfInterest[1],
                alpha=0.3, color='black')
    ax3.set_xlim([0, no_lowfreq.fs/2.*1.05])
    ax3.set_ylim([0, 1.5e-8])
    ax3.set_xlabel('f[Hz]')
    ax3.set_ylabel('%s$^2$/Hz' % no_lowfreq.unit)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        outputDir = 'results/mosaicPlot_pctRn_%i_fs_%i_and_%i' %(
                                                        no_lowfreq.pctRn,
                                                        no_lowfreq.fs,
                                                        no_highfreq.fs) 
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        plt.savefig(outputDir + 
                    '/row_%02i_col%02i_fs_%i_and_%i' %(row, col, 
                                                       no_lowfreq.fs,
                                                       no_highfreq.fs))
        plt.close()
    return (aliasingFraction, avg_powerFast, avg_powerSlow,
            std_powerFast, std_powerSlow, detectorFrequency)


def getGoodRowsColsFromNoiseStats(AR, pctRn, detFreq):
    nrows_slow = {'AR4':64, 'AR5': 55, 'AR6': 55}#this is only to get the csv file
    df = pd.read_csv('results/noiseStats/noiseStats_pctRn_%i_nrows_4_nrows_%i.csv' % (pctRn, nrows_slow[AR]))
    df.query('detFreq==%i and alias_fraction_avg>0.5 and alias_fraction_avg<1.5' % detFreq, inplace=True)
    row, col = df.row.values, df.col.values
    print "For array %s, pctRn: %i, detFreq: %i using:" %(AR, pctRn, detFreq)
    print "rows:"
    print row
    print "cols:"
    print col
    print "There are %i dets" % len(row)
    return row, col



def plotMedianDacNoise(nos, detFreq=150, getGoodDetsFromStats=True, show=True):
    ''' gets a list of noise observations, splits the data in detector types
    and makes a plot of the median psd
    '''
    no = nos[0]
    colors = cm.inferno(np.linspace(0,1,len(nos)))
    if not getGoodDetsFromStats:
        detFreqs = no.detFreqs
        row, col = np.where(detFreqs == detFreq)
        print "warning, blindly plotting dets of the same freq"
    alpha=0.4
    for j in range(len(nos)):
        color = colors[j]
        no = nos[j]
        if getGoodRowsColsFromNoiseStats:
            if no.pctRn !=0:
                row, col = getGoodRowsColsFromNoiseStats(no.array,
                                                     no.pctRn,
                                                     detFreq)
        md = np.mean(no.pxx[row, col, :], axis=-2)
        if no.pctRn == 0:
            color = 'blue'
            alpha=0.6
        plt.loglog(no.f, md, label='pctRn: %i, N_det: %i' %(no.pctRn, len(row)),
                   alpha=alpha,
                   color=color)
    plt.ylim([3e-4, 40.0])
    plt.legend()
    title = 'Array: %s. %i GHz mean psd.' %(no.array, detFreq)
    title = title.replace('AR', 'PA')
    plt.title(title)
    plt.xlabel('f [Hz]')
    plt.ylabel('psd [$\\frac{dac^2}{Hz}$]')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('results/average_HighFreqSpectrum_%s_detFreq_%i.pdf' %(no.array,detFreq))
        plt.close()
