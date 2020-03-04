import aliasingV4.openTodLib as openTodLib
import aliasingV4.todAnalysis as todAnalysis
import os
import matplotlib.pyplot as plt

def plotMCEFile_oneByOne(fname, mce, removePolynomial=True,
                         limitRows=4, pctRn=None,
                         tlims=None):
    mcefile, tod = openTodLib.openMCEFile(fname, mce,
                                          removePolynomial=removePolynomial,
                                          tlims=tlims)
    f, pxx = todAnalysis.getPeriodogramsNumpy(tod)
    outputDir = 'results/detsOneByOne/%s/'%fname
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
#    plt.figure(figsize=(6,12))
    if tlims is None:
        plotname = 'raw'
    else:
        plotname = 'clean'
    for row in range(limitRows):
        for col in range(32):
            print "plotting row %i col%i" % (row, col)
            plt.subplot(211)
            if pctRn is None:
                plt.plot(tod.t, tod.data[row, col, :])
            else:
                plt.plot(tod.t, tod.data[row, col, :], label="%i pctRn" % pctRn)
                plt.legend(loc='upper right')
            plt.xlabel('t [s]')
            plt.ylabel('raw dac')
            plt.subplot(212)
            plt.loglog(f, pxx[row, col, :])
            plt.ylim([1e-7, 100])
            plt.xlabel('f [Hz]')
            plt.ylabel('psd [$dac^2/Hz$]')
            
            plt.savefig(outputDir + "/%s_row_%02i_col_%02i.png"%(plotname,
                                                                 row, col))
            plt.close()
 

def plotMCEFile_transparent_timeDomainPlot(fname, mce, limitRows=4, pctRn=None):
    mcefile, tod = openTodLib.openMCEFile(fname, mce, removePolynomial=True)
    outputDir = 'results/transparentPlot_rawData'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    for row in range(limitRows):
        for col in range(32):
            print "plotting row %i col%i" % (row, col)
            if pctRn is None:
                plt.plot(tod.t, tod.data[row, col, :],
                         alpha=0.1, color='black')
            else:
                plt.plot(tod.t, tod.data[row, col, :], label="%i pctRn" % pctRn,
                         alpha=0.1, color='black')
                plt.legend(loc='upper right')
    plt.ylim([-500, 500])
    plt.title('raw data %s' %fname)
    plt.xlabel('t [s]')
    plt.ylabel('dac')
    plt.savefig(outputDir + "/%s.png" %fname, dpi=300)
    plt.close()


