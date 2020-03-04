import numpy as np
import matplotlib.pyplot as plt
import aliasingV4.openTodLib as otod

def plotHistogramOfBiasPowers(fnamePath, show=True,
                              intervalToPlot=[1,25]):
    fname = fnamePath.split('/')[-1]
    outfile = otod.IVrunfileContainer(fnamePath)
    biasPowers = outfile.getBiasPowers()

    plt.hist( biasPowers.flatten(), range=intervalToPlot,
             bins=150)
    plt.title(fname)
    plt.xlabel('Bias power [pW]')
    plt.ylabel('N')
    if show:
        plt.show()
    else:
        plt.savefig('%s_Histogram.png' %fname)
        plt.close()
