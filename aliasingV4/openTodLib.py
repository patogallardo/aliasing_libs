import numpy as np
from moby2.util.mce import MCEFile
from moby2.util.mce import MCERunfile
import os
import envVars
import getFilesFromOrigin as getFiles
import aliasingV4.todAnalysis as todAnalysis

def openMCEFile(fname, mce, deconvolve_digital_filter=False, add_metadata=True,
                removePolynomial=True, tlims=None):
    '''Opens MCEFile in the local folder, see envVars
    fname: filename of the file to get, no full path.
    mce: mce1, mce2, mce3 sets the folder where data is
    deconvolve_digital_filter: bool
    add_metadata: set this to True if you want to include useful data to the 
    mcedata structure.
    tlims: time limits to consider'''
    getFiles.getData(fname, mce)
    localDataOrigin = os.path.join(envVars.local_dataOrigin, mce)
    fname = os.path.join(localDataOrigin, fname)
    mcefile = MCEFile(fname)
    if deconvolve_digital_filter:
        print '!!! FYI, I\'m running w/ mcefile.Read(...):unfilter=True'
        mcedata = mcefile.Read(row_col=True, unfilter=True)
    else:
        mcedata = mcefile.Read(row_col=True, unfilter='DC')
    addMetaData_toData(mcedata, mcefile)
    if tlims is not None:
        sel = np.logical_and(mcedata.t > tlims[0], mcedata.t < tlims[1])
        mcedata.data = mcedata.data[:, :, sel]
    addMetaData_toData(mcedata, mcefile)
    if removePolynomial:
        todAnalysis.removePolynomialFromTod(mcedata, order=6)
    return mcefile, mcedata

def getResponsivitiesFromIV(ivfilename, mce):
    '''needs a .out filename'''
    runfile = MCERunfile(ivfilename)
    responsivities = runfile.Item2d('IV', 'Responsivity(W/DACfb)_C%i', type='float')
    responsivities = np.array(responsivities).T
    return runfile, responsivities

class IVrunfileContainer:
    '''receives a .out filename and gives access to the numbers in it. '''
    def __init__(self, ivfilename, mce):
        if not '.out' in ivfilename:
            ivfilename += '.out'
	getFiles.getData(ivfilename, mce)
        local_dataOrigin = os.path.join(envVars.local_dataOrigin, mce)
	ivfilename = os.path.join(local_dataOrigin, ivfilename)
        self.runfile = MCERunfile(ivfilename)
        self.responsivities = self.getResponsivities()
        self.biasPowers = self.getBiasPowers()
        self.targetPercentRn = self.getTargetPercentRn()

    def get_iv_array(self, key, type='float'):
        '''pass type as array'''
        return np.array(self.runfile.Item2d('IV', key, type=type)).transpose()

    def getResponsivities(self):
        key = 'Responsivity(W/DACfb)_C%i'
        return self.get_iv_array(key)

    def getBiasPowers(self):
        key = 'Bias_Power(pW)_C%i'
        return self.get_iv_array(key)

    def getTargetPercentRn(self):
        return self.runfile.Item('IV','target_percent_Rn', array=False, type='float')


def addMetaData_toData(mcedata, mcefile):
    '''Adds metadata as nrows, ncols, dt, fs, t to mcedata. Look at openMCEFile for details.
    This is only for convenience.'''
    [nrows, ncols, nsamples] = mcedata.data.shape
    fs = mcefile.freq
    dt = 1.0/fs
#    t = np.arange(0, nsamples*dt, dt)
    t = np.arange(nsamples) * dt

    mcedata.nrows = nrows
    mcedata.ncols = ncols
    mcedata.nsamples = nsamples
    mcedata.fs = fs
    mcedata.dt = dt
    mcedata.t = t
