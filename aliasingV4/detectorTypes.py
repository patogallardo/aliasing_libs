import moby2
import numpy as np

tods_path = '/mnt/d/data_depot/magnetarObs/'
SAMPLETOD_FNAMES = {'mce1': tods_path + '1540230569.1540241875.ar4.zip',
                    'mce2': tods_path + '1540230569.1540241846.ar5.zip',
                    'mce3': tods_path + '1540230569.1540237634.ar6.zip'}

SAMPLE_TOD = {'mce1': moby2.scripting.get_tod({'filename': SAMPLETOD_FNAMES['mce1'], 
                                               'read_data':False}), 
              'mce2': moby2.scripting.get_tod({'filename': SAMPLETOD_FNAMES['mce2'],
                                               'read_data':False}),
              'mce3': moby2.scripting.get_tod({'filename': SAMPLETOD_FNAMES['mce3'],
                                               'read_data':False})}

def get_det_freq(mce, row, col):
    '''Receives a mce string: mce1, mce2 or mce3.
    row, col: int 
    returns detector kind as in det freq. '''
    assert mce in ['mce1', 'mce2', 'mce3']
    tod = SAMPLE_TOD[mce]
    selc = tod.info.array_data['col'] == col
    selr = tod.info.array_data['row'] == row
    sel_det = np.logical_and(selc, selr)
    assert sel_det.sum() == 1
    det_freq = int(tod.info.array_data['nom_freq'][sel_det][0])
    return det_freq


def sel_byFreq(mce, rows, cols, whatFreq):
    '''Receives:
    a string with what mce this is from: 'mce1' for example
    rows, cols = vector of ints rows=[1,2,3] cols=[0,0,0]
    whatFreq: float with frequency ex: 90.0
    returns: a vector with True or False showing if that detector
    is a det of that frequency.'''
    assert mce in ['mce1', 'mce2', 'mce3']
    assert len(rows) == len(cols)
    sel = np.zeros(len(rows), dtype='bool')
    for j in range(len(rows)):
        row, col = rows[j], cols[j]
        freq = get_det_freq(mce, row, col)
        sel[j] = freq == whatFreq
    return sel
