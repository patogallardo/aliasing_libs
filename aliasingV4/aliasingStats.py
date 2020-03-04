
def getAliasingStats(noiseObservationList, badDets):
    '''Receives a list of noiseObservations and badDets, 
    returns statistics as the 50% percentile and 1sigma deviations.'''
    fss = np.array([noiseObservationList[j].fs for j in range(len(noiseObservationList))])
    maxFsLoc = np.where(fss == fss.max())[0][0]

    good_rows, good_cols = badDets.good_rows, badDets.good_cols
    
    indices = range(len(noiseObservationList))
    indices.pop(maxFsLoc) # remove the norm to iterate

    norm = noiseObservationList[maxFsLoc].avgNoise[good_rows, good_cols]

    aliasData = np.empty([len(indices), len(good_rows)]) # aliasFractions go here
    freqMarkers = np.array([noiseObservationList[indx].fs for indx in indices])
    for j, obsIndex in enumerate(indices):
        noiseObs = noiseObservationList[obsIndex]
        avgNoise = noiseObs.avgNoise[good_rows, good_cols]
        aliasFraction = avgNoise/norm
        aliasData[j, :] = aliasFraction
    p16, p50, p84 = np.percentile(aliasData, [16, 50, 84], axis=1)
    s1 = p50-p16
    s2 = p84-p50
    sigma = np.vstack([s1,s2])
    median = p50

    return median, sigma
