import pandas as pd

def openDB(dbFilename, commonLength=True, query=None):
    '''Opens csv files with filenames, pctRn, etc... Returns a pandas dataframe'''
    df = pd.read_csv(dbFilename)
    if query is not None:
        df.query(query, inplace=True)
    if commonLength:
        dts = df.tmax - df.tmin
        dt = dts.min()
        df.tmax = df.tmin + dt
    return df
