'''Here are the objects that handle what detecotors should we use. '''
import numpy as np
import pandas as pd

class badDets:
    def __init__(self, fnameDetList, nrows=4):
        '''Specify where teh bad dets file is and I will open it for you and store 
        a bunch of useful variables in here.
        To use it, see xlsx and csv files.'''
        self.fname = fnameDetList
        self.getBadDets(nrows)
    def getBadDets(self, nrows):
        fnameDetList = self.fname
        df = pd.read_csv(fnameDetList)
        bad_df = df.query('use==0')
        good_df = df.query('use==1')
        self.bad_rows = bad_df.row.values
        self.bad_cols = bad_df.col.values
        self.good_rows = good_df.row.values
        self.good_cols = good_df.col.values



