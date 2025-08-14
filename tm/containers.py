import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm
import tm
from tm.constants import *

def find_prefix_view(arr, prefix):
    # Find all indexes where elements start with the given prefix
    mask = np.char.startswith(arr, prefix)
    # Get the first and last index where the prefix appears
    indices = np.flatnonzero(mask)
    if indices.size > 0:
        start, end = indices[0], indices[-1] + 1  # Inclusive range
        return slice(start, end)    
    return None  # Return None array if no matches

def datetime_to_int(index: pd.DatetimeIndex):
    # Convert DatetimeIndex to Unix timestamps.
    return index.view(np.int64) // 10 ** 9

def int_to_datetime(ts):
    return pd.to_datetime(ts * 10 ** 9)

class Data:
    def __init__(
                self, 
                ts:np.ndarray, 
                y:np.ndarray, 
                s:np.ndarray = None, 
                x:np.ndarray = None, 
                z:np.ndarray = None, 
                t:np.ndarray = None, 
                msidx:np.ndarray = None, 
                pw:np.ndarray = None, 
                w:np.ndarray = None,
                y_cols:np.ndarray = np.array([]),
                x_cols:np.ndarray = None,
                t_cols:np.ndarray = None,  
                w_cols:np.ndarray = None              
                ):        
        self.ts = ts
        self.y = y
        self.x = x
        self.z = z
        self.t = t
        self.msidx = msidx
        self.msidx_start = None
        self.msidx_start_lookup = None
        self.s = s
        self.pw = pw
        self.w = w
        self.y_cols = y_cols
        self.x_cols = x_cols
        self.t_cols = t_cols

        assert y.ndim == 2, "y must be a matrix"
        self.n, self.p = self.y.shape

        # check inputs
        if self.msidx is None:
            self.msidx = np.zeros(self.n, dtype = int)
        
        # fix msidx
        self._process_msidx()

        if w_cols is None:
            self.w_cols = np.array([f"w_{e}" for e in self.y_cols])
        else:
            self.w_cols = w_cols

        if self.w is None:
            self.w = np.zeros((self.n, self.p))
        if self.pw is None:
            self.pw = np.ones(self.n)
        if self.s is None:
            self.s = np.zeros(self.n)

    def _process_msidx(self):
        #
        # this needs to create a new copy
        if not self.empty:
            diff = np.diff(self.msidx, prepend=self.msidx[0])
            change = (diff != 0).astype(int)
            self.msidx = np.cumsum(change) # store in the same array without creating a new one                    
            # compute starts
            mask = np.r_[True, self.msidx[1:] != self.msidx[:-1]]
            self.msidx_start = np.where(mask)[0]        

        else:
            self.msidx_start = np.array([0])
            
        self.msidx_start_lookup = self.msidx_start[np.searchsorted(self.msidx_start, np.arange(self.n+1), side='right') - 1]

        return self

    @property
    def empty(self):
        return self.n == 0

    def to_df(self):
        v = [self.y]
        c = list(self.y_cols)
        if self.x is not None:
            v.append(self.x)
            c += list(self.x_cols)
        if self.z is not None:
            v.append(self.z)
            c += ['z']
        if self.t is not None:
            v.append(self.t)
            c += list(self.t_cols)
        v += [self.msidx, self.s, self.pw, self.w]
        c += ['msidx', 's', 'pw'] + list(self.w_cols)
        v = np.hstack([np.atleast_2d(e.T).T for e in v])
        return pd.DataFrame(v, columns = c, index = self.index())
    
    def __repr__(self):
        return self.to_df().__repr__()

    def index(self):
        return int_to_datetime(self.ts)

    @classmethod
    def from_df(cls, df: pd.DataFrame):

        df.index = pd.to_datetime(df.index)
        
        columns = np.array(list(df.columns))
        # create variables
        # ts
        ts = datetime_to_int(df.index)
        # y
        tmp = find_prefix_view(columns, Y)
        assert tmp, "y must be present"
        y = np.array(df.values[:, tmp], dtype = np.float64)
        y_cols = columns[tmp]

        # x
        x = None
        x_cols = None
        tmp = find_prefix_view(columns, X)
        if tmp:
            x = np.array(df.values[:, tmp], dtype = np.float64)
            x_cols = columns[tmp]

        # z (vector)
        z = None
        tmp = find_prefix_view(columns, Z)
        if tmp:
            z = np.array(df.values[:, tmp.start], dtype = np.int64)

        # t 
        t = None
        t_cols = None
        tmp = find_prefix_view(columns, T)

        if tmp:
            t = np.array(df.values[:, tmp], dtype = np.float64)
            t_cols = columns[tmp]

        # msidx (vector)
        msidx = None
        tmp = find_prefix_view(columns, MSIDX)
        if tmp:
            msidx = df.values[:, tmp.start]
        return cls(ts = ts, y = y, x = x, z = z, t = t, msidx = msidx, y_cols = y_cols, x_cols = x_cols, t_cols = t_cols)

    def __getitem__(self, idx):
        # Support tuple indexing: if a tuple is provided, use its first element for row slicing.
        if isinstance(idx, tuple):
            idx = idx[0]

        return Data(
                    ts = self.ts[idx], 
                    y = self.y[idx], 
                    x = self.x[idx] if self.x is not None else None, 
                    z = self.z[idx] if self.z is not None else None, 
                    t = self.t[idx] if self.t is not None else None, 
                    s = self.s[idx],
                    pw = self.pw[idx],
                    w = self.w[idx],
                    msidx = self.msidx[idx], 
                    y_cols = self.y_cols, 
                    x_cols = self.x_cols, 
                    t_cols = self.t_cols,
                    w_cols = self.w_cols
                    )

    def copy(self):
        """
        Return a deep copy of the Data instance.
        Use this method when you need an independent copy of the data.
        """
        return copy.deepcopy(self)

    def after(self, ts: int):
        if self.n == 0:
            return self
        condition = np.where(self.ts > ts)[0]
        if condition.size:
            return self[condition[0]:] # we can use simple indexing here
        else:
            return self[:0]

    def before(self, ts: int):
        if self.n == 0:
            return self
        condition = np.where(self.ts < ts)[0]
        if condition.size:
            return self[:condition[-1]+1] # we can use simple indexing here
        else:
            return self[:0]

    def between(self, ts_lower: int, ts_upper: int):
        if self.n == 0:
            return self
        condition = np.where((self.ts >= ts_lower) & (self.ts <= ts_upper))[0]
        if condition.size:
            return self[condition[0]:condition[-1]+1] # we can use simple indexing here
        else:
            return self[:0]

    def random_segment(self, burn_fraction: float, min_burn_points: int):
        if self.n == 0:
            return self
        start = np.random.randint(max(min_burn_points, 1), max(int(self.n * burn_fraction), min_burn_points + 1))
        end = np.random.randint(max(min_burn_points, 1), max(int(self.n * burn_fraction), min_burn_points + 1))
        return self[start:self.n - end]

    def stack(self, data: 'Data', allow_both_empty: bool = False):
        if self.n == 0 and data.n == 0:
            if allow_both_empty:
                return self
            raise ValueError("Both Data objects are empty. Cannot stack.")
        if self.n == 0:
            vars(self).update(vars(data))
            return self

        if data.n == 0:
            return self
        
        # ts is not stacked as this is used to concat training data
        self.ts = np.hstack((self.ts,data.ts))
        
        self.y = np.vstack((self.y, data.y))             
        
        if self.x is not None:
            self.x = np.vstack((self.x, data.x))             
        if self.t is not None:
            self.t = np.vstack((self.t, data.t))
        if self.z is not None:
            self.z = np.hstack((self.z, data.z))

        self.s = np.hstack((self.s, data.s))
        self.pw = np.hstack((self.pw, data.pw))
        self.w = np.vstack((self.w, data.w))

        # stack msidx
        offset = self.msidx[-1] - data.msidx[0] + 1
        self.msidx = np.hstack((self.msidx, data.msidx + offset))
        self.n, self.p = self.y.shape
        # process again..
        self._process_msidx()
        return self     

    def as_dict(self):
        #  
        return {'y':self.y, 'x':self.x, 't':self.t, 'z':self.z, 'msidx':self.msidx}

    def input_at(self, idx: int = None):
        if idx is None:
            idx = self.n - 1
        # (!) idx should be 0 <= idx <= n-1
        start = self.msidx_start_lookup[idx] 
        return {
                'y':self.y[start: idx + 1], 
                'x':self.x[start: idx + 1] if self.x is not None else None, 
                't':self.t[start: idx + 1] if self.t is not None else None, 
                'z':self.z[start: idx + 1] if self.z is not None else None,
                'msidx':self.msidx[start: idx + 1], 
                }
        
    def at(self, idx: int = None):
        """
        Return the most recent subsequence at index idx
        If idx is None, use the last observation.
        Note: This uses the new tuple-indexing feature so you can write:
              return self[start, idx+1]
        """
        if idx is None:
            idx = self.n - 1
        
        #valid_starts = self.msidx_start[self.msidx_start <= idx]
        #start = valid_starts[-1] if valid_starts.size > 0 else 0
        start = self.msidx_start_lookup[idx] 
        return self[start: idx + 1]

# dict of Data with properties
class Dataset(dict):
    
    def __init__(self):
        self.folds_ts = None
        # methods to behave like dict
    
    def copy(self):
        """
        Return a deep copy of the Data instance.
        Use this method when you need an independent copy of the data.
        """
        dataset = Dataset()
        for k, data in self.items():
            dataset.add(k, data.copy())
        return dataset

    def add(self, key:str, item: Union[pd.DataFrame,Data]):
        if isinstance(item, pd.DataFrame):
            item = Data.from_df(item)
        else:
            if not isinstance(item, Data):            
                raise TypeError("Item must be an instance of pd.DataFrame or Data")
        self[key] = item        

    def split_ts(self, k_folds = 3):
        # join all ts arrays, compute unique values
        # and split array
        ts = []
        for k, data in self.items():
            ts.append(data.ts)
        ts = np.hstack(ts)
        ts = np.unique(ts)
        idx_folds = np.array_split(ts, k_folds)
        self.folds_ts = [(fold[0], fold[-1]) for fold in idx_folds]
        return self

    def split(
            self, 
            test_fold_idx: int, 
            burn_fraction: float = 0.1, 
            min_burn_points: int = 1, 
            seq_path: bool = False
            ):
        
        if seq_path and test_fold_idx == 0:
            raise ValueError("Cannot start at fold 0 when path is sequential")
        if len(self.folds_ts) is None:
            raise ValueError("Need to split before getting the split")
        
        train_dataset = Dataset()
        test_dataset = Dataset()

        for key, data in self.items():

            ts_lower, ts_upper = self.folds_ts[test_fold_idx]            
            
            # get test data
            test_data = data.between(ts_lower, ts_upper)

            # create training data
            train_data = data.before(ts = ts_lower).random_segment(burn_fraction, min_burn_points)
            # if path is non sequential add data after the test set
            if not seq_path:
                train_data_add = data.after(ts = ts_upper).random_segment(burn_fraction, min_burn_points)
                train_data.stack(train_data_add, allow_both_empty = True)
                        
            train_dataset.add(key, train_data)
            test_dataset.add(key, test_data)    
        
        return train_dataset, test_dataset


if __name__ == '__main__':

    def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01'):
        x=np.random.normal(0,scale,n)
        y=a+b*x+np.random.normal(0,scale,n)
        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
        return data

    df1 = linear(n=10, a=0, b=0.1, start_date='2000-01-01')
    df2 = linear(n=10, a=0, b=0.1, start_date='2000-01-01')
    
    data1 = Data.from_df(df1)
    data2 = Data.from_df(df2)
    data1.stack(data2)


    print('After stack')
    print(data1)
    print('-------')
    print('Evaluate at')
    tmp = data1.at(15)
    print(tmp)
    print()
    print(data1.msidx)

    t = trdm.utils.msidx_to_table(data1.msidx)
    print()
    print(t)
    for i in range(t.shape[0]):
        print(data1.msidx[t[i][0]:t[i][1]])
    exit(0)


    df = linear(n=10, a=0, b=0.1, start_date='2000-01-01')
    data = Data.from_df(df)
    print(data)
    

    tmp = data[5:]
    print(tmp)


    tmp.y[:] = 2
    print(tmp)
    print(data)

    exit(0)

        

    print(data1)
    print(data1.as_dict())
    print(data1.at(5))








