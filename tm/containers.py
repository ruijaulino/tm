import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm
from tm.constants import *

def split_ts(ts, k_folds=4):
    # join all ts arrays, compute unique values
    # and split array
    idx_folds = np.array_split(ts, k_folds)
    return [(fold[0], fold[-1]) for fold in idx_folds]

def datetime_to_int(index: pd.DatetimeIndex):
    return index.view(np.int64) // 10**9

def int_to_datetime(ts):
    return pd.to_datetime(ts * 10**9)

def contiguous_prefix_slices(columns: np.ndarray) -> Dict[str, slice]:
    prefixes = [''.join(filter(str.isalpha, name)) for name in columns]
    result = {}
    i = 0
    while i < len(columns):
        prefix = prefixes[i]
        start = i
        while i < len(columns) and prefixes[i] == prefix:
            i += 1
        result[prefix] = slice(start, i)
    return result

class Data:
    def __init__(self, 
                 ts, y, s=None, x=None, z=None, t=None,
                 msidx=None, pw=None, w=None,
                 y_cols=np.array([]), x_cols=None, t_cols=None, w_cols=None):
        
        self.ts = ts
        self.y = y
        self.x = x
        self.z = z
        self.t = t
        self.s = np.zeros(len(ts)) if s is None else s
        self.pw = np.ones(len(ts)) if pw is None else pw
        self.w = np.zeros((y.shape[0], y.shape[1])) if w is None else w
        self.msidx = np.zeros(len(ts), dtype=int) if msidx is None else msidx

        self.y_cols = y_cols
        self.x_cols = x_cols
        self.t_cols = t_cols
        self.w_cols = w_cols if w_cols is not None else np.array([f"{W}_{e}" for e in y_cols])

        self.n, self.p = self.y.shape
        self.msidx_start = None
        self.msidx_start_lookup = None
        self.folds_ts = None
        self._process_msidx()

    
    def split_ts(self, k_folds, **kwargs):
        self.folds_ts = split_ts(self.ts, k_folds = k_folds)


    def split(
            self, 
            test_fold_idx: int, 
            burn_fraction: float = 0.1, 
            min_burn_points: int = 1, 
            seq_path: bool = False,
            folds_ts: list = None,
            **kwargs
            ):
        
        if seq_path and test_fold_idx == 0:
            raise ValueError("Cannot start at fold 0 when path is sequential")
        folds_ts = self.folds_ts if folds_ts is None else folds_ts
        if folds_ts is None:            
            raise ValueError("Need to split before getting the split")
        
        ts_lower, ts_upper = folds_ts[test_fold_idx]            
        
        # get test data
        test_data = self.between(ts_lower, ts_upper)

        # create training data
        train_data = self.before(ts = ts_lower).random_segment(burn_fraction, min_burn_points)
        # if path is non sequential add data after the test set
        if not seq_path:
            train_data_add = self.after(ts = ts_upper).random_segment(burn_fraction, min_burn_points)
            train_data.stack(train_data_add, allow_both_empty = True)

        return train_data, test_data
        
    def _process_msidx(self):
        if self.empty:
            self.msidx_start = np.array([0])
        else:
            diff = np.diff(self.msidx, prepend=self.msidx[0])
            change = (diff != 0).astype(int)
            self.msidx = np.cumsum(change)
            mask = np.r_[True, self.msidx[1:] != self.msidx[:-1]]
            self.msidx_start = np.where(mask)[0]
        self.msidx_start_lookup = self.msidx_start[np.searchsorted(self.msidx_start, np.arange(self.n+1), side='right') - 1]

    @property
    def empty(self):
        return self.n == 0

    def index(self):
        return int_to_datetime(self.ts)

    def to_df(self):
        v, c = [], []

        def add_field(field, names):
            if field is not None:
                v.append(np.atleast_2d(field.T).T)
                c.extend(names)

        add_field(self.y, self.y_cols)
        add_field(self.x, self.x_cols)
        add_field(self.z, [Z] if self.z is not None else [])
        add_field(self.t, self.t_cols)
        add_field(self.msidx, [MSIDX])
        add_field(self.s, [S])
        add_field(self.pw, [PW])
        add_field(self.w, self.w_cols)

        return pd.DataFrame(np.hstack(v), columns=c, index=self.index())

    def __repr__(self):
        return self.to_df().__repr__()

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        df.index = pd.to_datetime(df.index)
        ts = datetime_to_int(df.index)
        
        # Sort columns by prefix to ensure contiguous slices
        def extract_prefix(name):
            return ''.join(filter(str.isalpha, name))
        df = df[sorted(df.columns, key=extract_prefix)]

        columns = np.array(df.columns)
        prefix_slices = contiguous_prefix_slices(columns)

        y = df.values[:, prefix_slices[Y]]
        y_cols = columns[prefix_slices[Y]]

        x = x_cols = t = t_cols = z = msidx = None

        if X in prefix_slices:
            x = df.values[:, prefix_slices[X]]
            x_cols = columns[prefix_slices[X]]
        if T in prefix_slices:
            t = df.values[:, prefix_slices[T]]
            t_cols = columns[prefix_slices[T]]
        if Z in prefix_slices:
            z = df.values[:, prefix_slices[Z]][:, 0]
        if MSIDX in prefix_slices:
            msidx = df.values[:, prefix_slices[MSIDX].start]

        return cls(ts=ts, y=y, x=x, z=z, t=t, msidx=msidx,
                   y_cols=y_cols, x_cols=x_cols, t_cols=t_cols)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]
        return Data(
            ts=self.ts[idx], y=self.y[idx],
            x=self.x[idx] if self.x is not None else None,
            z=self.z[idx] if self.z is not None else None,
            t=self.t[idx] if self.t is not None else None,
            s=self.s[idx], pw=self.pw[idx], w=self.w[idx],
            msidx=self.msidx[idx],
            y_cols=self.y_cols, x_cols=self.x_cols,
            t_cols=self.t_cols, w_cols=self.w_cols)

    def copy(self):
        return copy.deepcopy(self)

    def after(self, ts: int):
        if self.n == 0:
            return self
        idx = np.searchsorted(self.ts, ts, side='right')
        return self[idx:] if idx < self.n else self[:0]

    def before(self, ts: int):
        if self.n == 0:
            return self
        idx = np.searchsorted(self.ts, ts, side='left')
        return self[:idx] if idx > 0 else self[:0]

    def between(self, ts_lower: int, ts_upper: int):
        if self.n == 0:
            return self
        mask = (self.ts >= ts_lower) & (self.ts <= ts_upper)
        if not mask.any():
            return self[:0]
        first, last = np.where(mask)[0][[0, -1]]
        return self[first:last+1]

    def random_segment(self, burn_fraction: float, min_burn_points: int):
        if self.n == 0:
            return self
        start = np.random.randint(min_burn_points, max(int(self.n * burn_fraction), min_burn_points + 1))
        end = np.random.randint(min_burn_points, max(int(self.n * burn_fraction), min_burn_points + 1))
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

        self.ts = np.hstack((self.ts, data.ts))
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

        offset = self.msidx[-1] - data.msidx[0] + 1
        self.msidx = np.hstack((self.msidx, data.msidx + offset))

        self.n, self.p = self.y.shape
        self._process_msidx()
        return self

    def as_dict(self, is_live:bool = False):
        return {'y': self.y, 'x': self.x, 't': self.t, 'z': self.z, 'msidx': self.msidx, 'is_live': is_live}

    def input_at(self, idx: int = None):
        if idx is None:
            idx = self.n - 1
        start = self.msidx_start_lookup[idx]
        return {
            'y': self.y[start:idx+1],
            'x': self.x[start:idx+1] if self.x is not None else None,
            't': self.t[start:idx+1] if self.t is not None else None,
            'z': self.z[start:idx+1] if self.z is not None else None,
            'msidx': self.msidx[start:idx+1]
        }

    def at(self, idx: int = None):
        if idx is None:
            idx = self.n - 1
        start = self.msidx_start_lookup[idx]
        return self[start:idx+1]


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
        ts.sort()            
        self.folds_ts = split_ts(ts, k_folds=k_folds)
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
        if self.folds_ts is None:
            raise ValueError("Need to split before getting the split")
        
        train_dataset = Dataset()
        test_dataset = Dataset()

        for key, data in self.items():
            train_data, test_data = data.split(
                                                test_fold_idx = test_fold_idx,
                                                burn_fraction = burn_fraction,
                                                min_burn_points = min_burn_points,
                                                seq_path = seq_path,
                                                folds_ts = self.folds_ts
                                                )
            train_dataset.add(key, train_data)
            test_dataset.add(key, test_data)    
        
        return train_dataset, test_dataset


if __name__ == '__main__':


    import tm

    def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01'):
        x=np.random.normal(0,scale,n)
        y=a+b*x+np.random.normal(0,scale,n)
        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
        return data

    df1 = linear(n=10, a=0, b=0.1, start_date='2000-01-01')
    df2 = linear(n=10, a=0, b=0.1, start_date='2000-01-01')
    df1['msidx'] = 666
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

    t = tm.utils.msidx_to_table(data1.msidx)
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








