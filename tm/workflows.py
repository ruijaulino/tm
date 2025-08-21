
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm
import time
from tm.containers import Data, Dataset
from tm.model import Model, ModelSet
from tm.base import BaseModel
from tm.transforms.abstract import Transform, Transforms
from tm.post_process import Paths
from tm.constants import *    
# Dict of ModelPipeStack
# Objective here is to handle for several data in a dataset where
# each one has a ModelPipeStack associated


def load_model(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


# changes dataset in place
def cvbt_path(
            dataset:Union[Data, Dataset], 
            modelset:Union[Model, ModelSet],
            k_folds:int = 4, 
            seq_path:bool = False, 
            start_fold:int = 0, 
            burn_fraction:float = 0.1, 
            min_burn_points:int = 3,
            view_after_estimate:bool = False
            ) -> Dataset:
    

    dataset.split_ts(k_folds)                  
    
    start_fold = max(1, start_fold) if seq_path else start_fold     


    for fold_index in range(start_fold, k_folds):  
        # build train test split
        train_dataset, test_dataset = dataset.split(
                                                    fold_index, 
                                                    burn_fraction = burn_fraction, 
                                                    min_burn_points = min_burn_points, 
                                                    seq_path = seq_path
                                                    )

        # copy model pipe
        # this is an operation without much overhead
        tmp_model_set = copy.deepcopy(modelset) 
        # train model
        tmp_model_set.estimate(train_dataset)
        # estimate model - the results will be written in dataset because .between uses simple indexing
        if view_after_estimate:
            tmp_model_set.view()
        tmp_model_set.evaluate(test_dataset) #
        # set performance on dataset (maybe not needed because it will be already overriten! CHECK THIS)
    
    return dataset    

def cvbt(
        dataset:Union[Data, Dataset], 
        modelset:Union[Model,ModelSet],
        n_paths: int = 5,
        k_folds:int = 4, 
        seq_path:bool = False, 
        start_fold:int = 0, 
        burn_fraction:float = 0.1, 
        min_burn_points:int = 3,
        view_after_estimate:bool = False
        ) -> Paths:
    paths = Paths()
    for p in tqdm.tqdm(range(n_paths)):
        path = cvbt_path(
                            dataset = dataset.copy(), 
                            modelset = modelset.copy(),
                            k_folds = k_folds, 
                            seq_path = seq_path, 
                            start_fold = start_fold, 
                            burn_fraction = burn_fraction, 
                            min_burn_points = min_burn_points,
                            view_after_estimate = view_after_estimate
                            )
        paths.add(path)
    return paths


def test_w_model():

    def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01'):
        x=np.random.normal(0,scale,n)
        y=a+b*x+np.random.normal(0,scale,n)
        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
        return data

    df = linear(n=1000, a=0, b=0.5, start_date='2000-01-01')

    df_test = linear(n=10, a=0, b=0.5, start_date='2000-01-01')
    tmp = df_test['y1'].values.ravel()
    tmp[-1] = Y_LIVE_VALUE
    df_test['y1'] = tmp

    data = Data.from_df(df)

    data_test = Data.from_df(df_test)

    import tm
    base_model = tm.base.BayesLinRegr()
    c = 0.01

    seq_fees = False

    alloc = tm.allocation.Optimal(c = c, seq_w = seq_fees)

    model = Model(base_model = base_model, allocation = alloc)

    paths = cvbt(
            dataset = data, 
            modelset = model,            
            n_paths = 1,
            k_folds = 4, 
            seq_path = False, 
            start_fold = 0, 
            burn_fraction = 0.1, 
            min_burn_points = 3,
            view_after_estimate = False

            )
    
    model.estimate(data)
    out = model.live(data_test)
    print('LIVE: ', out, type(out))


    paths.post_process(pct_fee = c, seq_fees = seq_fees)


def test_w_model_set():

    def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01'):
        x=np.random.normal(0,scale,n)
        y=a+b*x+np.random.normal(0,scale,n)
        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
        return data

    df = linear(n=1000, a=0, b=0.5, start_date='2000-01-01')

    # data = Data.from_df(df)

    dataset = Dataset()
    dataset.add('data', df)


    import tm
    base_model = tm.base.BayesLinRegr()
    model = Model(base_model = base_model)

    modelset = ModelSet()
    modelset.add('data', model)


    paths = cvbt(
            dataset = dataset, 
            modelset = modelset,            
            n_paths = 1,
            k_folds = 4, 
            seq_path = False, 
            start_fold = 0, 
            burn_fraction = 0.1, 
            min_burn_points = 3,
            view_after_estimate = False

            )
    
    paths.post_process()





if __name__ == '__main__':
    test_w_model()
    #test_w_model_set()
