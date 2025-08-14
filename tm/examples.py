
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm

from tm import cvbt, cvbt_path, Paths, Data, Dataset, ModelPipe

def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01'):
    x=np.random.normal(0,scale,n)
    y=a+b*x+np.random.normal(0,scale,n)
    dates=pd.date_range(start_date,periods=n,freq='D')
    data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
    return data



def test_state_gaussian():
    # generate data
    n = 1000
    y = np.random.normal(0, 0.01, n)
    z = np.random.choice([0,1], n)
    y[z == 0] += 0.01
    
    dates = pd.date_range('2000-01-01', periods = n, freq = 'D')
    df = pd.DataFrame(np.hstack((y[:,None],z[:,None])), columns = ['y1','z'], index = dates)
    from tm.models import StateGaussian

    dataset = Dataset()
    dataset.add('dataset', df)
    model_pipe = ModelPipe()
    model_pipe.add('dataset', model = StateGaussian())
    paths = cvbt(
         dataset = dataset, 
         model_pipe = model_pipe,
         )

    paths.post_process()


if __name__ == '__main__':

    test_state_gaussian()
    exit(0)


    dataset = Dataset()

    df1 = linear(n=3000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01')
    dataset.add('dataset1', df1)

    df2 = linear(n=2000,a=0,b=0.1, scale = 0.02,start_date='2000-06-01')
    dataset.add('dataset2', df2)

    from tm.models import BayesLR
    from tm.portfolio_models import IdlePortfolioModel, InvVolPortfolioModel, StratStatPortfolioModel
    
    from tm.transforms import Transforms, ScaleTransform
    transforms = Transforms(x_transform = ScaleTransform(), y_transform = ScaleTransform())


    portfolio_model = StratStatPortfolioModel('invvol')
    model_pipe = ModelPipe(portfolio_model = portfolio_model)
    model_pipe.add('dataset1', model = BayesLR(), transforms = transforms)
    model_pipe.add('dataset2', model = BayesLR(), transforms = transforms)
    
    paths = cvbt(
         dataset = dataset, 
         model_pipe = model_pipe,
         )

    paths.portfolio_post_process()

    # estimate
    model_pipe.estimate(dataset)

    print('LIVE')
    # live
    dataset = Dataset()
    # create dataset for live
    df1 = linear(n=2,a=0,b=0.1, scale = 0.01,start_date='2000-01-01')
    df1['y1'] = 0
    dataset.add('dataset1', df1)

    df2 = linear(n=2,a=0,b=0.1, scale = 0.02,start_date='2000-06-01')
    df2['y1'] = 0    
    dataset.add('dataset2', df2)    

    out = model_pipe.live(dataset)
    print(out)
    