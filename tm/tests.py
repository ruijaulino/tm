
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm
from tm import cvbt, cvbt_path, Paths, Data, Dataset, ModelPipe


# generators
def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01', add_msidx = False):
    x=np.random.normal(0,scale,n)
    y=a+b*x+np.random.normal(0,scale,n)
    y[-1] = 0
    if add_msidx:
        msidx = np.array_split(np.ones(n, dtype = int), 4)
        for i in range(len(msidx)):
            msidx[i] += i
        msidx = np.hstack(msidx)
        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None], msidx[:,None])),columns=['y1','x1', 'msidx'],index=dates)
    else:

        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)

    return data


def test_estimate_1():

    df = linear(n=1000,a=0,b=0.5,start_date='2000-01-01', add_msidx = True)
    dataset = Dataset()
    dataset.add('dataset', df)

    from tm.models import BayesLinRegr
    model = BayesLinRegr()

    model_pipe = ModelPipe()
    model_pipe.add('dataset', model)
    model_pipe.estimate(dataset)
    # model_pipe.view()

    # evaluate
    df1 = linear(n=1000,a=0,b=0.5,start_date='2000-01-01', add_msidx = True)
    dataset1 = Dataset()
    dataset1.add('dataset', df1)

    model_pipe.evaluate(dataset1)
    print(dataset1)


    out = model_pipe.live(dataset1)
    print(out)
    



def test_cvbt_path():

    df = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    dataset = Dataset()
    dataset.add('dataset', df)


    from stm.models import LR
    model = LR()

    model_pipe = ModelPipe()
    model_pipe.add('dataset', model)

    cvbt_path(
                dataset = dataset, 
                model_pipe = model_pipe
                )
    

    print(dataset)
    plt.plot(np.cumsum(dataset['dataset'].s))
    plt.show()


def test_cvbt_path_w_master_model():

    dataset = Dataset()

    df1 = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    dataset.add('dataset1', df1)

    df2 = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    dataset.add('dataset2', df2)

    from stm.models import LR
    model = LR()

    model_pipe = ModelPipe(master_model = model)


    from stm.transforms import Transforms, ScaleTransform
    transforms = Transforms(x_transform = ScaleTransform(), y_transform = ScaleTransform())
    model_pipe.add('dataset1', transforms = transforms)
    model_pipe.add('dataset2', transforms = transforms)


    dataset = cvbt_path(
                dataset = dataset, 
                model_pipe = model_pipe
                )
    
    print(dataset)
    for k, data in dataset.items():
        plt.plot(np.cumsum(data.s))
    plt.show()


def test_cvbt():
    df = linear(n=5000,a=0,b=0.1,start_date='2000-01-01')
    dataset = Dataset()
    dataset.add('dataset', df)



    from stm.models import LR
    model = LR()

    model_pipe = ModelPipe()
    model_pipe.add('dataset', model)


    paths = cvbt(
        dataset = dataset, 
        model_pipe = model_pipe,
        )
    paths.post_process()

    

def test_estimate_pm_1():

    dataset = Dataset()

    df1 = linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01')
    dataset.add('dataset1', df1)

    df2 = linear(n=1000,a=0,b=0.1, scale = 0.02,start_date='2000-01-01')
    dataset.add('dataset2', df2)

    from stm.models import LR
    from stm.portfolio_models import IdlePortfolioModel, InvVolPortfolioModel, StratStatPortfolioModel
    model = LR()
    portfolio_model = StratStatPortfolioModel('sharpe')
    model_pipe = ModelPipe(portfolio_model = portfolio_model)
    model_pipe.add('dataset1', model = LR())
    model_pipe.add('dataset2', model = LR())

    
    model_pipe.estimate(dataset)

    model_pipe.evaluate(dataset)
    print(dataset)
    model_pipe.view()


def test_estimate_pm_2():

    dataset = Dataset()

    df1 = linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01')
    dataset.add('dataset1', df1)

    df2 = linear(n=1000,a=0,b=0.1, scale = 0.02,start_date='2000-01-01')
    dataset.add('dataset2', df2)

    from stm.models import LR
    from stm.portfolio_models import IdlePortfolioModel, InvVolPortfolioModel, StratStatPortfolioModel

    model = LR()
    portfolio_model = None#StratStatPortfolioModel('sharpe')
    model_pipe = ModelPipe(portfolio_model = portfolio_model)
    model_pipe.add('dataset1', model = LR())
    model_pipe.add('dataset2', model = LR())
    
    paths = cvbt(
        dataset = dataset, 
        model_pipe = model_pipe,
        )

    paths.portfolio_post_process()

    #for path in paths:
    #    print(path)

def test_transforms_1():

    df = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    dataset = Dataset()
    dataset.add('dataset', df)

    from stm.models import LR
    model = LR()

    from stm.transforms import Transforms, ScaleTransform
    transforms = Transforms()
    transforms.add('x', ScaleTransform())
    transforms.add('y', ScaleTransform())


    print(dataset)

    model_pipe = ModelPipe()
    model_pipe.add('dataset', model)
    model_pipe.estimate(dataset)

    print(dataset)
    model_pipe.view()


def test_transforms_2():


    df = linear(n=3000,a=0,b=0.1,start_date='2000-01-01')
    dataset = Dataset()
    dataset.add('dataset', df)


    from stm.models import LR
    model = LR()

    #from stm.transforms import Transforms, ScaleTransform
    #transforms = Transforms(ScaleTransform(), ScaleTransform())

    model_pipe = ModelPipe()
    model_pipe.add('dataset', model)


    # import cProfile
    # import pstats


    # profiler = cProfile.Profile()
    # profiler.enable()
    # cvbt_path(
    #     dataset = dataset, 
    #     model_pipe = model_pipe,
    #     )
    # profiler.disable()

    # stats = pstats.Stats(profiler)
    # stats.strip_dirs().sort_stats("tottime").print_stats(10)  # Sort by total time, show top 10

    
    from line_profiler import LineProfiler


    from stm.workflows import ModelPipeUnit
    from stm.containers import Data 

    profiler = LineProfiler()
    #profiler.add_function(cvbt_path)
    #profiler.add_function(Data.at)
    profiler.add_function(ModelPipeUnit.evaluate)
    #profiler.add_function(ModelPipeUnit.get_weight)
    
    profiler.enable()
    cvbt_path(
        dataset = dataset, 
        model_pipe = model_pipe,
        )
    profiler.disable()
    profiler.print_stats()




if __name__ == '__main__':
    test_estimate_1()
    
    # test_cvbt_path()   
    # test_cvbt_path_w_master_model()
    
    # test_cvbt()
    # test_transforms_2()
    
    # test_estimate_pm_1()
    # test_transforms_1()
    # test_estimate_pm_1()
    # test_estimate_pm_2()
    # test_transforms_1()
