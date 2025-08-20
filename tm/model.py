import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm
import time
from tm.base import BaseModel
from tm.allocation import Allocation, Optimal
from tm.transforms.abstract import Transforms
from tm.containers import Data
from tm.constants import *

# Model class
# a model is a set of operations: transform, probabilistic modelling and allocation strategy

class Model:

    def __init__(self, base_model:BaseModel = None, transforms:Transforms = None, allocation:Allocation = None):
        self.base_model = base_model
        self.transforms = transforms
        if not self.transforms: self.transforms = Transforms()
        self.allocation = allocation
        if not self.allocation: self.allocation = Optimal()
    
    def copy(self):
        return copy.deepcopy(self)

    def set_model(self, base_model:BaseModel):
        self.base_model = copy.deepcopy(base_model)

    def view(self, transforms_only = False):
        if transforms_only:
                self.transforms.view()
        else:
            print("* Model *")
            self.base_model.view()
            print()
            self.transforms.view()
            print()
            self.allocation.view()
        print()
        print()

    def estimate(self, data:Data):        
        # estimate transforms
        self.transforms.estimate(data)
        # apply to training data
        data = self.transforms.transform(data)        
        # the arguments passed are like model.estimate(y, x, z, t, msidx) 
        self.base_model.estimate(**data.as_dict())
        # get predictive distribution on training data
        mu, cov = self.base_model.posterior_predictive(**data.as_dict())
        if mu.ndim == 1:
            mu = mu.reshape((mu.size, 1))
        if cov.ndim == 1:
            cov = cov.reshape((cov.size, 1, 1))
        self.allocation.estimate(mu, cov)        

    def post_estimate(self, data:Data):
        # adjust parameters after master is estimated!
        pass

    def evaluate(self, data:Data):
        """Evaluate the model using the test data and return performance metrics."""
        # this will change fields s, weight_* in data object inplace        
        # evaluation of a model can be an expensive operation and so this tries to
        # be more efficient!        
        
        # apply transforms on whole data (it creates a copy if transformations are applied)
        # this prevents too much copies when iterating over the arrays
        transformed_data = self.transforms.transform(data)

        # compute the posterior predictive on data
        # this will generate arrays mu and cov that correspond to each point in data.y
        mu, cov = self.base_model.posterior_predictive(**transformed_data.as_dict())        
        if mu.ndim == 1:
            mu = mu.reshape((mu.size, 1))
        if cov.ndim == 1:
            cov = cov.reshape((cov.size, 1, 1))
        w = self.allocation.get_weight(mu, cov)        
        # set on original data!
        data.w[:] = w
        data.s[:] = np.einsum('ij,ij->i', w, data.y)
        
        return data

    def live(self, data:Data, prev_w = None, **kwargs):
        # live is implemented on it's own although it performs
        # similar computations as in evaluate
        # note that data must be provided in a defined way for live evaluation

        # apply transforms
        transformed_data = self.transforms.transform(data)
        use_t = transformed_data.t is not None
        # check data format for live execution
        assert (transformed_data.y[-1] == Y_LIVE_VALUE).all(), "In a live setting, the last observation of y must have been generated artificially with zeros.."    
        if use_t:
            assert (t[-1] == T_LIVE_VALUE).all(), "In a live setting, the last observation of t must have been generated artificially with zeros.."    
        # it does not matter that we are making more computations than needed here because it
        # is a fast operation done only once when execution live
        mu, cov = self.base_model.posterior_predictive(**transformed_data.as_dict())
        if mu.ndim == 1:
            mu = mu.reshape((mu.size, 1))
        if cov.ndim == 1:
            cov = cov.reshape((cov.size, 1, 1))        
        w = self.allocation.get_weight(mu, cov, live = True, prev_w = prev_w)        
        return w


if __name__ == '__main__':

    def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01'):
        x=np.random.normal(0,scale,n)
        y=a+b*x+np.random.normal(0,scale,n)
        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
        return data

    df_train = linear(n=1000, a=0, b=0.5, start_date='2000-01-01')
    df_test = linear(n=1000, a=0, b=0.5, start_date='2000-01-01')

    tmp = df_test['y1'].values.ravel()
    tmp[-1] = Y_LIVE_VALUE
    df_test['y1'] = tmp 
    
    tmp = df_test['x1'].values.ravel()
    tmp[-1] = np.max(tmp)
    df_test['x1'] = tmp 


    data_train = Data.from_df(df_train)
    data_test = Data.from_df(df_test)


    print(data_train)
    print('-----------------')




    from tm.base_models import BayesLinRegr
    regr = BayesLinRegr()

    model = Model(base_model = regr, transforms = None, allocation = None)
    model.estimate(data_train)
    model.view()
    data_test = model.evaluate(data_test)

    plt.plot(data_test.w)
    plt.show()

    plt.plot(np.cumsum(data_test.s))
    plt.show()

    print(data_test)
    print(model.live(data_test))








