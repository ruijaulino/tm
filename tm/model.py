from __future__ import annotations

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
from tm.containers import Data, Dataset
from tm.constants import *

# there is a circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tm.ensemble import EnsembleModel

# Model class
# a model is a set of operations: transform, probabilistic modelling and allocation strategy




class Model:

    def __init__(self, base_model:BaseModel = None, transforms:Transforms = None, allocation:Allocation = None):
        self.base_model = base_model
        self.transforms = transforms
        if not self.transforms: self.transforms = Transforms()
        self.allocation = allocation
        if not self.allocation: self.allocation = Optimal()
        if hasattr(self.base_model, 'use_m2'):
            self.allocation.set_use_m2(use_m2 = self.base_model.use_m2)
        self.needed_columns = None

    def copy(self):
        return copy.deepcopy(self)

    def set_base_model(self, base_model:BaseModel):
        self.base_model = copy.deepcopy(base_model)

    def set_transforms(self, transforms:Transforms):
        self.transforms = copy.deepcopy(transforms)

    def set_allocation(self, allocation:Allocation):
        self.allocation = copy.deepcopy(allocation)

    def view(self, plot = False, **kwargs):
        print()
        print()
        print("* Model *")
        self.base_model.view(plot = plot)
        print()
        self.transforms.view()
        print() 
        self.allocation.view()
        print()
        print()

    def estimate_transforms(self, data:Data):
        self.transforms.estimate(data)
        
    def transform(self, data:Data):
        return self.transforms.transform(data)

    def estimate_base_model(self, data:Data):
        self.base_model.estimate(**data.as_dict())
        
    def estimate_allocation(self, data:Data):
        # get predictive distribution on training data to get a measure of weight variation
        mu, cov = self.base_model.posterior_predictive(**data.as_dict())
        # transform back into appropriate scale
        if mu.ndim == 1:
            mu = mu.reshape((mu.size, 1))
        if cov.ndim == 1:
            cov = cov.reshape((cov.size, 1, 1))
        # put back in the original scale
        mu, cov = self.transforms.scale_back_moments(mu, cov)
        self.allocation.estimate(mu, cov)  

    def estimate(self, data:Data):        
        
        # store data columns to check and filter on evaluation and live
        self.needed_columns = data.columns

        # estimate transforms
        self.estimate_transforms(data)
        # apply to training data
        transformed_data = self.transform(data)        
        # the arguments passed are like model.estimate(y, x, z, t, msidx) 
        self.estimate_base_model(transformed_data)
        self.estimate_allocation(transformed_data)

    def evaluate(self, data:Data):
        """Evaluate the model using the test data and return performance metrics."""
        # this will change fields s, weight_* in data object inplace        
        # evaluation of a model can be an expensive operation and so this tries to
        # be more efficient!        
        
        # apply transforms on whole data (it creates a copy if transformations are applied)
        # this prevents too much copies when iterating over the arrays
        
        assert all(e in self.needed_columns for e in data.columns), "data for evaluate does not contain the needed columns"
        data_f = data._get_columns(self.needed_columns) # filter because it may come with more columns in some special cases

        if not data_f.empty:
            transformed_data = self.transform(data_f)

            # compute the posterior predictive on data
            # this will generate arrays mu and cov that correspond to each point in data.y
            mu, cov = self.base_model.posterior_predictive(**transformed_data.as_dict())      
            if mu.ndim == 1:
                mu = mu.reshape((mu.size, 1))
            if cov.ndim == 1:
                cov = cov.reshape((cov.size, 1, 1))
            # transform back into appropriate scale
            mu, cov = self.transforms.scale_back_moments(mu, cov)
            w = self.allocation.get_weight(mu, cov)        
            # set on original data!
            data.w[:] = w
            data.s[:] = np.einsum('ij,ij->i', w, data.y)
        
        return data

    def live(self, data:Data, prev_w = None, **kwargs):
        # live is implemented on it's own although it performs
        # similar computations as in evaluate
        # note that data must be provided in a defined way for live evaluation

        assert all(e in self.needed_columns for e in data.columns), "data for evaluate does not contain the needed columns"
        data_f = data._get_columns(self.needed_columns) # filter because it may come with more columns in some special cases

        # apply transforms
        transformed_data = self.transforms.transform(data_f)
        transformed_data.y[-1] = data.y[-1] # restore value        
        use_t = transformed_data.t is not None
        # check data format for live execution
        assert (transformed_data.y[-1] == Y_LIVE_VALUE).all(), f"In a live setting, the last observation of y must have been generated artificially with {Y_LIVE_VALUE}"    
        if use_t:
            transformed_data.t[-1] = data_f.t[-1] # restore value        
            assert (t[-1] == T_LIVE_VALUE).all(), f"In a live setting, the last observation of t must have been generated artificially with {T_LIVE_VALUE}"    
        # it does not matter that we are making more computations than needed here because it
        # is a fast operation done only once when execution live
        mu, cov = self.base_model.posterior_predictive(**transformed_data.as_dict(is_live = True))
        if mu.ndim == 1:
            mu = mu.reshape((mu.size, 1))
        if cov.ndim == 1:
            cov = cov.reshape((cov.size, 1, 1))        
        # transform back into appropriate scale
        mu, cov = self.transforms.scale_back_moments(mu, cov)
        w = self.allocation.get_weight(mu, cov, live=True)  
        return np.atleast_1d(w)
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class ModelSet(dict):
    def __init__(self, master_model:Model = None, ensemble_model:EnsembleModel = None, master_models_map:List = None, individual_alloc_norm:bool = False):
        self.master_model = master_model        
        self.ensemble_model = ensemble_model
        self.master_models_map = master_models_map # [{'master_model':Model, 'apply_to':[], 'columns':['y1', 'x1','x2','z1']}] - needs to exhaust list in data...
        self.individual_alloc_norm = individual_alloc_norm
        # after a model is run this variable stores the dataset 
        # that was used to estimate the model!    
        self.estimation_dataset = None

    def copy(self):
        return copy.deepcopy(self)

    def view(self, plot = False, **kwargs):
        print()
        print("******* ModelSet *******")
        print()
        if self.ensemble_model:
            self.ensemble_model.view(plot = plot)
        print()
        for k, m in self.items():
            print()
            print(f"-> For key {k}")
            m.view(plot = plot)
        print("*************************")


    def add(self, key:str, model:Model = None):
        assert self.master_model is None, "setting a model on a key when master model is defined"
        assert self.master_models_map is None, "setting a model on a key when master models map is defined"
        if key not in self:
            self[key] = model
        else:
            print(f'Warning: a model was already set for key {key}')
    
    def estimate(self, dataset:Dataset, store_details:bool = True):                
        
        assert isinstance(dataset, Dataset), "ModelSet can only be used with a Dataset object"

        
        # estimate ensemble_model, may do nestec cv here
        if self.ensemble_model:
            # create a model set without the ensemble model
            tmp_modelset = self.copy()
            tmp_modelset.ensemble_model = None # set to None
            self.ensemble_model.estimate(dataset, tmp_modelset)
        
        # estimate models
        if self.master_models_map:

            # if several models apply to the same dataset (because they may be trained with a larger dataset)
            #     we need to mix predictive distribution somehow

            # check if data keys are covered
            covered_keys = []
            for e in self.master_models_map: covered_keys += e.get('apply_to', [])
            covered_keys = list(set(covered_keys))
            for k, _ in dataset.items(): assert k in covered_keys, "not all elements in dataset assigned to a master_model"

            for elem in self.master_models_map:
                apply_to = elem.get('apply_to')
                master_model = elem.get('master_model')
                columns = elem.get('columns')

                data = None
                for k, data_ in dataset.items():
                    if k in apply_to:

                        if columns:
                            # check if data_ contain all needed columns
                            assert all(e in data_.columns for e in columns), "data does not contain the needed columns"
                            data_ = data_._get_columns(columns)

                        # copy the master model
                        k_model = master_model.copy()
                        k_model.estimate_transforms(data_)                
                        # transforms
                        transformed_data_ = k_model.transform(data_)                                
                        if not data: 
                            data = transformed_data_
                        else:
                            data.stack(transformed_data_, allow_both_empty = True)
                        # add to key
                        self[k] = k_model

                if data.empty: raise Exception('data is empty. should not happen')
                
                # store data columns to check and filter on evaluation and live
                master_model.needed_columns = data.columns
                
                # estimate master model
                master_model.estimate_base_model(data)            
                # estimate allocation
                if not self.individual_alloc_norm:
                    master_model.estimate_allocation(data)    

                # set base models and estimate allocation
                for k, data in dataset.items():
                    if k in apply_to:
                        self[k].set_base_model(master_model.base_model)
                        # set the global one (even if not estimated yet...)
                        self[k].set_allocation(master_model.allocation)
                        # estimate allocation for each one
                        if self.individual_alloc_norm:
                            self[k].estimate_allocation(self[k].transform(data))

        elif self.master_model:
            # if a master model is present, apply transforms, stack the data, and estimate it
            data = None
            for k, data_ in dataset.items():
                # copy the master model
                k_model = self.master_model.copy()
                k_model.estimate_transforms(data_)                
                # transforms
                transformed_data_ = k_model.transform(data_)                                
                if not data: 
                    data = transformed_data_
                else:
                    data.stack(transformed_data_, allow_both_empty = True)
                # add to key
                self[k] = k_model

            if data.empty: raise Exception('data is empty. should not happen')
            # store data columns to check and filter on evaluation and live
            self.master_model.needed_columns = data.columns
            # estimate master model
            self.master_model.estimate_base_model(data)            
            # estimate allocation
            if not self.individual_alloc_norm:
                self.master_model.estimate_allocation(data)    

            # set base models and estimate allocation
            for k, data in dataset.items():
                self[k].needed_columns = self.master_model.needed_columns
                self[k].set_base_model(self.master_model.base_model)
                # set the global one (even if not estimated yet...)
                self[k].set_allocation(self.master_model.allocation)
                # estimate allocation for each one
                if self.individual_alloc_norm:
                    self[k].estimate_allocation(self[k].transform(data))

        else:
            for k, data in dataset.items():
                assert k in self, "dataset contains a key that is not defined in ModelSet. Exit.."
                self[k].estimate(data)        


        # when we train a final model we can store the dataset that was used to estimate the
        # model. If future checks are needed we can just run inference again with it!
        if store_details:
            self.estimation_dataset = copy.deepcopy(dataset)

    def evaluate(self, dataset:Dataset):
        # dataset_dict is a dict of dataset
        for k, data in dataset.items():
            assert k in self, "dataset contains a key that is not defined in ModelSet. Exit.."                        
            self[k].evaluate(data)   
        # set portfolio weight on dataset                
        if self.ensemble_model:
            for k, data in dataset.items():
                data.pw[:] *= self.ensemble_model.get(k)        
        return dataset

    def live(self, dataset:Dataset):
        # to be used in a live setting

        out = {}
        for k, data in dataset.items():
            assert k in self, "dataset contains a key that is not defined in ModelSet. Exit.."                        
            out.update({k: {'w':self[k].live(data), 'w_cols':data.w_cols}})

        # set portfolio weight on dataset                
        for k, _ in dataset.items():
            tmp = 1
            if self.ensemble_model:
                tmp = self.ensemble_model.get(k)
            out[k].update({'pw':tmp})

        return out


    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)








if __name__ == '__main__':

    # generate some data
    n = 1000
    x = np.random.normal(0, 1, n)
    a = 0
    b = 0.2
    y = a+b*x+np.random.normal(0,0.1,n)
    


    df1 = pd.DataFrame()
    df1['x'] = x
    df1['y'] = y
    df1.index = pd.date_range('2000-01-01', freq = 'D', periods = n)
    df1.plot.scatter('x', 'y')
    plt.show()


    data = Data.from_df(df1)
    print(data)


    import tm
    base_model = tm.base.LinRegr()
    alloc = tm.allocation.Optimal()
    transforms = tm.transforms.Transforms(
                            y_transform = tm.transforms.ScaleTransform(),
                            x_transform = tm.transforms.ScaleTransform()
                            )

    model = Model(base_model = base_model, allocation = alloc, transforms = transforms)    
    model.estimate(data)
    model.view()
    model.evaluate(data)
    print('-------')
    plt.plot(data.w)
    plt.show()


    pass




