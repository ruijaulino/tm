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

    def copy(self):
        return copy.deepcopy(self)

    def set_base_model(self, base_model:BaseModel):
        self.base_model = copy.deepcopy(base_model)

    def set_transforms(self, transforms:Transforms):
        self.transforms = copy.deepcopy(transforms)

    def set_allocation(self, allocation:Allocation):
        self.allocation = copy.deepcopy(allocation)

    def view(self, plot = False, transforms_only = False, **kwargs):
        if transforms_only:
                self.transforms.view()
        else:
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
        # get predictive distribution on training data
        mu, cov = self.base_model.posterior_predictive(**data.as_dict())
        if mu.ndim == 1:
            mu = mu.reshape((mu.size, 1))
        if cov.ndim == 1:
            cov = cov.reshape((cov.size, 1, 1))
        self.allocation.estimate(mu, cov)  

    def estimate(self, data:Data):        
        # estimate transforms
        self.estimate_transforms(data)
        # apply to training data
        transformed_data = self.transform(data)        
        # the arguments passed are like model.estimate(y, x, z, t, msidx) 
        self.estimate_base_model(transformed_data)
        self.estimate_allocation(transformed_data)

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
        
        if not data.empty:
            transformed_data = self.transform(data)

            # compute the posterior predictive on data
            # this will generate arrays mu and cov that correspond to each point in data.y
            mu, cov = self.base_model.posterior_predictive(**transformed_data.as_dict())        
            if mu.ndim == 1:
                mu = mu.reshape((mu.size, 1))
            if cov.ndim == 1:
                cov = cov.reshape((cov.size, 1, 1))
            w = self.allocation.get_weight(mu, cov, cost_scale = self.transforms.cost_scale())        
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
        transformed_data.y[-1] = data.y[-1] # restore value        
        use_t = transformed_data.t is not None
        # check data format for live execution
        assert (transformed_data.y[-1] == Y_LIVE_VALUE).all(), f"In a live setting, the last observation of y must have been generated artificially with {Y_LIVE_VALUE}"    
        if use_t:
            transformed_data.t[-1] = data.t[-1] # restore value        
            assert (t[-1] == T_LIVE_VALUE).all(), f"In a live setting, the last observation of t must have been generated artificially with {T_LIVE_VALUE}"    
        # it does not matter that we are making more computations than needed here because it
        # is a fast operation done only once when execution live
        mu, cov = self.base_model.posterior_predictive(**transformed_data.as_dict(is_live = True))
        if mu.ndim == 1:
            mu = mu.reshape((mu.size, 1))
        if cov.ndim == 1:
            cov = cov.reshape((cov.size, 1, 1))        
        w = self.allocation.get_weight(mu, cov, cost_scale = self.transforms.cost_scale(), live = True, prev_w = prev_w)        
        return np.atleast_1d(w)


class ModelSet(dict):
    def __init__(self, master_model:Model = None, ensemble_model:EnsembleModel = None, individual_alloc_norm:bool = False):
        self.master_model = master_model        
        self.ensemble_model = ensemble_model
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
        assert self.master_model is None, "setting a model on a key a master model is defined"
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
        if self.master_model:
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
            # estimate master model
            self.master_model.estimate_base_model(data)            
            # estimate allocation
            if not self.individual_alloc_norm:
                self.master_model.estimate_allocation(data)    

            # set base models and estimate allocation
            for k, data in dataset.items():
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






