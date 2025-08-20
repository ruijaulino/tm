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


class ModelSet(dict):
    def __init__(self, master_model:Model = None, ensemble_model:EnsembleModel = None):
        self.master_model = master_model        
        self.ensemble_model = ensemble_model
        # after a model is run this variable stores the dataset 
        # that was used to estimate the model!    
        self.estimation_dataset = None

    def add(self, key:str, model:Model = None):
        if key not in self:
            self[key] = model

    def copy(self):
        return copy.deepcopy(self)

    def view(self):
        print()
        print("******* ModelSet *******")
        print()
        if self.ensemble_model:
            self.ensemble_model.view()
        print()
        if self.master_model:
            print("* Master Model *")
            self.master_model.view()
            for k, m in self.items():
                print()
                print(f"-> For key {k}")
                m.view(transforms_only = True)

        else:
            for k, m in self.items():
                print()
                print(f"-> For key {k}")
                m.view()
        print("*************************")

    def add(self, key:str, model:Model = None):
        if key not in self:
            self[key] = model
        else:
            print(f'Warning: a model was already set for key {key}')
    
    def estimate(self, dataset:Dataset, store_details:bool = False):                
        
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
                # transforms
                self[k].transforms.estimate(data_)
                data_ = self[k].transforms.transform(data_)                
                if not data: 
                    data = data_
                else:
                    data.stack(data_, allow_both_empty = True)
            if data.empty: raise Exception('data is empty. should not happen')
            self.master_model.estimate(**data.as_dict())            
            # set individual copies         
            # note that the Model may not exists!
            for k, data in dataset.items():
                if k not in self:
                    self[k] = Model()
                self[k].set_model(self.master_model)
                # eliminate this for now
                # self[k].post_estimate(data)
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






