import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm
import time
from tm.model import Model
from tm.base_models import BaseModel
from tm.transforms.abstract import Transform, Transforms
from tm.containers import Data, Dataset
from tm.post_process import Paths



# Dict of ModelPipeStack
# Objective here is to handle for several data in a dataset where
# each one has a ModelPipeStack associated

class ModelPipeContainer(dict):    

    def __init__(self, master_model:Model = None):
        self.master_model = master_model

    def view(self):
        if self.master_model:
            print("* Master Model *")
            self.master_model.view()
            for k, unit in self.items():
                print()
                print(f"-> For key {k}")
                unit.view(transforms_only = True)

        else:
            for k, unit in self.items():
                print()
                print(f"-> For key {k}")
                unit.view()

    def copy(self):
        return copy.deepcopy(self)
    
    def add(self, key:str, model:Model = None, transforms:Transforms = None):
        if key not in self:
            self[key] = ModelPipeUnit(model, transforms)
    
    def estimate(self, dataset:Dataset):
        '''
        porfolio model is used to filter out data used to estimate a master model!
        '''
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
            # note that the ModePipeUnit may not exists!
            for k, data in dataset.items():
                if k not in self:
                    self[k] = ModelPipeUnit()
                self[k].set_model(self.master_model)
                self[k].post_estimate(data)
        else:
            for k, data in dataset.items():
                assert k in self, "dataset contains a key that is not defined in ModelPipe. Exit.."
                self[k].estimate(data)
        
    def evaluate(self, dataset:Dataset):
        for k, data in dataset.items():
            assert k in self, "dataset contains a key that is not defined in ModelPipe. Exit.."                        
            self[k].evaluate(data)   
        return dataset    

    def live(self, dataset:Dataset):
        out = {}
        for k, data in dataset.items():
            assert k in self, "dataset contains a key that is not defined in ModelPipe. Exit.."                        
            out.update({k: {'w':self[k].live(data), 'w_cols':data.w_cols}})
        return out    

class PortfolioModel(ABC):

    @abstractmethod
    def estimate(self, dataset:Dataset, model_pipe_container:ModelPipeContainer):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()
        pass

    @abstractmethod
    def get(self, key:str) -> float:
        """Subclasses must implement this method"""
        # returns the portfolio weight for a key
        pass

    def view(self):
        print("PortfolioModel")
        for k, v in self.pws.items():
            print(f'Portfolio Weight for {k} = {v}')


class ModelSet():
    def __init__(self, master_model:Model = None, portfolio_model:PortfolioModel = None):
        self.portfolio_model = portfolio_model
        self.model_pipe_container = ModelPipeContainer(master_model = master_model)        
        # after a model is run this variable stores the dataset 
        # that was used to estimate the model!    
        self.estimation_dataset = None

    def view(self):
        print()
        print("******* ModelPipe *******")
        print()
        if self.portfolio_model:
            self.portfolio_model.view()
        self.model_pipe_container.view()
        print()
        print("*************************")

    def copy(self):
        return copy.deepcopy(self)

    def add(self, key, model:Model = None, transforms:Transforms = None):
        self.model_pipe_container.add(key, copy.deepcopy(model), copy.deepcopy(transforms))
    
    def estimate(self, dataset:Dataset, store_details:bool = False):        
        
        # dataset_dict is a dict of dataset        
        if self.portfolio_model:
            self.portfolio_model.estimate(dataset, self.model_pipe_container.copy())
        self.model_pipe_container.estimate(dataset)        
        # when we train a final model we can store the dataset that was used to estimate the
        # model. If future checks are needed we can just run inference again with it!
        if store_details:
            self.estimation_dataset = copy.deepcopy(dataset)

    def evaluate(self, dataset:Dataset):
        # dataset_dict is a dict of dataset
        self.model_pipe_container.evaluate(dataset)
        # set portfolio weight on dataset                
        if self.portfolio_model:
            for k, data in dataset.items():
                data.pw[:] *= self.portfolio_model.get(k)        
        return dataset

    def live(self, dataset:Dataset):
        # to be used in a live setting
        out = self.model_pipe_container.live(dataset)
        # set portfolio weight on dataset                
        for k, _ in dataset.items():
            tmp = 1
            if self.portfolio_model:
                tmp = self.portfolio_model.get(k)
            out[k].update({'pw':tmp})

        return out

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def load_model_pipe(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


# changes dataset in place
def cvbt_path(
            dataset:Union[Dataset, Data], 
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
        dataset:Union[Dataset], 
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


if __name__ == '__main__':

    def linear(n=1000,a=0,b=0.1, scale = 0.01,start_date='2000-01-01'):
        x=np.random.normal(0,scale,n)
        y=a+b*x+np.random.normal(0,scale,n)
        dates=pd.date_range(start_date,periods=n,freq='D')
        data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
        return data

    df = linear(n=1000, a=0, b=0.5, start_date='2000-01-01')

    data = Data.from_df(df)

    import tm
    base_model = tm.base_models.BayesLinRegr()

    model = Model(base_model = base_model)

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
    
    paths.post_process()

