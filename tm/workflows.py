import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm
import time
from tm.models.abstract import Model
from tm.transforms.abstract import Transform, Transforms
from tm.containers import Data, Dataset
from tm.post_process import Paths

# ModelPipeUnit
# Handles a Model and Transforms applied to a Data object
class ModelPipeUnit:
    def __init__(self, model:Model = None, transforms:Transforms = None):
        self.model = model
        self.transforms = transforms
        if not self.transforms: self.transforms = Transforms()
        self._master_model_set = False

    def set_model(self, model:Model):
        self.model = copy.deepcopy(model)

    def view(self, transforms_only = False):
        if transforms_only:
                self.transforms.view()
        else:
            print("* Model *")
            self.model.view()
            print()
            self.transforms.view()
        print()
        print()

    def estimate(self, data:Data):        
        self.transforms.estimate(data)
        data = self.transforms.transform(data)        
        # model estimation is done with data as dict
        # it should have the required fields
        # the arguments passed are like model.estimate(y, x, z, t, msidx) 
        self.model.estimate(**data.as_dict())
    
    def post_estimate(self, data:Data):
        # adjust parameters after master is estimated!
        
        if hasattr(self.model, "post_estimate"):
            # transforms are already estimated...
            # just need to transform!
            data = self.transforms.transform(data)        
            # model estimation is done with data as dict
            # it should have the required fields
            # the arguments passed are like model.estimate(y, x, z, t, msidx) 
            self.model.post_estimate(**data.as_dict())

    # ....................
    # ....................
    # ....................
    # these methods were a less efficient implementation
    # since this is evaluated many times, this way introduces a large overhead...
    def get_weight(self, d:dict, is_live:bool = False):
        # for a live setting, if features are present, we need an aditional
        # point on y (can be zeros) and this should be done when building the
        # data to send into the workflow. Just here the remainder
        # build q variables
        d_add = {}
        for k, v in d.items():
            if v is not None:
                d_add.update({f"{k}q":v[-1]})
                d[k] = v[:-1]
        d.update(d_add)        
        # if we are in a live setting, must insert here the check
        # to confirm that data was formated correctly        
        if is_live:
            assert (d.get('yq') == 0).all(), "In a live setting, the last observation of y must have been generated artificially with zeros.."        
        # get weight
        return self.model.get_weight(**d)
        
    def evaluate_(self, data:Data):
        """Evaluate the model using the test data and return performance metrics."""
        # this will change fields s, weight_* in data object inplace                
        
        # apply transforms on whole data (it creates a copy if transformations are applied)
        # this prevents too much copies when iterating over the arrays
        transformed_data = self.transforms.transform(data)
        for i in range(transformed_data.n):       
            # this will filter data at index i suitable to make decision
            # it also filter by multisequence index in the .at method
            tmp = transformed_data.input_at(i)
            data.w[i] = self.get_weight(tmp, is_live = False)
        # compute strategy returns (later, in post process this may be calculated again for variable fees...)
        data.s[:] = np.einsum('ij,ij->i', data.w, data.y)
        return data

    def live_(self, data:Data):
        # live is implemented on it's own although it performs
        # similar computations as in evaluate
        data = self.transforms.transform(data)
        return self.get_weight(data.input_at(), is_live = True)        
    # ....................
    # ....................
    # ....................

    def evaluate(self, data:Data):
        """Evaluate the model using the test data and return performance metrics."""
        # this will change fields s, weight_* in data object inplace        
        # evaluation of a model can be an expensive operation and so this tries to
        # be more efficient!        
        
        # apply transforms on whole data (it creates a copy if transformations are applied)
        # this prevents too much copies when iterating over the arrays
        transformed_data = self.transforms.transform(data)
        
        # for sequential models it may be helpfull if they contain a method
        # on how to get the weight on a sequence to re-use calculations        
        # this must be implemented with care and it is not recomended
        if hasattr(self.model, "_evaluate"):

            # get from transformed data
            y = transformed_data.y
            x = transformed_data.x
            t = transformed_data.t
            z = transformed_data.z
            # if the evaluation is being performed in a optimized way, pass the msidx
            w = self.model._evaluate(y = y, x = x, t = t, z = z, msidx = transformed_data.msidx)
            data.w[:] = w
            data.s[:] = np.einsum('ij,ij->i', w, data.y)  

        else:

            # this is a more efficient way to evaluate (1/2 time)
            # create references, instead of accessing the object
            w = transformed_data.w
            y = transformed_data.y
            x = transformed_data.x
            t = transformed_data.t
            z = transformed_data.z
            # lookup to start of data with multisequences
            msidx_start_lookup = transformed_data.msidx_start_lookup
            
            # Pre-define fixed inputs based on available data
            use_x = x is not None
            use_t = t is not None
            use_z = z is not None

            prev_w = None
            for i in range(transformed_data.n):
                start = msidx_start_lookup[i]

                # Prepare slices only if needed
                x_seq, x_q = (x[start:i], x[i]) if use_x else (None, None)
                t_seq = t[start:i] if use_t else None # tq is not passed
                z_seq, z_q = (z[start:i], z[i]) if use_z else (None, None)
                # no need to pass the msidx here as we are already filtering
                w[i] = self.model.get_weight(
                    y = y[start:i],
                    x = x_seq,
                    t = t_seq,
                    z = z_seq,
                    xq = x_q,
                    zq = z_q,
                    prev_w = prev_w,

                )
                prev_w = w[i]
            # set on original data!
            data.w[:] = w
            data.s[:] = np.einsum('ij,ij->i', w, data.y)
        
        return data

    def live(self, data:Data):
        # live is implemented on it's own although it performs
        # similar computations as in evaluate
        # note that data must be provided in a defined way for live evaluation

        # apply transforms
        data = self.transforms.transform(data)
        
        w = data.w
        y = data.y
        x = data.x
        t = data.t
        z = data.z
        # lookup to start of data with multisequences
        msidx_start_lookup = data.msidx_start_lookup

        # Pre-define fixed inputs based on available data
        use_x = x is not None
        use_t = t is not None
        use_z = z is not None
        # check data format
        assert (y[-1] == 0).all(), "In a live setting, the last observation of y must have been generated artificially with zeros.."    
        if use_t:
            assert (t[-1] == 0).all(), "In a live setting, the last observation of t must have been generated artificially with zeros.."    
        
        start = msidx_start_lookup[-1]

        # Prepare slices only if needed
        x_seq, x_q = (x[start:-1], x[-1]) if use_x else (None, None)
        t_seq = t[start:-1] if use_t else None # tq is not passed
        z_seq, z_q = (z[start:-1], z[-1]) if use_z else (None, None)
        # no need to pass the msidx here as we are already filtering
        prev_w = None
        w = self.model.get_weight(
            y = y[start:-1],
            x = x_seq,
            t = t_seq,
            z = z_seq,
            xq = x_q,
            zq = z_q,
            prev_w = prev_w
        )

        return w



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




class ModelPipe():
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
            dataset:Union[Data, Dataset], 
            model_pipe:Union[ModelPipeContainer, ModelPipe],
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
        tmp_model_pipe = copy.deepcopy(model_pipe) 
        # train model
        tmp_model_pipe.estimate(dataset = train_dataset)
        # estimate model - the results will be written in dataset because .between uses simple indexing
        if view_after_estimate:
            tmp_model_pipe.view()
        tmp_model_pipe.evaluate(test_dataset) #
        # set performance on dataset (maybe not needed because it will be already overriten! CHECK THIS)
    
    return dataset    

def cvbt(
        dataset:Union[Dataset,Data], 
        model_pipe:Union[ModelPipeContainer,ModelPipe],
        n_paths: int = 5,
        k_folds:int = 4, 
        seq_path:bool = False, 
        start_fold:int = 0, 
        burn_fraction:float = 0.1, 
        min_burn_points:int = 3,
        view_after_estimate:bool = False
        ) -> Paths:
    paths = Paths()
    for path in tqdm.tqdm(range(n_paths)):
        # need to send in copies..
        path_dataset = cvbt_path(
                            dataset = dataset.copy(), 
                            model_pipe = model_pipe.copy(),
                            k_folds = k_folds, 
                            seq_path = seq_path, 
                            start_fold = start_fold, 
                            burn_fraction = burn_fraction, 
                            min_burn_points = min_burn_points,
                            view_after_estimate = view_after_estimate
                            )
        paths.add(path_dataset)
    return paths





