from abc import ABC, abstractmethod
import numpy as np
import copy
from tm.containers import Data
from tm.constants import *

# Transform class
class Transform(ABC):
    
    @abstractmethod
    def view(self):
        pass

    @abstractmethod
    def estimate(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass
    
    @abstractmethod
    def inverse_transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass


# Transforms class: a dict of transforms
class Transforms():

    def __init__(self, y_transform:Transform = None, x_transform:Transform = None, t_transform:Transform = None):
        self.y_transform = y_transform
        self.x_transform = x_transform
        self.t_transform = t_transform
        self.has_transforms = False
        if self.y_transform or self.x_transform or self.t_transform: self.has_transforms = True

    def view(self):
        if self.y_transform:
            print('y transform')
            self.y_transform.view()

        if self.x_transform:
            print('x transform')
            self.x_transform.view()

        if self.t_transform:
            print('t transform')
            self.t_transform.view()

    def estimate(self, data:Data):
        # maybe refactor a bit Data class to make this more clear!
        if self.y_transform:
            self.y_transform.estimate(data.y) 
        if self.x_transform:
            if data.x is None: raise Exception('x_transform is defined for data without x...')
            self.x_transform.estimate(data.x) 
        if self.t_transform:
            if data.t is None: raise Exception('t_transform is defined for data without t...')
            self.t_transform.estimate(data.t) 
    
    def transform(self, data:Data):
        # maybe refactor a bit Data class to make this more clear!
        if self.has_transforms:
            data = data.copy()
        if self.y_transform:
            data.y[:] = self.y_transform.transform(data.y)
        if self.x_transform:
            data.x[:] = self.x_transform.transform(data.x)
        if self.t_transform:
            data.t[:] = self.t_transform.transform(data.t)
        return data

