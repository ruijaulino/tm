import numpy as np
import matplotlib.pyplot as plt
from tm.base import BaseModel

class uGaussian(BaseModel):
    def __init__(self):
        self.m = None
        self.v = None

    def view(self, plot = False, **kwargs):
        print('** Gaussian **')
        print('mean: ', self.m)
        print('variance: ', self.v)

    def estimate(self, y, **kwargs):
        '''
        y: numpy (n, ) array with targets
        x: numpy (n, p) array with features
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]
        
        n = y.size
        self.m = np.mean(y)
        self.v = np.var(y)

    def posterior_predictive(self, y, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        n = y.shape[0]
        return self.m * np.ones((n, 1)), self.v*np.ones((n, 1, 1))

