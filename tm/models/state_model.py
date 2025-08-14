import numpy as np
import copy
from tm.models.abstract import Model

class StateModel(Model):
    def __init__(self, base_model, min_points = 10, zero_states = []):
        self.min_points = min_points
        self.base_model = base_model
        self.states_distribution = {}
        self.zero_states = zero_states
        self.default_mean = 0
        self.default_var = 1
        self.w_norm = 1
        self.p = 1

    def view(self, plot_hist = True):
        print('StateGaussian')
        print('w norm: ', self.w_norm)
        for k, v in self.states_distribution.items():
            print(f"State z={k}")
            print(v)
            print()

    def estimate(self, y, z, **kwargs):
        
        assert isinstance(z, np.ndarray), "z must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"        
        assert z.ndim == 1, "z must be a vector"
        assert y.ndim == 2, "y must be a matrix"
        assert y.shape[0] == z.size, "y and z must have the same number of observations"

        n, self.p = y.shape        
        states = np.unique(z)
        max_w_norm = 0
        for state in states:                        
            if state in self.zero_states:
                self.states_distribution.update({state: {'w':np.zeros(self.p)}})
            else:
                idx = np.where(z == state)[0]
                if idx.size > self.min_points:
                    m = np.mean(y[idx], axis = 0)
                    c = np.atleast_2d(np.cov(y[idx].T))
                    m2 = c + np.outer(m, m)
                    w = np.dot(np.linalg.inv(m2), m)
                    self.states_distribution.update({state: {'m':m, 'c':c, 'w':w}})
                    w_norm = np.sum(np.abs(w))
                    max_w_norm = max(max_w_norm, w_norm)
                else:
                    self.states_distribution.update({state: {'w':np.zeros(self.p)}})
        if max_w_norm == 0: max_w_norm = 1
        self.w_norm = max_w_norm 
        for k, v in self.states_distribution.items():
            v['w'] /= self.w_norm 

    def get_weight(self, zq , **kwargs):
        return self.states_distribution.get(zq, {}).get('w', np.zeros(self.p))

