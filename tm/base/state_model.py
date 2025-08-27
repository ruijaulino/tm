import numpy as np
import copy
from tm.base import utils
from tm.base import BaseModel

class StateModel(BaseModel):
    def __init__(self, min_points = 10, zero_states = []):
        self.min_points = min_points
        self.states_distribution = {}
        self.zero_states = zero_states
        self.default_mean = 0
        self.default_var = 1
        self.w_norm = 1
        self.p = 1

    def view(self, plot_hist = True):

        print('** State Model **')
        for k, v in self.states_distribution.items():
            print(f"State z = {k}")
            print(v)
            print()

    def estimate(self, y, z, **kwargs):
        
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target (for now)"
            y = y[:, 0]

        assert z.ndim == 1, "z must be a vector"
        assert y.shape[0] == z.size, "y and z must have the same number of observations"

        n = y.size        
        states = np.unique(z)
        for state in states:                        
            if state in self.zero_states:
                self.states_distribution.update({state: {'m':0, 'v':1}})
            else:
                idx = z == state
                if idx.size > self.min_points:
                    m = np.mean(y[idx])
                    v = np.var(y[idx])
                    self.states_distribution.update({state: {'m':m, 'v':v}})

                else:
                    self.states_distribution.update({state: {'m':0, 'v':1}})

    def posterior_predictive(self, z, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        assert z.ndim == 1, "z must be a vector"
        n = z.size
        m = np.zeros(n)
        v = np.ones(n)
        states = np.unique(z)
        for state in states:                        
            idx = z == state
            m[idx] = self.states_distribution.get(state, {'m':0, 'v':1}).get('m')
            v[idx] = self.states_distribution.get(state, {'m':0, 'v':1}).get('v')        
        return m, v


if __name__ == '__main__':
    y = np.random.normal(0,1,100)
    z = np.random.choice([0,1],100)
    model = StateModel()
    model.estimate(y = y, z = z)
    m, v = model.posterior_predictive(z)
    print(m)
    print(v)
    model.view()

