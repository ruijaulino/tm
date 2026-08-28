import numpy as np
import copy
from tm.base import BaseModel




class StateModel(BaseModel):
    def __init__(self, min_points = 10, zero_states = [], use_var = False):
        self.min_points = min_points
        self.states_distribution = {}
        self.zero_states = zero_states
        self.use_var = use_var
        self.default_mean = 0
        self.default_var = 1
        self.w_norm = 1
        self.p = 1

    def view(self, plot = False, **kwargs):

        print('** State Model **')
        for k, v in self.states_distribution.items():
            print(f"State z = {k}")
            print(v)
            print()

    def estimate(self, y, z, var = None, **kwargs):
        
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target (for now)"
            y = y[:, 0]

        if z.ndim == 2:
            assert z.shape[1] == 1, "z must contain a single state (for now)"
            z = z[:, 0]

        # assert z.ndim == 1, "z must be a vector"

        assert y.shape[0] == z.size, "y and z must have the same number of observations"



        n = y.size        
        if var is None:
            var = np.ones(n)
        if not self.use_var:
            var = np.ones(n)
        var[var<1e-6] = 1e-6
        states = np.unique(z)
        for state in states:                        
            if state in self.zero_states:
                self.states_distribution.update({state: {'m':0, 'v':1}})
            else:
                idx = z == state
                if idx.size > self.min_points:
                    m = np.sum(y[idx]/var[idx]) / max(np.sum(1/var[idx]), 1e-8)
                    v = np.var(y[idx])
                    self.states_distribution.update({state: {'m':m, 'v':v}})

                else:
                    self.states_distribution.update({state: {'m':0, 'v':1}})

    def posterior_predictive(self, z, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if z.ndim == 2:
            assert z.shape[1] == 1, "z must contain a single state (for now)"
            z = z[:, 0]
        n = z.size
        m = np.zeros(n)
        v = np.ones(n)
        states = np.unique(z)
        for state in states:                        
            idx = z == state
            m[idx] = self.states_distribution.get(state, {'m':0, 'v':1}).get('m')
            v[idx] = self.states_distribution.get(state, {'m':0, 'v':1}).get('v')        
        return m, v



class EnsembleStateModel(BaseModel):
    '''
    Build a individual base_model for each feature
    '''
    def __init__(self, model_weights = None, min_points = 10, use_var = False, base_models = None):
        if base_models is None:
            self.base_model = StateModel(min_points = min_points, use_var = use_var)
            self.predefined_base_models = None
        else:
            assert isinstance(base_models, list), "base_models must be a list"
            for e in base_models:
                assert isinstance(e, StateModel), "base_models must be instances of StateModel"
            self.predefined_base_models = copy.deepcopy(base_models)
        self.base_models = None
        self.model_weights = model_weights



    def view(self, plot = False, **kwargs):
        for i, m in enumerate(self.base_models):
            print('Model for state: ', i)
            m.view(plot = plot)        

    def estimate(self, y, z, var = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''

        assert z.ndim == 2, "z must be a matrix!"
        p = z.shape[1]
        


        if self.predefined_base_models is not None:
            assert len(self.predefined_base_models) == p, "provided base_models do not match the states"
            create = False
        else:
            create = True
        
        self.base_models = []

        for i in range(p):
            if create:
                tmp = copy.deepcopy(self.base_model)
            else:
                tmp = self.predefined_base_models[i]
            tmp.estimate(y = y, z = z[:,[i]], var = var)
            self.base_models.append(tmp)

        if self.model_weights is None: self.model_weights = np.ones(p)
        assert len(self.model_weights) == p, "model weights do not have the same dimension as number of features!"
        self.model_weights = np.array(self.model_weights)
        self.model_weights /= np.sum(np.abs(self.model_weights))

    def posterior_predictive(self, z, **kwargs):
        assert z.ndim == 2, "z must be a matrix!"
        m, cov = np.zeros(z.shape[0], dtype = np.float64), np.zeros(z.shape[0], dtype = np.float64)  
        # note:
        # this only works for a single target
        # also, StateModel only supports single target for now
        # so this should work fine

        for i in range(z.shape[1]):
            m_, cov_ = self.base_models[i].posterior_predictive(
                                                        z = z[:,i], 
                                                        )
            m += self.model_weights[i]*m_

            cov += self.model_weights[i]*cov_
        return m, cov

class toStateModel(BaseModel):
    def __init__(self, min_points = 10, th = 0):
        self.th = th
        self.state_model = StateModel(min_points = min_points)

    def view(self, plot = False, **kwargs):
        self.state_model.view(plot = plot)

    def estimate(self, y, x, **kwargs):
        if x.ndim == 2: x = x[:,0]     
        # create z
        z = np.zeros(x.size)
        z[x>self.th] = 1        
        self.state_model.estimate(y, z)

    def posterior_predictive(self, x, **kwargs):
        if x.ndim == 2: x = x[:,0]
        # create z
        z = np.zeros(x.size)
        z[x>self.th] = 1        
        return self.state_model.posterior_predictive(z)





if __name__ == '__main__':
    y = np.random.normal(0,1,100)
    z = np.random.choice([0,1],(100,2))
    model = EnsembleStateModel()
    model.estimate(y = y, z = z)
    model.view()
    m, v = model.posterior_predictive(z)
    print(m)
    print(v)
    model.view()

