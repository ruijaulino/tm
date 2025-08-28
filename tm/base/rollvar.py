import numpy as np
import matplotlib.pyplot as plt
from tm.base import BaseModel
from tm.base import LinRegr, StateModel
import numpy as np

def rollvar(y, f):
    ysq = y * y
    # Ensure f is normalized to sum to 1
    f = f / f.sum()
    # np.convolve flips filter internally
    v = np.convolve(ysq, f, mode='full')[:len(ysq)]
    return v

def predictive_rollvar(y, f):
    v = rollvar(y, f)
    v = np.hstack((v[0], v[:-1]))
    return v


class RollVar(BaseModel):
    def __init__(self, 
                 base_model:BaseModel,
                 phi = 0.95,  
                 phi_frac_cover = 0.95
                ):
        self.phi = phi
        self.phi_frac_cover = min(phi_frac_cover, 0.9999)
        self.base_model = base_model 
    
    def view(self, plot = False, **kwargs):
        self.base_model.view(plot = plot)

    def estimate(self, y = None, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''
        self.base_model.estimate(y = y, x = x, t = t, z = z, msidx = msidx)

    def posterior_predictive(self, y = None, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target for a RollVar model"
            y = y[:, 0]          
        if y.size != 0:
            m, _ = self.base_model.posterior_predictive(y = y, x = x, t = t, z = z, msidx = msidx)                
            # create filter
            k_f = np.log(1-self.phi_frac_cover)/np.log(self.phi) - 1
            f = (1-self.phi)*np.power(self.phi, np.arange(int(k_f)+1))
            v = predictive_rollvar(y, f)
            # burn some observations
            if is_live and y.size < f.size:
                print('Data is not enough for live. Return zero weight...')
            m[:f.size] = 0
            v[:f.size] = 1
            return m, v
        else:
            return np.zeros_like(y), np.ones_like(y)


class RollVarLinRegr(RollVar):
    def __init__(self, phi = 0.95, phi_frac_cover = 0.95, intercept = True):
        super().__init__(LinRegr(intercept = intercept), phi = phi, phi_frac_cover = phi_frac_cover)

class RollVarStateModel(RollVar):
    def __init__(self, phi = 0.95, phi_frac_cover = 0.95, min_points = 10, zero_states = []):
        super().__init__(StateModel(min_points = min_points, zero_states = zero_states), phi = phi, phi_frac_cover = phi_frac_cover)


if __name__ == '__main__':

    n = 1000
    a=0.1
    b=0.5
    scale = 0.01
    x=np.random.normal(0,scale,n)
    y=a+b*x+np.random.normal(0,scale,n)
    linregr = RollVarLinRegr()
    linregr.estimate(y = y, x = x[:,None])
    linregr.view()
    print(linregr.posterior_predictive(y, x[:,None]))


    print('........')

    y = np.random.normal(0,1,100)
    z = np.random.choice([0,1],100)
    model = RollVarStateModel()
    model.estimate(y = y, z = z)
    m, v = model.posterior_predictive(y = y, z = z)
    print(m)
    print(v)
    model.view()




