from tm.base import BaseModel
import numpy as np

class AsUnivariate(BaseModel):
    '''
    Build individual base_model for each target
    Diagonal covariance
    can select the highest prediction only..
    '''
    def __init__(self, 
                 base_model:BaseModel,
                 bet_on_max:bool = False,
                 side:str = 'long'
                ):
        assert side in ['all', 'long', 'short'], "unknown side parameter"
        self.side = side
        self.bet_on_max = bet_on_max
        self.base_model = base_model
        self.base_models = []
        
    def view(self, plot = False, **kwargs):
        for i, m in enumerate(self.base_models):
            print('Model for variable: ', i)
            self.m.view(plot = plot)

    def estimate(self, y, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''
        if y.ndim == 1: y = y[:, None]
        p = y.shape[1]
        for i in range(p):
            tmp = copy.deepcopy(self.base_model)
            tmp.estimate(y = y[:,[i]], x = x, t = t, z = z, msidx = msidx)
            self.base_models.append(tmp)

    def posterior_predictive(self, y, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        '''
        x: numpy (m, p) array
        '''       
        if y.ndim == 1: y = y[:, None]
        m, v = np.zeros_like(y), np.zeros_like(y)
        for i in range(y.shape[1]):
            m[:, i], v[:,i] = self.base_models[i].posterior_predictive(
                                                        y = y[:,[i]], 
                                                        x = x, 
                                                        t = t, 
                                                        z = z, 
                                                        msidx = msidx
                                                        )
        if self.bet_on_max:
            if self.side == 'all':
                d = np.argmax(np.abs(m), axis = 1)
            elif self.side == 'long':
                d = np.argmax(m, axis = 1)
            elif self.side == 'short':
                d = np.argmin(m, axis = 1)

            rows = np.arange(m.shape[0])
            new_m = np.zeros_like(m)            
            new_m[rows, d] = m[rows, d]
            m = new_m
        # build cov
        I = np.eye(v.shape[1]) 
        cov = v[:, :, None] * I  
        return m, cov
