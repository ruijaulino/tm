from tm.base import BaseModel
import copy
import numpy as np



class AsSingle(BaseModel):
    '''
    Build a individual base_model for each feature
    '''
    def __init__(self, base_model:BaseModel, model_weights = None):
        self.base_model = base_model
        self.base_models = []
        self.model_weights = model_weights

    def view(self, plot = False, **kwargs):
        for i, m in enumerate(self.base_models):
            print('Model for feature: ', i)
            m.view(plot = plot)        

    def estimate(self, y, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''
        assert x.ndim is not None, "x must be defined"
        assert x.ndim == 2, "x must be a matrix!"
        p = x.shape[1]
        for i in range(p):
            tmp = copy.deepcopy(self.base_model)
            tmp.estimate(y = y, x = x[:,[i]], t = t, z = z, msidx = msidx)
            self.base_models.append(tmp)
        if self.model_weights is None: self.model_weights = np.ones(p)
        assert len(self.model_weights) == p, "model weights do not have the same dimension as number of features!"
        self.model_weights = np.array(self.model_weights)
        self.model_weights /= np.sum(np.abs(self.model_weights))

    def posterior_predictive(self, y, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        m, cov = np.zeros_like(y, dtype = np.float64), np.zeros_like(y, dtype = np.float64)  
        for i in range(x.shape[1]):
            m_, cov_ = self.base_models[i].posterior_predictive(
                                                        y = y, 
                                                        x = x[:,i], 
                                                        t = t, 
                                                        z = z, 
                                                        msidx = msidx
                                                        )
            if y.ndim == 2 and m_.ndim == 1:
                m += self.model_weights[i]*m_[:,None]
            else:
                m += self.model_weights[i]*m_

            if y.ndim == 2 and cov_.ndim == 1:
                cov += self.model_weights[i]*cov_[:,None]
            else:
                cov += self.model_weights[i]*cov_
        return m, cov


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
            m.view(plot = plot)

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
        m, v = np.zeros_like(y, dtype = np.float64), np.zeros_like(y, dtype = np.float64)
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
            if self.side == 'long':
                m[m<0] = 0
            if self.side == 'short':
                m[m>0] = 0
                    
        # build cov
        I = np.eye(v.shape[1]) 
        cov = v[:, :, None] * I  
        return m, cov
