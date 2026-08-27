import numpy as np
import matplotlib.pyplot as plt
from tm.base import BaseModel
import numpy as np
from scipy.signal import convolve

def apply_filter_vector(v, f):
    # Ensure f is normalized to sum to 1
    f = f / f.sum()
    # np.convolve flips filter internally
    return np.convolve(v, f, mode='full')[:v.size]

def rollmean(y, f):
    return apply_filter_vector(y, f)

def rollvar(y, f):
    return apply_filter_vector(y*y, f)

def predictive_rollmean(y, f, lag = 0):
    m = rollmean(y, f)    
    m = np.hstack((np.zeros(1+lag), m[:-1-lag]))
    return m

def predictive_rollvar(y, f, lag = 0):
    v = rollvar(y, f)
    v = np.hstack((v[0]*np.ones(1+lag), v[:-1-lag]))
    return v    

# based on blog post
class uTrend():
    def __init__(self, 
                 phi = 0.95, # variance parameter
                 psi = 0.97, # mean parameters
                 roll_var_model = True, # estimate the mean with a changing var
                 frac_cover = 0.95,
                 lag = 0
                ):
        
        self.phi = phi
        self.psi = psi
        self.roll_var_model = roll_var_model
        self.frac_cover = min(frac_cover, 0.9999)
        self.lag = lag
    
    def view(self, plot = False, **kwargs):
        pass

    def estimate(self, y = None, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''
        pass

    def posterior_predictive(self, y = None, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target for uTrend model"
            y = y[:, 0]          
        if y.size != 0:
            
            k_f_var = np.log(1-self.frac_cover)/np.log(self.phi) - 1
            f_var = (1-self.phi)*np.power(self.phi, np.arange(int(k_f_var)+1))
            v = predictive_rollvar(y, f_var, lag = self.lag)
            if v.size<=f_var.size:
                v[:] = 1
            else:
                v[:f_var.size] = v[f_var.size]
            # estimate roll variance for normalizaton
            k_f_mean = np.log(1-self.frac_cover)/np.log(self.psi) - 1
            f_mean = (1-self.psi)*np.power(self.psi, np.arange(int(k_f_mean)+1))
            if self.roll_var_model:
                v_norm = rollvar(y, f_var)            
                v_norm[v_norm == 0] = 1e8
                if v_norm.size <= f_var.size:
                    v_norm[:] = 1
                else:
                    v_norm[:f_var.size] = v_norm[f_var.size] 
                y_n = y/v_norm
                m = predictive_rollmean(y_n, f_mean, lag = self.lag)
                m /= (predictive_rollmean(1/v_norm, f_mean, lag = 0)+1e-8)
            else:
                m = predictive_rollmean(y, f_mean, lag = self.lag)
                
            
            m[:f_mean.size] = 0
            

            return m, v
        else:
            return np.zeros_like(y), np.ones_like(y)