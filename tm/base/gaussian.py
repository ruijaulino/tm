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



class ConditionalGaussian(BaseModel):
    def __init__(self):
        pass

    def view(self, **kwargs):
        print('ConditionalGaussian')
        print('Cyy')
        print(self.Cyy)
        print('Cxx')
        print(self.Cxx)
        print('Cyx')
        print(self.Cyx)
        print('Gain')
        print(self.pred_gain)
        

    def estimate(self, y, x, **kwargs):
        y = np.atleast_2d(y.T).T
        x = np.atleast_2d(x.T).T
        assert y.shape[0] == x.shape[0], "x and y must have the same number of observations"
        p=y.shape[1]
        q=x.shape[1]
        y_idx=np.arange(p)
        x_idx=np.arange(p,p+q)      
        cov = np.cov(y.T, x.T)
        self.my = np.mean(y, axis = 0)
        self.mx = np.mean(x, axis = 0)
        self.Cyy = cov[y_idx][:,y_idx]
        self.Cxx = cov[x_idx][:,x_idx]
        self.Cyx = cov[y_idx][:,x_idx]
        self.invCxx = np.linalg.inv(self.Cxx)
        self.pred_gain = np.dot(self.Cyx, self.invCxx)
        self.cov_reduct = np.dot(self.pred_gain, self.Cyx.T)
        self.pred_cov = self.Cyy-self.cov_reduct
        self.pred_cov_inv = np.linalg.inv(self.pred_cov)        

    def posterior_predictive(self, y, x, **kwargs):
        xc = x-self.mx
        ev = self.my + xc@self.pred_gain.T
        n = x.shape[0]
        c = np.tile(self.pred_cov, (n, 1, 1))
        return ev, c

if __name__ == '__main__':
    y = np.random.normal(0,1,(10,2))
    x = np.random.normal(0,1,(10,3))
    model = ConditionalGaussian()
    model.estimate(y, x)
    model.view()
    ev, c = model.posterior_predictive(y, x)
    cinv = np.linalg.inv(c)
    print(np.einsum('ni,nj->nij', ev, ev))
    print(ev[0][:,None]*ev[0][None, :])
    #print(np.einsum('nij,ni->nj', cinv, ev))
    #print(cinv[0]@ev[0])
    pass