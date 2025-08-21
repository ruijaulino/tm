import numpy as np
from tm.base import BaseModel


class LinRegr(BaseModel):
    def __init__(self, intercept = True):
        self.intercept = intercept
        self.w = None
        self.v = 1

    def view(self, **kwargs):
        print('** Linear Regression **')
        print('Weights: ', self.w)
        print('Variance: ', self.v)


    def estimate(self, y, x, **kwargs):
        '''
        y: numpy (n, ) array with targets
        x: numpy (n, p) array with features
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]
        assert x.ndim == 2, "x must be a matrix with the features!"
        assert y.size == x.shape[0], "y and x must have the same number of observations"
        n = y.size
        if self.intercept:
            x = np.hstack((np.ones((n, 1)), x))
        Q, R = np.linalg.qr(x)
        self.w = np.linalg.solve(R, Q.T @ y)
        self.v = np.var(y - x @ self.w)

    def posterior_predictive(self, x, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        assert x.ndim == 2, "x must be a matrix with the features!"
        n = x.shape[0]
        if self.intercept:
            x = np.hstack((np.ones((n, 1)), x))
        m = x @ self.w        
        return m, self.v*np.ones_like(m)


class RollVarLinRegr:
    def __init__(self, 
                 window = 20,                 
                 intercept = True                 
                ):
        self.window = window
        self.regr = LinRegr(intercept = intercept) 
    
    def view(self, **kwargs):
        self.regr.view()
        
    def estimate(self, y, x, msidx = None, **kwargs):   
        self.regr.estimate(y = y, x = x)


    def posterior_predictive(self, y, x, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]          

        m, _ = self.regr.posterior_predictive(x = x)        
        
        v = np.zeros_like(y)
        for i in range(self.window, y.size):
            v[i] = np.var(y[i-self.window:i])
        v[:self.window] = v[self.window] 
        return m, v

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
    pass


