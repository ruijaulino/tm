from abc import ABC, abstractmethod
import numpy as np


class Allocation(ABC):

    w_norm = 1.

    # then this can be overridden
    def view(self, **kwargs):
        pass

    @abstractmethod
    def estimate(self, mu, cov, **kwargs):
        '''
        mu: numpy (n, p) array with expected values
        cov: numpy (n, p, p) array with expected covariances
        '''
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def get_weight(self, mu, cov, **kwargs):
        '''
        mu: numpy (n, p) array with expected values
        cov: numpy (n, p, p) array with expected covariances
        '''
        pass

    def set_use_M(self, use_M = True):
        self.use_M = use_M


def soft(m, v, c, b):
    '''
    Solution to the subproblem
    '''
    if m > b*v + c:
        return (m-c) / v
    elif m < b*v - c:
        return (m+c) / v
    else:
        return b


class Optimal(Allocation):
    def __init__(self, quantile=0.95, diagonal=False, use_M=False, demean=False):
        self.quantile = quantile
        self.diagonal = diagonal
        self.demean = demean
        self.use_M = True
        
        self.w_mean = None
        self.quantiles = None
        self.k = 1

    def set_use_M(self, use_M = True):
        self.use_M = use_M

    def view(self):
        print('k: ', self.k)
        print('Weight mean: ', self.w_mean)

    def estimate(self, mu, cov, **kwargs):                
        # make sure inputs make sense
        # w = self.get_weight(mu, cov, live=False)
        
        # calculate quantiles to clip weights later
        w = self.get_weight(mu, cov, live=False)
        self.quantiles = np.quantile(np.abs(w), self.quantile, axis = 0, method = 'closest_observation')
        # clip weights
        w = np.clip(w, -self.quantiles, self.quantiles)
        if self.demean:
            self.w_mean = np.mean(w, axis = 0)
        else:
            self.w_mean = np.zeros(mu.shape[1])
        self.k = np.quantile(np.sum(np.abs(w), axis = 1), self.quantile, method = 'closest_observation') # using this method also work for state models
        if self.k == 0: self.k = 1
    
    def norm_w_2d(self, w):
        if self.demean:
            w -= self.w_mean
        w /= self.w_norm
        idx = np.sum(np.abs(w), axis = 1) > self.max_w
        w[idx] /= np.sum(np.abs(w[idx]), axis = 1)[:,None] #np.sign(w[idx])*self.max_w
        return w
        
    def get_weight(self, mu, cov, live=False, **kwargs):
        assert mu.ndim == 2, "mu must be a matrix"
        assert cov.ndim == 3, "cov must be a tensor"

        if self.use_M:
            M = cov + np.einsum('ni,nj->nij', mu, mu)
        else:
            M = cov

        if self.diagonal:
            w = mu / np.diagonal(M, axis1=1, axis2=2)
        else:
            w = np.linalg.solve(M, mu[..., None])[..., 0]
            
        if self.quantiles:
            # clip weights
            w = np.clip(w, -self.quantiles, self.quantiles)            

        if self.w_mean:
            w -= self.w_mean

        #w /= self.k

        if not live:
            return w   
        else:
            return w[-1]

if __name__ == '__main__':
    np.random.seed(0)
    mu = np.random.normal(0, 1, (5, 2))
    cov = np.ones((5,1,1))
    cov = np.zeros((5,2,2))
    cov[:,np.arange(cov.shape[1]), np.arange(cov.shape[1])] = 1
    
    opt = Optimal()
    out = opt.get_weight(mu, cov)
    print(out)
    print()

