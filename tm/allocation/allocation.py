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



class Optimal(Allocation):
    def __init__(self, quantile = 0.95, max_w = 1, c = None, seq_w = False):
        self.quantile = quantile
        self.max_w = max_w
        self.c = c
        self.seq_w = seq_w

    def view(self):
        print('Weight norm: ', self.w_norm)

    def estimate(self, mu, cov, **kwargs):                
        # make sure inputs make sense
        p = mu.shape[1]
        if p == 1:
            m = mu.ravel()
            v = cov.ravel()
            w = m / (v + m*m)
            self.w_norm = np.quantile(np.abs(w), self.quantile)
        else:
            raise Exception('Optimal for p>1 not yet implemented')
    
    def get_weight(self, mu, cov, live = False, prev_w = None, **kwargs):                
        p = mu.shape[1]
        if p == 1:
            m = mu.ravel()
            v = cov.ravel()
            if self.c is None:
                w = m / (v + m*m)
                w /= self.w_norm
                idx = np.abs(w) > self.max_w
                w[idx] = np.sign(w[idx])*self.max_w
                if not live:
                    return np.atleast_2d(w.T).T   
                else:
                    return w[-1]
            else:
                raise Exception('Optimal with costs not yet implemented')    
        else:
            raise Exception('Optimal for p>1 not yet implemented')
