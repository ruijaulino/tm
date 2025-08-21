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
    def __init__(self, quantile = 0.95, max_w = 1, c = None, seq_w = False):
        self.quantile = quantile
        self.max_w = max_w
        self.c = c
        self.seq_w = seq_w

    def view(self):
        print('Weight norm: ', self.w_norm)

    def estimate(self, mu, cov, **kwargs):                
        # make sure inputs make sense
        assert mu.ndim == 2, "mu must be a matrix"
        assert cov.ndim == 3, "cov must be a tensor"

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
            # no costs aware alloc
            if self.c is None:
                w = m / (v + m*m)
                w /= self.w_norm
                idx = np.abs(w) > self.max_w
                w[idx] = np.sign(w[idx])*self.max_w
                if not live:
                    return np.atleast_2d(w.T).T   
                else:
                    return w[-1]

            # costs aware alloc
            else:
                # sequential weights
                if self.seq_w:
                    w = np.zeros_like(m)
                    b = 0
                    for i in range(w.size):
                        w[i] = soft(m[i], v[i], self.c, b)
                        b = w[i] # use current weight to condition the next step
                    if not live:
                        return np.atleast_2d(w.T).T   
                    else:
                        # adjust last entry
                        if prev_w is not None:
                            w[-1] = soft(m[i], v[i], self.c, prev_w)    
                        return w[-1]

                else:
                    w = np.zeros_like(m)
                    idx = m>self.c 
                    w[idx] = (m[idx]-self.c) / (v[idx]+m[idx]*m[idx])
                    idx = m<-self.c 
                    w[idx] = (m[idx]+self.c) / (v[idx]+m[idx]*m[idx])
                    if not live:
                        return np.atleast_2d(w.T).T   
                    else:
                        return w[-1]
                        
        else:
            raise Exception('Optimal for p>1 not yet implemented')



if __name__ == '__main__':
    np.random.seed(0)
    mu = np.random.normal(0, 1, (5, 1))
    cov = np.ones((5,1,1))
    
    opt = Optimal()
    out = opt.get_weight(mu, cov)
    print(out)
    print()

    opt = Optimal(c = 0.5)
    out = opt.get_weight(mu, cov)
    print(out)
    print()

    opt = Optimal(c = 0.5, seq_w = True)
    out = opt.get_weight(mu, cov)
    print(out)
    print()
