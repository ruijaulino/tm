
# from abc import ABC, abstractmethod
# import matplotlib.pyplot as plt
# import numpy as np


# class Allocation(ABC):

#     w_norm = 1.

#     # then this can be overridden
#     def view(self, **kwargs):
#         pass

#     @abstractmethod
#     def estimate(self, mu, cov, **kwargs):
#         '''
#         mu: numpy (n, p) array with expected values
#         cov: numpy (n, p, p) array with expected covariances
#         '''
#         """Subclasses must implement this method"""
#         pass

#     @abstractmethod
#     def get_weight(self, mu, cov, **kwargs):
#         '''
#         mu: numpy (n, p) array with expected values
#         cov: numpy (n, p, p) array with expected covariances
#         '''
#         pass

#     def set_use_M(self, use_M = True):
#         self.use_M = use_M


# def soft(m, v, c, b):
#     '''
#     Solution to the subproblem
#     '''
#     if m > b*v + c:
#         return (m-c) / v
#     elif m < b*v - c:
#         return (m+c) / v
#     else:
#         return b


# class Optimal(Allocation):
#     def __init__(self, clip_quantile=0.99, k_std = 3, diagonal=False, use_M=False, max_w = 1., demean=False, w_precision = 0.0001):
#         self.clip_quantile = clip_quantile
#         self.k_std = k_std
#         self.diagonal = diagonal
#         self.demean = demean
#         self.w_precision = w_precision
#         self.use_M = use_M
#         self.max_w = max_w        
#         self.w_mean = None
#         self.quantiles = None
#         self.k = 1
#         self.external_multiplier = 1
#         self.leverage_scale = 1
#         self.aux_mult = 1


#     def set_use_M(self, use_M = True):
#         self.use_M = use_M

#     def view(self):
#         print('k: ', self.k)
#         print('Weight mean: ', self.w_mean)
#         print('external multiplier: ', self.external_multiplier)
#         print('leverage scale: ', self.leverage_scale)

#     def estimate(self, mu, cov, **kwargs):                
#         # make sure inputs make sense
#         # calculate quantiles to clip weights later
#         self.k = 1. # make sure it is 1
#         if mu.size > 50:        
#             w = self.get_weight(mu, cov, live=False, in_estimate = True)            
#             self.quantiles = np.quantile(np.abs(w), self.clip_quantile, axis = 0, method = 'closest_observation')
#             # clip weights
#             w = np.clip(w, -self.quantiles, self.quantiles)
#             if self.demean:
#                 self.w_mean = np.mean(w, axis = 0)
#             else:
#                 self.w_mean = np.zeros(mu.shape[1])
#             # calculate leverage scale
#             l = np.abs(w).sum(axis=1)
#             self.leverage_scale = np.std(l)
#             self.k = self.k_std*self.leverage_scale
#             if self.k == 0: self.k = 1
#         else:
#             self.aux_mult = 0
        
#     def get_weight(self, mu, cov, live=False, in_estimate=False, **kwargs):
#         assert mu.ndim == 2, "mu must be a matrix"
#         assert cov.ndim == 3, "cov must be a tensor"

#         if self.use_M:
#             M = cov + np.einsum('ni,nj->nij', mu, mu)
#         else:
#             M = cov

#         if self.diagonal:
#             w = mu / np.diagonal(M, axis1=1, axis2=2)
#         else:
#             w = np.linalg.solve(M, mu[..., None])[..., 0]
            
#         if self.quantiles is not None:
#             # clip weights
#             w = np.clip(w, -self.quantiles, self.quantiles)            

#         if self.w_mean is not None:
#             w -= self.w_mean
#         w /= self.k

#         if not in_estimate:
#             # need to clip again because master k
#             w = np.clip(w, -self.max_w, self.max_w)            

#         w = np.trunc(w / self.w_precision) * self.w_precision

#         if not live:
#             return w   
#         else:
#             return w[-1]

# if __name__ == '__main__':
#     np.random.seed(0)
#     mu = np.random.normal(0, 1, (5, 2))
#     cov = np.ones((5,1,1))
#     cov = np.zeros((5,2,2))
#     cov[:,np.arange(cov.shape[1]), np.arange(cov.shape[1])] = 1
    
#     opt = Optimal()
#     out = opt.get_weight(mu, cov)
#     print(out)
#     print()

#####################################################
#####################################################
#####################################################


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

    def set_use_m2(self, use_m2 = True):
        self.use_m2 = use_m2


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
    def __init__(self, quantile = 0.95, max_w = 1, c = None, seq_w = False, demean = False):
        self.quantile = quantile
        self.max_w = max_w
        self.c = c
        self.seq_w = seq_w
        self.demean = demean
        self.use_m2 = True
        self.w_norm = 1
        self.w_mean = 0
        self.k = 1


    def set_use_m2(self, use_m2 = True):
        self.use_m2 = use_m2

    def view(self):
        print('Weight norm: ', self.w_norm)
        print('Weight mean: ', self.w_mean)

    def estimate(self, mu, cov, **kwargs):                
        # make sure inputs make sense
        assert mu.ndim == 2, "mu must be a matrix"
        assert cov.ndim == 3, "cov must be a tensor"
        p = mu.shape[1]
        if p == 1:
            m = mu.ravel()
            v = cov.ravel()
            if self.use_m2:
                w = m / (v + m*m)
            else:
                w = m / v
            if w.size != 0:
                self.w_norm = np.quantile(np.abs(w), self.quantile, method = 'closest_observation') # using this method also work for state models
                if self.w_norm == 0: self.w_norm = 1
                self.w_mean = np.mean(w)
            else:
                self.w_norm = 1
                self.w_mean = 0

        else:

            if self.use_m2:
                M = cov + np.einsum('ni,nj->nij', mu, mu) 
                M_inv = np.linalg.inv(M)
            else:
                M_inv = np.linalg.inv(cov)
            w = np.einsum('nij,ni->nj', M_inv, mu)

            if w.shape[0] != 0:
                self.w_norm = np.quantile(np.sum(np.abs(w), axis = 1), self.quantile, method = 'closest_observation') # using this method also work for state models
                if self.w_norm == 0: self.w_norm = 1
                self.w_mean = np.mean(w, axis = 0)
            else:
                self.w_norm = 1
                self.w_mean = np.zeros(w.shape[1])
            #raise Exception('Optimal for p>1 not yet implemented')
    
    def norm_w_1d(self, w):
        # demean
        if self.demean:
            w -= self.w_mean
        w /= self.w_norm
        idx = np.abs(w) > self.max_w
        w[idx] = np.sign(w[idx])*self.max_w
        return w

    def norm_w_2d(self, w):
        if self.demean:
            w -= self.w_mean
        w /= self.w_norm
        idx = np.sum(np.abs(w), axis = 1) > self.max_w
        w[idx] /= np.sum(np.abs(w[idx]), axis = 1)[:,None] #np.sign(w[idx])*self.max_w
        return w
        

    def get_weight(self, mu, cov, cost_scale = 1, live = False, prev_w = None, **kwargs):                
        p = mu.shape[1]
        if p == 1:
            m = mu.ravel()
            v = cov.ravel()
            # no costs aware alloc
            if self.c is None:
                if self.use_m2:
                    w = m / (v + m*m)
                else:
                    w = m / v
                w = self.norm_w_1d(w)
                if not live:
                    return np.atleast_2d(w.T).T   
                else:
                    return w[-1]

            # costs aware alloc
            else:
                # create costs
                if isinstance(cost_scale, np.ndarray):
                    assert cost_scale.ndim == 1, "cost_scale must be a vector"
                    assert cost_scale.size == 1, "cost_scale must have a single entry"
                    cost_scale = cost_scale[0]
                c = self.c / cost_scale
                # sequential weights
                if self.seq_w:
                    w = np.zeros_like(m)
                    b = 0
                    for i in range(w.size):
                        w[i] = soft(m[i], v[i], c, b)
                        b = w[i] # use current weight to condition the next step
                    if not live:
                        w = self.norm_w_1d(w)
                        return np.atleast_2d(w.T).T   
                    else:
                        # adjust last entry
                        if prev_w is not None:                            
                            w[-1] = soft(m[i], v[i], c, prev_w)    
                        w = self.norm_w_1d(w)
                        return w[-1]

                else:
                    w = np.zeros_like(m)
                    idx = m>c 
                    w[idx] = (m[idx]-c) / (v[idx]+m[idx]*m[idx])
                    idx = m<-c 
                    w[idx] = (m[idx]+c) / (v[idx]+m[idx]*m[idx])
                    w = self.norm_w_1d(w)
                    if not live:
                        return np.atleast_2d(w.T).T   
                    else:
                        return w[-1]
                        
        else:
                        
            if self.use_m2:
                M = cov + np.einsum('ni,nj->nij', mu, mu) 
                M_inv = np.linalg.inv(M)
            else:
                M_inv = np.linalg.inv(cov)

            w = np.einsum('nij,ni->nj', M_inv, mu)

            w = self.norm_w_2d(w)
            if not live:
                return np.atleast_2d(w.T).T   
            else:
                return w[-1]

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
