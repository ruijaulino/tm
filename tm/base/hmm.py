import numpy as np
from scipy.stats import invgauss, invgamma, norm, gamma
import matplotlib.pyplot as plt
from tm.base import BaseModel
from numba import jit
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import tm

@jit(nopython=True)
def hmm_forward(prob, A, P, alpha, c):
    '''
    Forward algorithm for the HMM
    prob: numpy (n,n_states) array with
        the probability of each observation
        for each state
    A: numpy (n_states,n_states) array with the state
        transition matrix
    P: numpy (n_states,) array with the initial
        state probability
    returns:
        alpha: numpy (n,n_states) array meaning
            p(state=i|obs <= i)
        c: numpy (n,) array with the normalization
            constants
    '''
    n = prob.shape[0]
    alpha[0] = P*prob[0]
    c[0] = 1 / np.sum(alpha[0])
    alpha[0] *= c[0]
    for i in range(1, n):
        alpha[i] = np.dot(A.T, alpha[i-1]) * prob[i] 
        c[i] = 1 / np.sum(alpha[i])
        alpha[i] *= c[i]

# this can be compiled
@jit(nopython=True)
def hmm_backward_sample(A, alpha, z, transition_counter, init_state_counter):
    '''
    Backward sample from the state transition matrix and state sequence
    A: numpy (n_states,n_states) array with the state
        transition matrix
    alpha: numpy (n,n_states) array meaning
        p(state=i|obs <= i)     
    z: numpy (n,) to store the sample of state sequence
    transition_counter: numpy (n_states,n_states) array to store 
        transition counts to be used to sample a state transition 
        matrix
    init_state_counter: numpy (n_states,) array to store the
        number of times state i is the initial one
    returns:
        none (q and transition_counter are changed inside this function)
    ''' 
    # backward walk to sample from the state sequence
    n = z.size
    # sample the last hidden state with probability alpha[-1]
    z[n-1] = np.searchsorted(np.cumsum(alpha[-1]), np.random.random(), side = "right")
    # aux variable
    p = np.zeros(A.shape[0], dtype = np.float64)
    # iterate backwards
    for j in range(n-2, -1, -1):
        # from formula
        p = A[:,z[j+1]] * alpha[j] 
        # normalize (from formula)
        p /= np.sum(p) 
        # sample hidden state with probability p
        z[j] = np.searchsorted(np.cumsum(p), np.random.random(), side="right")
        # increment transition counter (we can do this calculation incrementally)
        transition_counter[z[j],z[j+1]] += 1 
    # increment initial state counter
    init_state_counter[z[0]] += 1



# auxiliar function for univariate Laplace


def sample_taus(x, mu, b, eps=1e-12):
    d = np.abs(x - mu)
    taus = np.empty_like(d)
    # Case A: |x - mu| > 0  → sample u = 1/τ from IG, then invert
    mask = d > eps
    if np.any(mask):
        mu_s = (b / d[mask])              # SciPy's 'mu' (shape)
        scale = 1.0 / (b**2)              # SciPy's 'scale' = λ
        u = invgauss.rvs(mu=mu_s, scale=scale, size=mask.sum())
        u[u<1e10] = 1e10
        taus[mask] = 1.0 / u
    # Case B: |x - mu| = 0  → τ | d=0 ∝ τ^{-1/2} exp(-τ/(2 b^2))  = Gamma(1/2, rate=1/(2 b^2))
    zero_mask = ~mask
    if np.any(zero_mask):
        shape = 0.5
        rate = 1.0 / (2.0 * b**2)
        scale_gamma = 1.0 / rate          # Gamma in SciPy is shape, scale
        taus[zero_mask] = gamma.rvs(shape, scale=scale_gamma, size=zero_mask.sum())
    return taus


# Generic emissions class!
class HMMEmissions(ABC):

    def set_gibbs_parameters(self, n_gibbs, f_burn, n_gibbs_sim = None):
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        aux = int(self.n_gibbs*(1+self.f_burn))
        self.n_gibbs_sim = aux if n_gibbs_sim is None else n_gibbs_sim

    def view(self, plot_hist:bool = False, **kwargs):
        """Subclasses must implement this method"""
        pass
    
    def get_n_states(self):
        """Subclasses must implement this method"""
        return self.n_states
    
    @abstractmethod
    def gibbs_initialize(self, y:np.ndarray, x:np.ndarray, t:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_posterior_sample(self, z:np.ndarray, y:np.ndarray, x:np.ndarray, t:np.ndarray, iteration:int, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_burn_and_mean(self):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_prob(self, prob:np.ndarray, y:np.ndarray, x:np.ndarray, t:np.ndarray, iteration:int, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def prob(self, y:np.ndarray, x:np.ndarray, t:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass    

    @abstractmethod
    def posterior_predictive(self, next_state_prob:np.ndarray, xq:np.ndarray, tq:np.ndarray, **kwargs):
    # def get_weight(self, next_state_prob:np.ndarray, xq:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass        
        

# Generic base univariate emission class!
class uHMMBaseEmission(ABC):

    def set_gibbs_parameters(self, n_gibbs, f_burn, n_gibbs_sim = None):
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        aux = int(self.n_gibbs*(1+self.f_burn))
        self.n_gibbs_sim = aux if n_gibbs_sim is None else n_gibbs_sim

    def view(self, plot_hist:bool = False, plot:bool = False, **kwargs):
        """Subclasses must implement this method"""
        pass
        
    @abstractmethod
    def gibbs_initialize(self, y:np.ndarray, x:np.ndarray, t:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_posterior_sample(self, y:np.ndarray, x:np.ndarray, t:np.ndarray, iteration:int, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_burn_and_mean(self):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_prob(self, y:np.ndarray, x:np.ndarray, t:np.ndarray, iteration:int, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def prob(self, y:np.ndarray, x:np.ndarray, t:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass    

    @abstractmethod
    def posterior_moments(self, **kwargs):
    # def get_weight(self, next_state_prob:np.ndarray, xq:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass        
        
# Univariate emission models...

class uBaseLaplaceEmission(uHMMBaseEmission):
    def __init__(self, n_gibbs:int = 1000, f_burn:float = 0.1, min_points_update = 5):
        self.LOWER = 1e-16
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.min_points_update = min_points_update

    def view(self, plot_hist = False, plot = False, **kwargs):
        print()
        print('uBaseLaplaceEmission')
        print('mean: ', self.mean)
        print('b: ', self.b)
        print()
        pass
    
    def gibbs_initialize(self, y, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseLaplaceEmission only work for univariate observations"
            y = y[:,0]
        # Scale samples
        self.gibbs_b = np.zeros(self.n_gibbs_sim) 
        # Mean samples
        self.gibbs_mean = np.zeros(self.n_gibbs_sim)          
        
        # compute data variance
        y_var = np.var(y)
        
        self.gibbs_b[0] = np.sqrt(y_var) 
        self.gibbs_mean[0] = np.mean(y)
        
        self.m0 = 0
        self.s0 = 1000*y_var
        self.alpha0 = 2
        self.beta0 = 0.01*y_var   
        
        self.prev_mn = self.m0
        self.prev_sn = self.s0
        self.prev_alphan = self.alpha0
        self.prev_betan = self.beta0
        
    def gibbs_posterior_sample(self, y:np.ndarray, iteration:int, **kwargs):
        '''
        y: current set of observations
        to be called while in sampler
        '''
        # update each one
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseLaplaceEmission only work for univariate observations"
            y = y[:,0]        
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"
        # no observations in y
        # for each state do the update
        
        if y.size < self.min_points_update:
            b2 = invgamma.rvs(self.prev_alphan, scale=self.prev_betan)
            self.gibbs_b[iteration] = np.sqrt(b2)              
            self.gibbs_mean[iteration] = norm.rvs(self.prev_mn, self.prev_sn)
        else:                
            n = y.size
            y_median = np.median(y)

            tau = sample_taus(y, self.gibbs_mean[iteration-1], self.gibbs_b[iteration-1])

            # 2) Update mu
            prec = 1/self.s0**2 + np.sum(1/tau)
            mean_mu = (self.m0/self.s0**2 + np.sum(y/tau)) / prec
            self.gibbs_mean[iteration] = norm.rvs(mean_mu, np.sqrt(1/prec))

            # 3) Update b^2
            alpha = self.alpha0 + n
            beta = self.beta0 + 0.5*np.sum(tau)
            b2 = invgamma.rvs(alpha, scale=beta)
            self.gibbs_b[iteration] = np.sqrt(b2)   

            self.prev_mn = mean_mu
            self.prev_sn = np.sqrt(1/prec)
            self.prev_alphan = alpha
            self.prev_betan = beta
                
    def gibbs_burn_and_mean(self):
        self.gibbs_mean = self.gibbs_mean[-self.n_gibbs:]
        self.gibbs_b = self.gibbs_b[-self.n_gibbs:] 
        self.mean = np.mean(self.gibbs_mean)
        self.b = np.mean(self.gibbs_b)
        
    def gibbs_prob(self, y:np.ndarray, iteration:int, **kwargs):
        '''
        changes prob array in place
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseLaplaceEmission only work for univariate observations"
            y = y[:,0]    
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"        
        out = np.exp(-np.abs(y-self.gibbs_mean[iteration-1])/self.gibbs_b[iteration-1])
        out /= (2*self.gibbs_b[iteration-1])
        out[out < self.LOWER] = self.LOWER
        return out
    
    def prob(self, y:np.ndarray, **kwargs):
        '''
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseLaplaceEmission only work for univariate observations"
            y = y[:,0]    
        out = np.exp(-np.abs(y-self.mean)/self.b)
        out /= (2*self.b)
        return out

    def posterior_moments(self, **kwargs):
        return self.mean, 2*self.b*self.b
    


class uBaseGaussianEmission(uHMMBaseEmission):
    def __init__(self, 
                 n_gibbs:int = 1000, 
                 f_burn:float = 0.1, 
                 min_points_update = 5, 
                 normalize = True):
        self.LOWER = 1e-16
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.min_points_update = min_points_update

    def view(self, plot_hist = False, plot = False, **kwargs):
        print()
        print('uBaseGaussianEmission')
        print('mean: ', self.mean)
        print('scale: ', np.sqrt(self.var))
        print()
        pass
            
    def gibbs_initialize(self, y, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseGaussianEmission only work for univariate observations"
            y = y[:,0]
        # this will be updated later
        self.mean = 0         
        # Covariance samples
        self.gibbs_var = np.zeros(self.n_gibbs_sim) 
        # Mean samples
        self.gibbs_mean = np.zeros(self.n_gibbs_sim)          
        # compute data variance
        self.y_var = np.var(y)
        self.y_scale = np.std(y)
        # Prior distribution parameters
        self.m0 = 0 
        self.v0 = 1000*self.y_var # mean: prior covariance        
        self.a0 = 2 # infinite variance...
        self.b0 = 0.01*self.y_var
        # initialize
        self.gibbs_mean[0] = self.m0
        self.gibbs_var[0] = self.y_var
        # store parameters
        self.prev_mn = self.m0
        self.prev_vn = self.v0
        
        self.prev_an = self.a0
        self.prev_bn = self.b0

    def gibbs_posterior_sample(self, y:np.ndarray, iteration:int, **kwargs):
        '''
        y: current set of observations
        to be called while in sampler
        '''
        # update each one
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseGaussianEmission only work for univariate observations"
            y = y[:,0]        
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"
        # no observations in y
        # for each state do the update
        
        if y.size < self.min_points_update:
            self.gibbs_mean[iteration] = np.random.normal(self.prev_mn, self.prev_vn)
            self.gibbs_var[iteration] = 1 / np.random.gamma(self.prev_an, 1 / self.prev_bn)
        else:                
            n = y.size
            y_sum = np.sum(y)            
            # Sample from mean
            mn = (self.m0*self.gibbs_var[iteration-1] + self.v0*y_sum)
            mn /= (n*self.v0 + self.gibbs_var[iteration-1])            
            vn = self.gibbs_var[iteration-1] * self.v0 / (n*self.v0 + self.gibbs_var[iteration-1])

            self.prev_mn = mn
            self.prev_vn = vn
            self.gibbs_mean[iteration] = np.random.normal(mn, vn)
            # Sample from variance
            an = self.a0 + n/2
            # bn = self.b0[s] + 0.5*np.sum(np.power(y[idx_states]-y_sum/n,2))
            bn = self.b0 + 0.5*np.sum(np.power(y-self.gibbs_mean[iteration],2))
            # non informative
            self.prev_an = an
            self.prev_bn = bn
            self.gibbs_var[iteration] = 1 / np.random.gamma(an, 1 / bn)  

    def gibbs_burn_and_mean(self):
        self.gibbs_mean = self.gibbs_mean[-self.n_gibbs:]
        self.gibbs_var = self.gibbs_var[-self.n_gibbs:] 
        self.mean = np.mean(self.gibbs_mean)
        self.var = np.mean(self.gibbs_var)
    
    def gibbs_prob(self, y:np.ndarray, iteration:int, **kwargs):
        '''
        changes prob array in place
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseGaussianEmission only work for univariate observations"
            y = y[:,0]                
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"        
        out = np.exp(-0.5*np.power(y-self.gibbs_mean[iteration-1],2)/self.gibbs_var[iteration-1])
        out /= np.sqrt(2*np.pi*self.gibbs_var[iteration-1])
        out[out < self.LOWER] = self.LOWER
        return out
    
    def prob(self, y:np.ndarray, **kwargs):
        '''
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uBaseGaussianEmission only work for univariate observations"
            y = y[:,0]    
        out = np.exp(-0.5*np.power(y-self.mean,2)/self.var)
        out /= np.sqrt(2*np.pi*self.var)
        return out
    
    def posterior_moments(self, **kwargs):
        return self.mean, self.var
    



class uBaseGaussianMixtureEmission(uHMMBaseEmission):
    def __init__(self, 
                 k_components:int = 2, 
                 n_gibbs:int = 1000, 
                 f_burn:float = 0.1, 
                 min_points_update = 5
                ):
        
        self.LOWER = 1e-8        
        self.k_components = k_components
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.min_points_update = min_points_update

    def view(self, plot_hist = False, plot = False, **kwargs):
        print('uBaseGaussianMixtureEmission')
        print('phi: ', self.phi)
        print('means: ', self.mean)
        print('scales: ', np.sqrt(self.var))

    def gibbs_initialize(self, y, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianMixtureEmissions only work for univariate observations"
            y = y[:,0]
        # this will be updated later
        self.mean = 0         
        # Covariance samples
        self.gibbs_var = np.zeros((self.n_gibbs_sim, self.k_components)) 
        # Mean samples
        self.gibbs_mean = np.zeros((self.n_gibbs_sim, self.k_components))   
        # Phi samples
        self.gibbs_phi = np.zeros((self.n_gibbs_sim, self.k_components))           
        # compute data variance
        self.y_var = np.var(y)
        self.y_scale = np.std(y)
        # Prior distribution parameters
        self.m0 = np.zeros(self.k_components)
        self.v0 = 1000*self.y_var*np.ones(self.k_components) # mean: prior covariance        
        self.a0 = 2*np.ones(self.k_components) # infinite variance...
        self.b0 = 0.01*self.y_var*np.ones(self.k_components)
        # initialize
        self.gibbs_mean[0] = self.m0
        self.gibbs_var[0] = self.y_var
        self.gibbs_phi[0] = 1 / float(self.k_components)
        # store parameters
        self.prev_mn = np.copy(self.m0)
        self.prev_vn = np.copy(self.v0)
        
        self.prev_an = np.copy(self.a0)
        self.prev_bn = np.copy(self.b0)
        
        self.prev_n_count = np.ones(self.k_components)

    def gibbs_posterior_sample(self, y:np.ndarray, iteration:int, **kwargs):
        '''
        y: current set of observations
        to be called while in sampler
        '''
        # update each one
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianMixtureEmissions only work for univariate observations"
            y = y[:,0]        
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"
        # no observations in y
        # for each state do the update
        
        if y.size < self.min_points_update:
            for k in range(self.k_components):
                self.gibbs_mean[iteration, k] = np.random.normal(self.prev_mn[k], self.prev_vn[k])
                self.gibbs_var[iteration, k] = 1 / np.random.gamma(self.prev_an[k], 1 / self.prev_bn[k])
            self.gibbs_phi[iteration] = np.random.dirichlet(self.prev_n_count + 1)
        
        else:                
            n = y.size
            # declare array
            aux = np.zeros((n, self.k_components))                
            # sample c
            for k in range(self.k_components):
                aux[:,k] = self.gibbs_phi[iteration-1, k] * np.exp(-0.5*np.power(y - self.gibbs_mean[iteration-1, k],2) / self.gibbs_var[iteration-1, k]) / np.sqrt(2*np.pi*self.gibbs_var[iteration-1, k])
            
            # this is a hack to sample fast from a multinomial with different probabilities!
            aux[aux < self.LOWER] = self.LOWER
            aux /= np.sum(aux, axis = 1)[:,None]
            uni = np.random.uniform(0, 1, size = n)
            aux = np.cumsum(aux, axis = 1)
            wrows, wcols = np.where(aux > uni[:,None])
            un, un_idx = np.unique(wrows, return_index = True)
            c_ = wcols[un_idx]

            # sample for each substate
            n_count = np.zeros(self.k_components)
            for k in range(self.k_components):
                idx_states = np.where(c_ == k)[0]            
                n_count[k] = idx_states.size 
                if idx_states.size < self.min_points_update:
                    self.gibbs_mean[iteration, k] = np.random.normal(self.prev_mn[k], self.prev_vn[k])
                    self.gibbs_var[iteration, k] = 1 / np.random.gamma(self.prev_an[k], 1 / self.prev_bn[k])
                else:                
                    y_sum = np.sum(y[idx_states])            
                    # Sample from mean
                    mn = (self.m0[k]*self.gibbs_var[iteration-1, k] + self.v0[k]*y_sum)
                    mn /= (n_count[k]*self.v0[k] + self.gibbs_var[iteration-1, k])            
                    vn = self.gibbs_var[iteration-1, k] * self.v0[k] / (n_count[k]*self.v0[k] + self.gibbs_var[iteration-1, k])

                    self.prev_mn[k] = mn
                    self.prev_vn[k] = vn
                    self.gibbs_mean[iteration, k] = np.random.normal(mn, vn)
                    # Sample from variance
                    an = self.a0[k] + n_count[k] / 2
                    bn = self.b0[k] + 0.5*np.sum(np.power(y[idx_states]-self.gibbs_mean[iteration, k],2))
                    self.prev_an[k] = an
                    self.prev_bn[k] = bn
                    self.gibbs_var[iteration, k] = 1 / np.random.gamma(an, 1 / bn)  
            # update phis
            self.gibbs_phi[iteration] = np.random.dirichlet(n_count + 1)
            self.prev_n_count = n_count
                
    def gibbs_burn_and_mean(self):
        self.gibbs_mean = self.gibbs_mean[-self.n_gibbs:]
        self.gibbs_var = self.gibbs_var[-self.n_gibbs:] 
        self.gibbs_phi = self.gibbs_phi[-self.n_gibbs:] 
        
        self.mean = np.mean(self.gibbs_mean, axis = 0)
        self.var = np.mean(self.gibbs_var, axis = 0)
        self.phi = np.mean(self.gibbs_phi, axis = 0)
        
        self.k_mean = np.dot(self.mean,self.phi)
        self.k_var = np.dot(self.var, self.phi)
        
    def gibbs_prob(self, y:np.ndarray, iteration:int, **kwargs):
        '''
        changes prob array in place
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"        
        out = np.zeros(y.size)
        for k in range(self.k_components):
            out += self.gibbs_phi[iteration-1, k] * np.exp(-0.5*np.power(y-self.gibbs_mean[iteration-1, k], 2)/self.gibbs_var[iteration-1, k]) / np.sqrt(2*np.pi * self.gibbs_var[iteration-1, k])
        out[out < self.LOWER] = self.LOWER
        return out
    
    def prob(self, y:np.ndarray, **kwargs):
        '''
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        out = np.zeros(y.size)
        for k in range(self.k_components):
            out += self.phi[k] * np.exp(-0.5*np.power(y-self.mean[k],2)/self.var[k]) / np.sqrt(2*np.pi*self.var[k])
        out[out < self.LOWER] = self.LOWER
        return out

    def posterior_moments(self, **kwargs):
        return self.k_mean, self.k_var
    
 

# Generic univariate emissions          
class uHMMEmissions(HMMEmissions):
    def __init__(self, 
                 emissions:List[uHMMBaseEmission], 
                 n_gibbs:int = 1000, 
                 f_burn:float = 0.1, 
                 min_points_update = 5, 
                 normalize = True):
        self.emissions = emissions
        self.LOWER = 1e-16
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.min_points_update = min_points_update
        self.n_states = len(self.emissions)

    def set_parameters(self, mean, var):
        self.mean = mean
        self.var = var
    
    def view(self, plot_hist = False, plot = False, **kwargs):
        print('uHMMEmissions')
        for emission in self.emissions:
            emission.view(plot_hist = plot_hist, plot = plot)
    
    def gibbs_initialize(self, y, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "uHMMEmissions only work for univariate observations"
            y = y[:,0]
        for emission in self.emissions:
            emission.gibbs_initialize(y)

    def gibbs_posterior_sample(self, z:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        y: current set of observations
        to be called while in sampler
        '''
        # update each one
        if y.ndim == 2:
            assert y.shape[1] == 1, "uHMMEmissions only work for univariate observations"
            y = y[:,0]        
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"
        # no observations in y
        # for each state do the update
        
        for s, emission in enumerate(self.emissions):
            idx_states = np.where(z == s)[0]  
            emission.gibbs_posterior_sample(y[idx_states], iteration)
            
    def gibbs_burn_and_mean(self):
        self.mean = np.zeros(self.n_states)
        self.var = np.zeros(self.n_states)    
        for s, emission in enumerate(self.emissions):
            emission.gibbs_burn_and_mean()
            self.mean[s], self.var[s] = emission.posterior_moments()
        
        
    def gibbs_prob(self, prob:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        changes prob array in place
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uHMMEmissions only work for univariate observations"
            y = y[:,0]    
        assert prob.shape[1] == self.n_states, "prob array does not match dimensions"                
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"        
        for s, emission in enumerate(self.emissions):
            prob[:, s] = emission.gibbs_prob(y, iteration)
        
    def prob(self, y:np.ndarray, **kwargs):
        '''
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uHMMEmissions only work for univariate observations"
            y = y[:,0]    
        prob = np.zeros((y.shape[0], self.n_states))
        for s, emission in enumerate(self.emissions):
            prob[:, s] = emission.prob(y)
        return prob

    def posterior_predictive(self, next_state_prob:np.ndarray, **kwargs):
        # create ar
        mix_mean = np.dot(next_state_prob, self.mean)
        mix_var = np.dot(next_state_prob, self.var + self.mean*self.mean)
        return mix_mean, mix_var

    # later delete
    def get_n_states(self):
        """Subclasses must implement this method"""
        return self.n_states
    
    def set_gibbs_parameters(self, n_gibbs, f_burn, n_gibbs_sim = None):
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        aux = int(self.n_gibbs*(1+self.f_burn))
        self.n_gibbs_sim = aux if n_gibbs_sim is None else n_gibbs_sim     
        # set for each emission..
        for emission in self.emissions:
            emission.set_gibbs_parameters(self.n_gibbs, self.f_burn, self.n_gibbs_sim)                    
        
        





class HMM(BaseModel):
    def __init__(
                self,
                emissions:HMMEmissions,
                n_gibbs = 1000,
                f_burn = 0.1,
                A_zeros = [],
                pred_l = None,
                **kwargs
                ):
        '''
        emissions: instance of HMMEmissions class                
        n_gibbs: number of gibbs samples
        A_zeros: list of lists with entries of A to be set to zero
        f_burn: fraction to burn  
        '''
        self.emissions = emissions   
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.A_zeros = A_zeros
        self.n_states = self.emissions.get_n_states()
        # real number of samples to simulate
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.pred_l = pred_l
        self.P = None
        self.gibbs_P = None
        
        self.A = None
        self.gibbs_A = None
        
        self.w_norm = 1
        
        self.emissions.set_gibbs_parameters(self.n_gibbs, self.f_burn, self.n_gibbs_sim)
        
        # **
        # Dirichelet prior parameters
        self.ALPHA = 1
        self.ZERO_ALPHA = 0.001
        self.ALPHA_P = 0.05 
        # A initial mass (persistency)
        self.INIT_MASS = 0.9

    def set_parameters(self, A, P):
        self.A = A
        self.P = P

    def view(self, plot_hist = False, plot = False, **kwargs):
        '''
        plot_hist: if true, plot histograms, otherwise just show the parameters
        '''
        print('** HMM **')
        print('Initial state probability')
        print(self.P)
        if plot_hist:
            for i in range(self.n_states):
                plt.hist(self.gibbs_P[:,i], density=True, alpha=0.5, label='P[%s]'%(i))
            plt.legend()
            plt.grid(True)
            plt.show()
        print('State transition matrix')
        print(np.round(self.A,3))
        print()
        if plot_hist:
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if [i,j] not in self.A_zeros:
                        plt.hist(
                                self.gibbs_A[:,i,j],
                                density=True,
                                alpha=0.5,
                                label='A[%s->%s]'%(i,j)
                                )
            plt.legend()
            plt.grid(True)
            plt.show()
        self.emissions.view(plot_hist = plot_hist, plot = plot)

    def transitions_dirichlet_priors(self):
        alphas = []
        for s in range(self.n_states):
            tmp = self.ALPHA*np.ones(self.n_states)
            for e in self.A_zeros:
                if e[0] == s:
                    tmp[e[1]] = self.ZERO_ALPHA
            alphas.append(tmp)
        return alphas
    
    def estimate(self, y:np.ndarray, x:np.ndarray = None, t:np.ndarray = None, msidx = None, **kwargs):   
        # **
        # number of observations
                
        if y.ndim == 1:
            y = np.array(y)[:,None]

        n = y.shape[0]
        # idx for multisequence

        if msidx is None:
            msidx = np.array([[0, n]], dtype = int)
        else:
            assert msidx.ndim == 1, "msidx must be a vector"
            # convert into table
            msidx = tm.utils.msidx_to_table(msidx)
        #msidx = np.array([[0, n]], dtype = int)

        # convert to integer to make sure this is well defined
        msidx = np.array(msidx, dtype = int)
        #print(msidx)
        # number of sequences (now we are working on a table!)
        n_seqs = msidx.shape[0]
        # **
        # Dirichlet prior
        A_alphas = self.transitions_dirichlet_priors()
        # **
        # Containers
        # counter for state transitions
        transition_counter = np.zeros((self.n_states, self.n_states)) 
        # counter for initial state observations
        init_state_counter = np.zeros(self.n_states) 
        # probability of observation given state
        # this will be modified inside emissions
        prob = np.zeros((n, self.n_states), dtype = np.float64)         
        # forward alpha
        forward_alpha = np.zeros((n, self.n_states), dtype = np.float64)
        # forward c (normalizer)
        c = np.zeros(n, dtype = np.float64)
                
        # transition matrix samples
        self.gibbs_A = np.zeros((self.n_gibbs_sim, self.n_states, self.n_states)) 
        # initial state probability samples
        self.gibbs_P = np.zeros((self.n_gibbs_sim, self.n_states))        
        # ** 
        # **
        # Initialize
        # Transition matrix A
        # assume some persistency of state as a initial parameter
        tmp = self.INIT_MASS * np.eye(self.n_states)
        remaining_mass = (1-self.INIT_MASS) / (self.n_states - 1)
        tmp[tmp == 0] = remaining_mass  
        # set zeros
        for e in self.A_zeros:
            tmp[e[0],e[1]] = 0
        # normalize
        tmp /= np.sum(tmp,axis=1)[:,None]
        self.gibbs_A[0] = tmp
        
        # Initial State Distribution
        self.gibbs_P[0] = np.ones(self.n_states)
        self.gibbs_P[0] /= np.sum(self.gibbs_P[0])      
        
        # initialize emissions
        self.emissions.gibbs_initialize(y = y, x = x, t = t)
        # **
        # create and initialize variable with
        # the states associated with each variable
        # assume equal probability in states
        z = np.random.choice(np.arange(self.n_states, dtype = int), size = n)
        # **
        # Gibbs sampler
        for i in range(1, self.n_gibbs_sim):
            # **
            # set counters to zero
            transition_counter *= 0 
            init_state_counter *= 0 
            # **
            # evaluate the probability of each observation
            # this modifies prob variable!
            self.emissions.gibbs_prob(prob = prob, y = y, x = x, t = t, iteration = i)            
            # **
            # sample form hidden state variable
            for l in range(n_seqs):
                # compute alpha variable
                hmm_forward(
                            prob[msidx[l][0]:msidx[l][1]],
                            self.gibbs_A[i-1],
                            self.gibbs_P[i-1],
                            forward_alpha[msidx[l][0]:msidx[l][1]], 
                            c[msidx[l][0]:msidx[l][1]]
                            )

                # backward walk to sample from the state sequence
                hmm_backward_sample(
                                    self.gibbs_A[i-1],
                                    forward_alpha[msidx[l][0]:msidx[l][1]],
                                    z[msidx[l][0]:msidx[l][1]],
                                    transition_counter,
                                    init_state_counter
                                    )
            # **
            # sample from transition matrix
            for s in range(self.n_states):
                self.gibbs_A[i,s] = np.random.dirichlet(A_alphas[s] + transition_counter[s])
            # make sure that the entries are zero!
            for A_zero in self.A_zeros:
                self.gibbs_A[i, A_zero[0], A_zero[1]] = 0.
            # normalize 
            self.gibbs_A[i] /= np.sum(self.gibbs_A[i],axis=1)[:,None]
            # **
            # sample from initial state distribution
            self.gibbs_P[i] = np.random.dirichlet(self.ALPHA_P + init_state_counter)   
            # perform the gibbs step with the state sequence sample q
            self.emissions.gibbs_posterior_sample(z = z, y = y, x = x, t = t, iteration = i)

        # burn observations
        self.gibbs_A = self.gibbs_A[-self.n_gibbs:]
        self.gibbs_P = self.gibbs_P[-self.n_gibbs:]
        self.emissions.gibbs_burn_and_mean()
        # take mean
        self.A = np.mean(self.gibbs_A, axis = 0)
        self.P = np.mean(self.gibbs_P, axis = 0)
        # self.view(True)


    def posterior_predictive(self, y:np.ndarray, x:np.ndarray = None, t:np.ndarray = None, msidx:np.ndarray = None, **kwargs):
        
    
        if y.ndim == 1:
            y = np.array(y)[:,None]

        n, p = y.shape
        
        # idx for multisequence
        if msidx is None:
            msidx = np.array([[0, n]], dtype = int)
        else:
            assert msidx.ndim == 1, "msidx must be a vector"
            # convert into table
            msidx = tm.utils.msidx_to_table(msidx)
        # convert to integer to make sure this is well defined
        msidx = np.array(msidx, dtype = int)
        # number of sequences (now we are working on a table!)
        n_seqs = msidx.shape[0]     
        #
        forward_alpha = np.zeros((n, self.n_states), dtype = np.float64)
        c = np.zeros(n, dtype = np.float64)
        prob = self.emissions.prob(y = y, x = x, t = t)          

        pred_m = np.zeros((n, p))
        pred_c = np.zeros((n, p, p))
        
        for l in range(n_seqs):
            # compute alpha variable
            hmm_forward(
                        prob[msidx[l][0]:msidx[l][1]],
                        self.A,
                        self.P,
                        forward_alpha[msidx[l][0]:msidx[l][1]], 
                        c[msidx[l][0]:msidx[l][1]]
                        )
            pred_c[msidx[l][0]] = np.eye(p)
            # print(pred_c[msidx[l][0]])
            for m in range(msidx[l][0]+1, msidx[l][1]):
                next_state_prob = np.dot(self.A.T, forward_alpha[m-1])  
                xq = None if not x else x[m]
                tq = None if not t else t[m]                                
                pred_m_, pred_c_ = self.emissions.posterior_predictive(next_state_prob = next_state_prob, xq = xq, tq = tq)
                # print(m, pred_m_, pred_c_)
                pred_m[m], pred_c[m] = np.atleast_1d(pred_m_), np.atleast_2d(pred_c_)
        return pred_m, pred_c


    # def _evaluate(self, y:np.ndarray, x:np.ndarray = None, msidx:np.ndarray = None, **kwargs):
    #     n = y.shape[0]
    #     # idx for multisequence
    #     if msidx is None:
    #         msidx = np.array([[0, n]], dtype = int)
    #     else:
    #         assert msidx.ndim == 1, "msidx must be a vector"
    #         # convert into table
    #         msidx = tm.utils.msidx_to_table(msidx)
    #     # convert to integer to make sure this is well defined
    #     msidx = np.array(msidx, dtype = int)
    #     # number of sequences (now we are working on a table!)
    #     n_seqs = msidx.shape[0]            
    #     #
    #     forward_alpha = np.zeros((n, self.n_states), dtype = np.float64)
    #     c = np.zeros(n, dtype = np.float64)
    #     prob = self.emissions.prob(y = y, x = x, t = t)  
    #     for l in range(n_seqs):
    #         # compute alpha variable
    #         hmm_forward(
    #                     prob[msidx[l][0]:msidx[l][1]],
    #                     self.A,
    #                     self.P,
    #                     forward_alpha, 
    #                     c
    #                     )

    #     w = np.zeros_like(y)
    #     for i in range(1, n):
    #         next_state_prob = np.dot(self.A.T, forward_alpha[i-1])  
    #         xq = None if not x else x[i]
    #         w[i] = self.emissions.get_weight(next_state_prob = next_state_prob, xq = xq)
    #     return np.atleast_2d(w.T).T     
    
    # def get_weight(self,  y:np.ndarray, x:np.ndarray = None, xq:np.ndarray = None, **kwargs):
        
    #     n = y.shape[0]
    #     prob = self.emissions.prob(y = y, x = x, t = t)  
    #     forward_alpha = np.zeros((n, self.n_states), dtype = np.float64)
    #     c = np.zeros(n, dtype = np.float64)
    #     hmm_forward(
    #                 prob,
    #                 self.A,
    #                 self.P,
    #                 forward_alpha, 
    #                 c
    #                 )
    #     next_state_prob = np.dot(self.A.T, forward_alpha[-1]) 
    #     return self.emissions.get_weight(next_state_prob = next_state_prob, xq = xq)

class uGaussianEmissions(HMMEmissions):
    def __init__(self, n_states = 2, n_gibbs:int = 1000, f_burn:float = 0.1, min_points_update = 5, normalize = True):
        self.LOWER = 1e-16
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.min_points_update = min_points_update
        self.n_states = n_states
        # variables to be computed
        self.var = None
        self.mean = None        
        self.S0aux = None
        self.invV0, self.invV0m0 = None, None
        self.prev_mn, self.prev_Vn = None, None
        self.prev_vn, self.prev_Sn = None, None 
        self.normalize = normalize
        self.w_norm = 1

    def set_parameters(self, mean, var):
        self.mean = mean
        self.var = var
    
    def view(self, plot_hist = False, plot = False, **kwargs):
        print('uGaussianEmissions')
        print('mean: ', self.mean)
        print('scale: ', np.sqrt(self.var))
        if plot:
            plt.plot(self.mean, '.-', label = 'Mean')
            plt.plot(np.sqrt(self.var), '.-', label = 'Scale')
            plt.legend()
            plt.show()
        if plot_hist:
            for s in range(self.n_states):
                plt.hist(
                        self.gibbs_mean[:,s],
                        density=True,
                        alpha=0.5,
                        label=f'Posterior mean for state {s+1} samples histogram'
                        )
            plt.legend()
            plt.grid(True)
            plt.show()
            for s in range(self.n_states):                
                plt.hist(
                        self.gibbs_var[:,s],
                        density=True,
                        alpha=0.5,
                        label=f'Posterior variance for state {s+1} samples histogram'
                        )
            plt.legend()
            plt.grid(True)
            plt.show()            
            
    def gibbs_initialize(self, y, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]
        # this will be updated later
        self.mean = 0         
        # Covariance samples
        self.gibbs_var = np.zeros((self.n_gibbs_sim, self.n_states)) 
        # Mean samples
        self.gibbs_mean = np.zeros((self.n_gibbs_sim, self.n_states))          
        # compute data variance
        self.y_var = np.var(y)
        self.y_scale = np.std(y)
        # Prior distribution parameters
        self.m0 = np.zeros(self.n_states)
        #for s in range(self.n_states):
        #    self.m0[s] = ((-1)**s)*np.abs(self.y_scale)
        #print(self.m0)
        self.v0 = 1000*self.y_var*np.ones(self.n_states) # mean: prior covariance        
        self.a0 = 2*np.ones(self.n_states) # infinite variance...
        self.b0 = 0.01*self.y_var*np.ones(self.n_states)        
        # initialize
        self.gibbs_mean[0] = self.m0
        self.gibbs_var[0] = self.y_var*np.ones(self.n_states)
        # store parameters
        self.prev_mn = np.copy(self.m0)
        self.prev_vn = np.copy(self.v0)
        
        self.prev_an = np.copy(self.a0)
        self.prev_bn = np.copy(self.b0)

    def gibbs_posterior_sample(self, z:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        y: current set of observations
        to be called while in sampler
        '''
        # update each one
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]        
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"
        # no observations in y
        # for each state do the update
        
        for s in range(self.n_states):
            idx_states = np.where(z == s)[0]            
            if idx_states.size < self.min_points_update:
                self.gibbs_mean[iteration, s] = np.random.normal(self.prev_mn[s], self.prev_vn[s])
                self.gibbs_var[iteration, s] = 1 / np.random.gamma(self.prev_an[s], 1 / self.prev_bn[s])
            else:                
                n = idx_states.size
                y_sum = np.sum(y[idx_states])            
                # Sample from mean
                mn = (self.m0[s]*self.gibbs_var[iteration-1, s] + self.v0[s]*y_sum)
                mn /= (n*self.v0[s] + self.gibbs_var[iteration-1, s])            
                vn = self.gibbs_var[iteration-1, s] * self.v0[s] / (n*self.v0[s] + self.gibbs_var[iteration-1, s])
                
                # non informative
                #mn = y_sum / n
                #vn = self.gibbs_var[iteration-1, s]/n
                                
                self.prev_mn[s] = mn
                self.prev_vn[s] = vn
                self.gibbs_mean[iteration, s] = np.random.normal(mn, vn)
                # Sample from variance
                an = self.a0[s] + n/2
                # bn = self.b0[s] + 0.5*np.sum(np.power(y[idx_states]-y_sum/n,2))
                bn = self.b0[s] + 0.5*np.sum(np.power(y[idx_states]-self.gibbs_mean[iteration, s],2))
                # non informative
                #an = n/2                
                #bn = 0.5*np.sum(np.power(y[idx_states]-y_sum/n,2))
                self.prev_an[s] = an
                self.prev_bn[s] = bn
                self.gibbs_var[iteration, s] = 1 / np.random.gamma(an, 1 / bn)  

    def gibbs_burn_and_mean(self):
        self.gibbs_mean = self.gibbs_mean[-self.n_gibbs:]
        self.gibbs_var = self.gibbs_var[-self.n_gibbs:] 
        self.mean = np.mean(self.gibbs_mean, axis = 0)
        self.var = np.mean(self.gibbs_var, axis = 0)
        self.w_norm = np.max(np.abs(self.mean) / self.var)
    
    def gibbs_prob(self, prob:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        changes prob array in place
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        assert prob.shape[1] == self.n_states, "prob array does not match dimensions"                
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"        
        for s in range(self.n_states):
            prob[:, s] = np.exp(-0.5*np.power(y-self.gibbs_mean[iteration-1, s],2)/self.gibbs_var[iteration-1, s])
            prob[:, s] /= np.sqrt(2*np.pi*self.gibbs_var[iteration-1, s])
        #prob[prob<self.LOWER] = 0
    
    def prob(self, y:np.ndarray, **kwargs):
        '''
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        prob = np.zeros((y.shape[0], self.n_states))
        for s in range(self.n_states):
            prob[:, s] = np.exp(-0.5*np.power(y-self.mean[s],2)/self.var[s])
            prob[:, s] /= np.sqrt(2*np.pi*self.var[s])
        return prob

    def posterior_predictive(self, next_state_prob:np.ndarray, **kwargs):
        mix_mean = np.dot(next_state_prob, self.mean)
        mix_var = np.dot(next_state_prob, self.var + self.mean*self.mean)
        return mix_mean, mix_var

class uLaplaceEmissions(HMMEmissions):
    def __init__(self, n_states = 2, n_gibbs:int = 1000, f_burn:float = 0.1, min_points_update = 5):
        self.LOWER = 1e-16
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.min_points_update = min_points_update
        self.n_states = n_states
        # variables to be computed
        self.var = None
        self.mean = None        
        self.S0aux = None
        self.invV0, self.invV0m0 = None, None
        self.prev_mn, self.prev_Vn = None, None
        self.prev_vn, self.prev_Sn = None, None 

    def set_parameters(self, mean, var):
        self.mean = mean
        self.var = var
    
    def view(self, plot_hist = False, plot = False, **kwargs):
        print('uLaplaceEmissions')
        print('mean: ', self.mean)
        print('scale: ', self.b)
        if plot:
            plt.plot(self.mean, '.-', label = 'Mean')
            plt.plot(self.b, '.-', label = 'Scale')
            plt.legend()
            plt.show()
    
    def gibbs_initialize(self, y, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]
        # Scale samples
        self.gibbs_b = np.zeros((self.n_gibbs_sim, self.n_states)) 
        # Mean samples
        self.gibbs_mean = np.zeros((self.n_gibbs_sim, self.n_states))          
        
        # compute data variance
        y_var = np.var(y)
        
        self.gibbs_b[0] = np.sqrt(y_var) 
        self.gibbs_mean[0] = np.mean(y)
        
        self.m0 = 0
        self.s0 = 1000*y_var
        self.alpha0 = 2
        self.beta0 = 0.01*y_var   
        
        self.prev_mn = self.m0*np.ones(self.n_states)
        self.prev_sn = self.s0*np.ones(self.n_states)
        self.prev_alphan = self.alpha0*np.ones(self.n_states)
        self.prev_betan = self.beta0*np.ones(self.n_states)
        
    def gibbs_posterior_sample(self, z:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        y: current set of observations
        to be called while in sampler
        '''
        # update each one
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]        
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"
        # no observations in y
        # for each state do the update
        
        for s in range(self.n_states):
            idx_states = np.where(z == s)[0]            
            if idx_states.size < self.min_points_update:
                b2 = invgamma.rvs(self.prev_alphan[s], scale=self.prev_betan[s])
                self.gibbs_b[iteration, s] = np.sqrt(b2)              
                self.gibbs_mean[iteration, s] = norm.rvs(self.prev_mn[s], self.prev_sn[s])
            else:                
                n = idx_states.size
                
                y_ = y[idx_states]
                y_median = np.median(y_)
                
                tau = sample_taus(y_, self.gibbs_mean[iteration-1, s], self.gibbs_b[iteration-1, s])

                # 2) Update mu
                prec = 1/self.s0**2 + np.sum(1/tau)
                mean_mu = (self.m0/self.s0**2 + np.sum(y_/tau)) / prec
                self.gibbs_mean[iteration, s] = norm.rvs(mean_mu, np.sqrt(1/prec))

                # 3) Update b^2
                alpha = self.alpha0 + n
                beta = self.beta0 + 0.5*np.sum(tau)
                b2 = invgamma.rvs(alpha, scale=beta)
                self.gibbs_b[iteration, s] = np.sqrt(b2)   

                self.prev_mn[s] = mean_mu
                self.prev_sn[s] = np.sqrt(1/prec)
                self.prev_alphan[s] = alpha
                self.prev_betan[s] = beta
                
                
                
    def gibbs_burn_and_mean(self):
        self.gibbs_mean = self.gibbs_mean[-self.n_gibbs:]
        self.gibbs_b = self.gibbs_b[-self.n_gibbs:] 
        self.mean = np.mean(self.gibbs_mean, axis = 0)
        self.b = np.mean(self.gibbs_b, axis = 0)
        
    def gibbs_prob(self, prob:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        changes prob array in place
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        assert prob.shape[1] == self.n_states, "prob array does not match dimensions"                
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"        
        for s in range(self.n_states):
            prob[:, s] = np.exp(-np.abs(y-self.gibbs_mean[iteration-1, s])/self.gibbs_b[iteration-1, s])
            prob[:, s] /= (2*self.gibbs_b[iteration-1, s])
        prob[prob < self.LOWER] = self.LOWER
        
    def prob(self, y:np.ndarray, **kwargs):
        '''
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        prob = np.zeros((y.shape[0], self.n_states))
        for s in range(self.n_states):
            prob[:, s] = np.exp(-np.abs(y-self.mean[s])/self.b[s])
            prob[:, s] /= (2*self.b[s])
        return prob

    def posterior_predictive(self, next_state_prob:np.ndarray, **kwargs):
        mix_mean = np.dot(next_state_prob, self.mean)
        mix_var = np.dot(next_state_prob, 2*self.b*self.b + self.mean*self.mean)
        return mix_mean, mix_var


class uGaussianMixtureEmissions(HMMEmissions):
    def __init__(self, 
                 n_states:int = 2, 
                 k_components:int = 2, 
                 n_gibbs:int = 1000, 
                 f_burn:float = 0.1, 
                 min_points_update = 5
                ):
        
        self.LOWER = 1e-8
        
        self.n_states = n_states
        self.k_components = k_components
        self.n_gibbs = n_gibbs
        self.f_burn = f_burn
        self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
        self.min_points_update = min_points_update
        
        # variables to be computed
        self.var = None
        self.mean = None        
        self.phi = None
        self.S0aux = None
        self.invV0, self.invV0m0 = None, None
        self.prev_mn, self.prev_Vn = None, None
        self.prev_vn, self.prev_Sn = None, None 
    
    def view(self, plot_hist = False, plot = False, **kwargs):
        print('uGaussianMixtureEmissions')
        print('phi: ', self.phi)
        print('means: ', self.mean)
        print('scales: ', np.sqrt(self.var))
        print()
        print('k mean: ', self.k_mean)
        print('k scale: ', np.sqrt(self.k_var))
        print()
        if plot:
            for i in range(self.n_states):
                k_std = 3
                x_pdf_min = np.min(self.mean[i] - k_std*np.sqrt(self.var[i]))
                x_pdf_max = np.max(self.mean[i] + k_std*np.sqrt(self.var[i]))
                x_pdf = np.linspace(x_pdf_min, x_pdf_max, 500)
                pdf = np.zeros_like(x_pdf)
                for k in range(self.k_components):
                    pdf += self.phi[i,k] * np.exp(-0.5*np.power(x_pdf-self.mean[i,k],2)/self.var[i,k])/np.sqrt(2*np.pi*self.var[i,k])
                plt.plot(x_pdf, pdf, label = f'Mix distribution for state {i+1}')
                plt.legend()
                plt.show()
                
    def gibbs_initialize(self, y, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianMixtureEmissions only work for univariate observations"
            y = y[:,0]
        # this will be updated later
        self.mean = 0         
        # Covariance samples
        self.gibbs_var = np.zeros((self.n_gibbs_sim, self.n_states, self.k_components)) 
        # Mean samples
        self.gibbs_mean = np.zeros((self.n_gibbs_sim, self.n_states, self.k_components))   
        # Phi samples
        self.gibbs_phi = np.zeros((self.n_gibbs_sim, self.n_states, self.k_components))           
        # compute data variance
        self.y_var = np.var(y)
        self.y_scale = np.std(y)
        # Prior distribution parameters
        self.m0 = np.zeros((self.n_states, self.k_components))
        self.v0 = 1000*self.y_var*np.ones((self.n_states, self.k_components)) # mean: prior covariance        
        self.a0 = 2*np.ones((self.n_states, self.k_components)) # infinite variance...
        self.b0 = 0.01*self.y_var*np.ones((self.n_states, self.k_components))        
        # initialize
        self.gibbs_mean[0] = self.m0
        self.gibbs_var[0] = self.y_var# *np.ones((self.n_states)
        self.gibbs_phi[0] = 1 / float(self.k_components)
        # store parameters
        self.prev_mn = np.copy(self.m0)
        self.prev_vn = np.copy(self.v0)
        
        self.prev_an = np.copy(self.a0)
        self.prev_bn = np.copy(self.b0)
        
        self.prev_n_count = np.ones(self.k_components)

    def gibbs_posterior_sample(self, z:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        y: current set of observations
        to be called while in sampler
        '''
        # update each one
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianMixtureEmissions only work for univariate observations"
            y = y[:,0]        
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"
        # no observations in y
        # for each state do the update
        
        for s in range(self.n_states):
            idx_states = np.where(z == s)[0]            
            if idx_states.size < self.min_points_update:
                for k in range(self.k_components):
                    self.gibbs_mean[iteration, s, k] = np.random.normal(self.prev_mn[s, k], self.prev_vn[s, k])
                    self.gibbs_var[iteration, s, k] = 1 / np.random.gamma(self.prev_an[s, k], 1 / self.prev_bn[s, k])
                self.gibbs_phi[iteration, s] = np.random.dirichlet(self.prev_n_count + 1)
            else:                
                n = idx_states.size
                y_ = y[idx_states]
                # declare array
                aux = np.zeros((n, self.k_components))                
                # sample c
                for k in range(self.k_components):
                    aux[:,k] = self.gibbs_phi[iteration-1, s, k] * np.exp(-0.5*np.power(y_ - self.gibbs_mean[iteration-1, s, k],2) / self.gibbs_var[iteration-1, s, k]) / np.sqrt(2*np.pi*self.gibbs_var[iteration-1, s, k])
                    # aux[:,j] = self.gibbs_phi[iteration-1, s, k] * np.exp(-0.5*np.power(y_ - self.gibbs_mean[iteration-1, s, k],2) / self.gibbs_var[iteration-1, s, k]) / np.sqrt(2*np.pi*self.gibbs_var[iteration-1, s, k])
                
                # this is a hack to sample fast from a multinomial with different probabilities!
                aux[aux < self.LOWER] = self.LOWER
                aux /= np.sum(aux, axis = 1)[:,None]
                uni = np.random.uniform(0, 1, size = n)
                aux = np.cumsum(aux, axis = 1)
                wrows, wcols = np.where(aux > uni[:,None])
                un, un_idx = np.unique(wrows, return_index = True)
                c_ = wcols[un_idx]
                
                # sample for each substate
                n_count = np.zeros(self.k_components)
                for k in range(self.k_components):
                    idx_states_ = np.where(c_ == k)[0]            
                    n_count[k] = idx_states_.size 
                    if idx_states.size < self.min_points_update:
                        # TODO!
                        self.gibbs_mean[iteration, s, k] = np.random.normal(self.prev_mn[s, k], self.prev_vn[s, k])
                        self.gibbs_var[iteration, s, k] = 1 / np.random.gamma(self.prev_an[s, k], 1 / self.prev_bn[s, k])
                    else:                
                        y_sum_ = np.sum(y_[idx_states_])            
                        # Sample from mean
                        mn = (self.m0[s, k]*self.gibbs_var[iteration-1, s, k] + self.v0[s, k]*y_sum_)
                        mn /= (n_count[k]*self.v0[s, k] + self.gibbs_var[iteration-1, s, k])            
                        vn = self.gibbs_var[iteration-1, s, k] * self.v0[s, k] / (n_count[k]*self.v0[s, k] + self.gibbs_var[iteration-1, s, k])
                        
                        self.prev_mn[s, k] = mn
                        self.prev_vn[s, k] = vn
                        self.gibbs_mean[iteration, s, k] = np.random.normal(mn, vn)
                        # Sample from variance
                        an = self.a0[s, k] + n_count[k] / 2
                        bn = self.b0[s, k] + 0.5*np.sum(np.power(y_[idx_states_]-self.gibbs_mean[iteration, s, k],2))
                        self.prev_an[s, k] = an
                        self.prev_bn[s, k] = bn
                        self.gibbs_var[iteration, s, k] = 1 / np.random.gamma(an, 1 / bn)  
                # update phis
                self.gibbs_phi[iteration, s] = np.random.dirichlet(n_count + 1)
                self.prev_n_count = n_count
                
    def gibbs_burn_and_mean(self):
        self.gibbs_mean = self.gibbs_mean[-self.n_gibbs:]
        self.gibbs_var = self.gibbs_var[-self.n_gibbs:] 
        self.gibbs_phi = self.gibbs_phi[-self.n_gibbs:] 
        
        self.mean = np.mean(self.gibbs_mean, axis = 0)
        self.var = np.mean(self.gibbs_var, axis = 0)
        self.phi = np.mean(self.gibbs_phi, axis = 0)
        
        self.k_mean = np.sum(self.mean*self.phi, axis = 1)
        self.k_var = np.sum(self.var*self.phi, axis = 1)
        
    
    def gibbs_prob(self, prob:np.ndarray, y:np.ndarray, iteration:int, **kwargs):
        '''
        changes prob array in place
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        assert prob.shape[1] == self.n_states, "prob array does not match dimensions"                
        assert 0 < iteration < self.n_gibbs_sim, "iteration is larger than the number of iterations"        
        for s in range(self.n_states):
            # set to zero..
            prob[:, s] *= 0
            for k in range(self.k_components):
                tmp = np.exp(-0.5*np.power(y-self.gibbs_mean[iteration-1, s, k], 2)/self.gibbs_var[iteration-1, s, k])
                tmp /= np.sqrt(2*np.pi * self.gibbs_var[iteration-1, s, k])
                tmp *= self.gibbs_phi[iteration-1, s, k]                
                prob[:, s] += tmp
        prob[prob < self.LOWER] = self.LOWER

                
    def prob(self, y:np.ndarray, **kwargs):
        '''
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "uGaussianEmissions only work for univariate observations"
            y = y[:,0]    
        prob = np.zeros((y.shape[0], self.n_states))
        for s in range(self.n_states):
            for k in range(self.k_components):
                tmp = np.exp(-0.5*np.power(y-self.mean[s, k],2)/self.var[s, k])
                tmp /= np.sqrt(2*np.pi*self.var[s, k])
                tmp *= self.phi[s, k]
                prob[:, s] += tmp
        prob[prob < self.LOWER] = self.LOWER
        return prob

    def posterior_predictive(self, next_state_prob:np.ndarray, **kwargs):
        mix_mean = np.dot(next_state_prob, self.k_mean)
        mix_var = np.dot(next_state_prob, self.k_var + self.k_mean*self.k_mean)
        return mix_mean, mix_var



class FastTFHMM(BaseModel):
    def __init__(
                self,
                p = 0.97,
                **kwargs
                ):

        emissions = uGaussianEmissions(2)
        self.hmm = HMM(
                    emissions = emissions,
                    )

        self.P = np.array([0.5, 0.5])        
        self.A = np.array([[p, 1-p],[1-p, p]])
        self.hmm.set_parameters(A = self.A, P = self.P)

    def view(self, **kwargs):
        '''
        plot_hist: if true, plot histograms, otherwise just show the parameters
        '''
        self.hmm.view()

    def estimate(self, y, **kwargs):
        if y.ndim == 2:
            y = y[:,0]
        mean = np.zeros(2)
        var = np.ones(2)
        idx = y<0
        mean[0] = np.mean(y[idx])
        var[0] = np.var(y[idx])        
        mean[1] = np.mean(y[~idx])
        var[1] = np.var(y[~idx])
        self.hmm.emissions.set_parameters(mean = mean, var = var)

    def posterior_predictive(self, y, **kwargs):
        return self.hmm.posterior_predictive(y = y)


# simple test
if __name__ == '__main__':
    def simulate_hmm(n, A, mean, var):
        states = np.arange(A.shape[0], dtype = int)
        z = np.zeros(n, dtype = int)
        y = np.zeros((n, mean[0].size))
        z[0] = np.random.choice(states)
        y[0] = np.random.normal(mean[z[0]], np.sqrt(var[z[0]]))
        for i in range(1,n):
            z[i] = np.random.choice(states, p = A[z[i-1]])
            y[i] = np.random.normal(mean[z[i]],np.sqrt(var[z[i]]) )
        return y, z


    n = 2000
    A = np.array([[0.8,0.2],[0.3,0.7]])
    mean = np.array([0.2, -0.1])
    var = np.array([0.2, 0.1])
    y, z = simulate_hmm(n, A, mean, var)
    plt.plot(y)
    plt.show()

    #emissions = uFactorialMeanVarEmissions()
    emissions = uGaussianEmissions(2)
    hmm = HMM(
                emissions = emissions,
                n_gibbs = 2000
                )

    hmm = FastTFHMM()

    print('estimate')
    hmm.estimate(y)
    print('done')
    hmm.view()
    print(hmm.posterior_predictive(y = y))





