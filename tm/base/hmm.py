import numpy as np
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
    def gibbs_initialize(self, y:np.ndarray, x:np.ndarray = None, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_posterior_sample(self, z:np.ndarray, y:np.ndarray, x:np.ndarray, iteration:int, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_burn_and_mean(self):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def gibbs_prob(self, prob:np.ndarray, y:np.ndarray, x:np.ndarray, iteration:int, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def prob(self, y:np.ndarray, x:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass    

    @abstractmethod
    def get_weight(self, next_state_prob:np.ndarray, xq:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass        
        
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

    def view(self, plot_hist = False, **kwargs):
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
        self.emissions.view(plot_hist = plot_hist)

    def transitions_dirichlet_priors(self):
        alphas = []
        for s in range(self.n_states):
            tmp = self.ALPHA*np.ones(self.n_states)
            for e in self.A_zeros:
                if e[0] == s:
                    tmp[e[1]] = self.ZERO_ALPHA
            alphas.append(tmp)
        return alphas
    
    def estimate(self, y:np.ndarray, x:np.ndarray = None, msidx = None, **kwargs):   
        # **
        # number of observations
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
        self.emissions.gibbs_initialize(y = y, x = x)
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
            self.emissions.gibbs_prob(prob = prob, y = y, x = x, iteration = i)            
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
            self.emissions.gibbs_posterior_sample(z = z, y = y, x = x, iteration = i)

        # burn observations
        self.gibbs_A = self.gibbs_A[-self.n_gibbs:]
        self.gibbs_P = self.gibbs_P[-self.n_gibbs:]
        self.emissions.gibbs_burn_and_mean()
        # take mean
        self.A = np.mean(self.gibbs_A, axis = 0)
        self.P = np.mean(self.gibbs_P, axis = 0)
        # self.view(True)
    
    def _evaluate(self, y:np.ndarray, x:np.ndarray = None, msidx:np.ndarray = None, **kwargs):
        n = y.shape[0]
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
        prob = self.emissions.prob(y = y, x = x)  
        for l in range(n_seqs):
            # compute alpha variable
            hmm_forward(
                        prob[msidx[l][0]:msidx[l][1]],
                        self.A,
                        self.P,
                        forward_alpha, 
                        c
                        )

        w = np.zeros_like(y)
        for i in range(1, n):
            next_state_prob = np.dot(self.A.T, forward_alpha[i-1])  
            xq = None if not x else x[i]
            w[i] = self.emissions.get_weight(next_state_prob = next_state_prob, xq = xq)
        return np.atleast_2d(w.T).T     
    
    def get_weight(self,  y:np.ndarray, x:np.ndarray = None, xq:np.ndarray = None, **kwargs):
        
        n = y.shape[0]
        prob = self.emissions.prob(y = y, x = x)  
        forward_alpha = np.zeros((n, self.n_states), dtype = np.float64)
        c = np.zeros(n, dtype = np.float64)
        hmm_forward(
                    prob,
                    self.A,
                    self.P,
                    forward_alpha, 
                    c
                    )
        next_state_prob = np.dot(self.A.T, forward_alpha[-1]) 
        return self.emissions.get_weight(next_state_prob = next_state_prob, xq = xq)

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
    
    def view(self, plot_hist = False, **kwargs):
        print('uGaussianEmissions')
        print('mean: ', self.mean)
        print('var: ', self.var)
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

    def get_weight(self, next_state_prob:np.ndarray, **kwargs):
        mix_mean = np.dot(next_state_prob, self.mean)
        mix_var = np.dot(next_state_prob, self.var + self.mean*self.mean)
        # mix_var -= mix_mean*mix_mean
        if self.normalize:
            return mix_mean / mix_var / self.w_norm
        else:
            return mix_mean / mix_var

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
    hmm.estimate(y)
    hmm.view(False)
    print(hmm.get_weight(y = y))





