import numpy as np
import matplotlib.pyplot as plt
from tm.base_models import BaseModel
import tm

# numerical routines for bayesian linear regression

class CoreBayesLinRegr:
    def __init__(self, 
                 intercept = True, 
                 n_iter = 5000, 
                 tol = 1e-6, 
                 ard = False):
        self.intercept = intercept
        self.n_iter = n_iter
        self.tol = tol
        self.ard = ard
        self.LARGE = 1e18
        # to be computed
        self.w, self.b, self.a, self.S = None, None, None, None

    def view(self, plot = False, **kwargs):
        print('** Bayes Linear Regression **')
        print('Weights: ', self.w)
        print('Precision: ', self.b)
        print('Variance: ', 1/self.b)
        print('Scale: ', 1/np.sqrt(self.b))
        print('Prior precision: ', self.a)
        

        if plot:

            plt.title('Weights')
            plt.plot(self.w)
            plt.grid(True)
            plt.show()

            if self.ard:
                plt.title('Priors')
                plt.plot(1 / self.a)
                plt.grid(True)
                plt.show()
                plt.title('Prior precision convergence')
                for i in range(self.eb_a.shape[1]):
                    plt.plot(np.log(self.eb_a[:,i]), label = f'variable {i}')
                plt.grid(True)
                plt.legend()
                plt.show()
            else:
                plt.title('Prior precision convergence')
                plt.plot(self.eb_a)
                plt.grid(True)
                plt.show()                    
            plt.title('Precision convergence')
            plt.plot(self.eb_b)
            plt.grid(True)
            plt.show()    
        return self
        
    def estimate(self, y, x, v = None, **kwargs):
        '''
        y: numpy (n, ) array with targets
        x: numpy (n, p) array with features
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]
        assert x.ndim == 2, "x must be a matrix with the features!"
        assert y.size == x.shape[0], "y and x must have the same number of observations"
        compute_b = True
        if v is not None:
            compute_b = False
            if v.ndim == 2:
                assert v.shape[1] == 1, "v must contain a single variance"
                v = v[:, 0]
            assert y.size == v.size, "y and v must have the same number of observations"            
        else:
            v = np.ones(y.size, dtype = np.float64)
        
        # initialize b
        self.a = 1.
        self.b = 1. # / np.var(y)
        
        n = y.size
        if self.intercept:
            x = np.hstack((np.ones((n, 1)), x))
        
        # pre calculations
        p = x.shape[1]
        I = np.eye(p)
        XT_Dinv_X = x.T @ np.diag(1/v) @ x
        XT_Dinv_y = x.T @ np.diag(1/v) @ y            
            
        # 
        # Empirical Bayes
        # if ard, then we have a different a for each variable
        if self.ard:
            self.eb_a = np.zeros((self.n_iter, p))
        else:
            self.eb_a = np.zeros(self.n_iter)
        
        self.eb_b = np.ones(self.n_iter, dtype = np.float64)
        
        self.eb_a[0] = self.a
        self.eb_b[0] = self.b
                
        for i in range(1, self.n_iter):
            if self.ard:
                A = np.diag(self.eb_a[i-1])
            else:
                A = self.eb_a[i-1]*I

            b = self.eb_b[i-1]
            Sn = np.linalg.inv(A + b*XT_Dinv_X)
            wn = b * Sn @ XT_Dinv_y
            
            # update
            if self.ard:
                gamma = 1 - self.eb_a[i-1]*np.diag(Sn)
                tmp = gamma / np.power(wn, 2)                
                # control explosion in a for irrelevant features
                tmp[tmp>self.LARGE] = self.LARGE
                self.eb_a[i] = tmp            
                if compute_b:
                    self.eb_b[i] = (n-np.sum(gamma)) / np.sum(np.power(y - x@wn,2))
            else:
                tmp = p / np.dot(wn, wn)
                self.eb_a[i] = min(tmp, self.LARGE)
                if compute_b:
                    self.eb_b[i] = n / np.sum(np.power(y - x@wn,2))
        
            # evaluate convergence (use convergence in a!)
            d = np.mean(np.abs(self.eb_a[i] - self.eb_a[i-1]))
            # d = np.abs(self.eb_b[i]-self.eb_b[i-1])            
            if (d < self.tol):
                self.eb_a = self.eb_a[:i+1]
                self.eb_b = self.eb_b[:i+1]
                break
        if i == self.n_iter - 1:
            print('Evidence Approximation did not converge...')
        #
        # Fit final model
        self.a = self.eb_a[-1]
        self.b = self.eb_b[-1]
        if self.ard:
            A = np.diag(self.a)
        else:
            A = self.a*I
        self.S = np.linalg.inv(A + self.b*XT_Dinv_X)
        self.w = self.b * self.S @ XT_Dinv_y
        # 
        # Compute betting parameters
        # second non central moment otherwise weight
        # goes to zero for large moves
        if np.isnan(self.w).any():
            raise Exception('Error in computing evidence parameters. ')
        return self
    
    def posterior_predictive(self, x, v = None, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        assert x.ndim == 2, "x must be a matrix with the features!"
        n = x.shape[0]
        if self.intercept:
            x = np.hstack((np.ones((n, 1)), x))
        m = x @ self.w
        if v is not None:
            if v.ndim == 2:
                assert v.shape[1] == 1, "v must contain a single variance"
                v = v[:, 0]
            assert x.shape[0] == v.size, "x and v must have the same number of observations"            
        else:
            v = 1 / self.b + np.einsum('ij,jk,ik->i', x, self.S, x)
        return m, v

class BayesLinRegr(BaseModel):
    def __init__(self, 
                 w_quantile:float = 0.9, 
                 max_w:float = 1, 
                 intercept = True, 
                 n_iter = 5000, 
                 tol = 1e-6,
                 ard = False,  
                 post_w_norm = False               
                ):
        self.w_quantile = w_quantile
        self.max_w = max_w
        self.core_regr = CoreBayesLinRegr(intercept = intercept, n_iter = n_iter, tol = tol, ard = ard) 
        self.p = 1        
        self.w_norm = 1
        self.post_w_norm = post_w_norm

    def view(self, plot = False, **kwargs):
        self.core_regr.view(plot = plot)

    def estimate(self, y, x, **kwargs):        
        # base model estimate
        self.core_regr.estimate(y = y, x = x)
        
    def posterior_predictive(self, x, **kwargs):
        '''
        approximate with normal
        '''
        m, v = self.core_regr.posterior_predictive(x)
        return m, v

    #def get_weight(self, xq, **kwargs):
    #    if not isinstance(xq, np.ndarray):
    #        xq = np.array(xq)
    #    xq = np.atleast_2d(xq)                
    #    m, v = self.core_regr.posterior_predictive(xq)
    #    w = m / (v + m*m)
    #    w /= self.w_norm
    #    idx = np.abs(w) > self.max_w
    #    w[idx] = np.sign(w[idx])*self.max_w
    #    return w[-1]        

    #def _evaluate(self, x, **kwargs):
    #    m, v = self.core_regr.posterior_predictive(x)
    #    w = m / (v + m*m)
    #    w /= self.w_norm
    #    idx = np.abs(w) > self.max_w
    #    w[idx] = np.sign(w[idx])*self.max_w
    #    return np.atleast_2d(w.T).T        
        
    def post_estimate(self, y, x, **kwargs):
        if self.post_w_norm:
            m, v = self.core_regr.posterior_predictive(x)
            w = m / (v + m*m)
            self.w_norm = np.quantile(np.abs(w), self.w_quantile)

# ROLL VARIANCE REGR..

class RollVarBayesLinRegr:
    def __init__(self, 
                 window = 20,                 
                 w_quantile:float = 0.9, 
                 max_w:float = 1, 
                 intercept = True, 
                 n_iter = 5000, 
                 tol = 1e-6,
                 ard = False
                ):
        self.window = window
        
        self.w_quantile = w_quantile
        self.max_w = max_w
        self.core_regr = CoreBayesLinRegr(intercept = intercept, n_iter = n_iter, tol = tol, ard = ard) 
    
    def view(self, **kwargs):
        self.core_regr.view()
        
    def estimate(self, y, x, msidx = None, **kwargs):

        n = y.shape[0]
        # idx for multisequence

        if msidx is None:
            msidx = np.array([[0, n]], dtype = int)
        else:
            assert msidx.ndim == 1, "msidx must be a vector"
            # convert into table
            msidx = tm.utils.msidx_to_table(msidx)

        # number of sequences (now we are working on a table!)
        n_seqs = msidx.shape[0]

        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]  
        # later optimize this
        v = np.zeros_like(y)
        
        for l in range(n_seqs):
            # compute alpha variable
            for i in range(msidx[l][0] + self.window, msidx[l][1]):
                v[i] = np.var(y[i-self.window:i])
            v[msidx[l][0]:msidx[l][0] + self.window] = v[msidx[l][0] + self.window]            
        self.core_regr.estimate(y = y, x = x, v = v)

        # compute norm
        m, _ = self.core_regr.posterior_predictive(x, v = v)        
        w = m / (v + m*m)
        self.w_norm = np.quantile(np.abs(w), self.w_quantile)        

    def _evaluate(self, y, x, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]          
        v = np.zeros_like(y)
        for i in range(self.window, y.size):
            v[i] = np.var(y[i-self.window:i])
        v[:self.window] = v[self.window] 
        m, _ = self.core_regr.posterior_predictive(x, v = v)
        w = m / (v + m*m)
        w /= self.w_norm
        idx = np.abs(w) > self.max_w
        w[idx] = np.sign(w[idx])*self.max_w
        return np.atleast_2d(w.T).T  

    def get_weight(self, y, xq, **kwargs):        
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]    
        assert y.size > self.window, f"y must contain more than {self.window} observations"
        v = np.var(y[-self.window:])

        if not isinstance(xq, np.ndarray):
            xq = np.array(xq)
        
        m, v = self.core_regr.posterior_predictive(x = np.array([xq]), v = np.array([v]))
        w = m / (v + m*m)
        w /= self.w_norm
        idx = np.abs(w) > self.max_w
        w[idx] = np.sign(w[idx])*self.max_w
        return w[-1]    


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    model = RollVarRegr(window = 100, ard = True)
    n = 1000
    x = np.random.normal(0,1,n)
    y = 0.1 + 0.5*x + np.random.normal(0, 1, n)
    
    # add irrelevant feature
    x = np.hstack((x[:,None],np.random.normal(0,1,(n,1))))
    model.estimate(y = y, x = x)
    
    w = model._evaluate(y, x)
    print(w)
    print()
    plt.plot(w)
    plt.show()
    print(model.get_weight(y = y, xq = x[-1]))
    
