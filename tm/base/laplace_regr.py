
import numpy as np
import matplotlib.pyplot as plt
from tm.base import BaseModel
from scipy.stats import invgauss, invgamma, norm, gamma

# def sample_taus(z, z_center, s, eps=1e-12):
#     d = np.abs(z - z_center)
#     taus = np.empty_like(d)
#     # Case A: |x - mu| > 0  → sample u = 1/τ from IG, then invert
#     mask = d > eps
#     if np.any(mask):
#         mu_s = (s / d[mask])              # SciPy's 'mu' (shape)
#         scale = 1.0 / (s**2)              # SciPy's 'scale' = λ
#         u = invgauss.rvs(mu=mu_s, scale=scale, size=mask.sum())
#         u[u<1e-10] = 1e-10
#         taus[mask] = 1.0 / u
#     # Case B: |x - mu| = 0  → τ | d=0 ∝ τ^{-1/2} exp(-τ/(2 b^2))  = Gamma(1/2, rate=1/(2 b^2))
#     zero_mask = ~mask
#     if np.any(zero_mask):
#         shape = 0.5
#         rate = 1.0 / (2.0 * s**2)
#         scale_gamma = 1.0 / rate          # Gamma in SciPy is shape, scale
#         taus[zero_mask] = gamma.rvs(shape, scale=scale_gamma, size=zero_mask.sum())
#     return taus


# class LaplaceRegr():
#     def __init__(self, n_gibbs = 1000, f_burn = 0.1):
#         self.n_gibbs = n_gibbs
#         self.f_burn = f_burn
#         self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))

#     def view(self, plot = False, **kwargs):
#         print('** Laplace Regression **')
#         print('Bias: ', self.b)
#         print('Weights: ', self.w)
#         print('Scale: ', self.s)
    
#     def estimate(self, y, x, **kwargs):
#         '''
#         y: numpy (n, ) array with targets
#         x: numpy (n, p) array with features
#         '''
#         if y.ndim == 2:
#             assert y.shape[1] == 1, "y must contain a single target"
#             y = y[:, 0]
#         assert x.ndim == 2, "x must be a matrix with the features!"
#         assert y.size == x.shape[0], "y and x must have the same number of observations"
        
#         n = y.size
        
#         # initialize
#         self.gibbs_b = np.zeros(self.n_gibbs_sim)          
#         self.gibbs_w = np.zeros((self.n_gibbs_sim, x.shape[1]))    
#         self.gibbs_s = np.zeros(self.n_gibbs_sim) 
        
#         y_var = np.var(y)
#         # Prior distribution parameters
        
#         self.s0 = 1000*y_var # bias variance prior
#         self.a0 = 2 # scale a prior
#         self.b0 = 0.01*y_var # scale b prior
#         self.Psi0 = 1000*y_var*np.eye(x.shape[1])
#         self.Psi0_inv = np.linalg.inv(self.Psi0)
        
#         # initialize
#         self.gibbs_s[0] = np.sqrt(y_var)
                
#         # -----------
#         # sample
        
#         for i in range(1, self.n_gibbs_sim):
            
#             # sample taus (aux variables)
#             tau = sample_taus(y, self.gibbs_b[i-1]+x@self.gibbs_w[i-1], self.gibbs_s[i-1])
            
#             # sample s (scale)
#             alpha = self.a0 + n
#             beta = self.b0 + 0.5*np.sum(tau)
#             s2 = invgamma.rvs(alpha, scale=beta)
#             self.gibbs_s[i] = np.sqrt(s2)   
            
#             # sample b (bias)        
#             sn = 1 / (1 / self.s0 + np.sum(1/tau))            
#             cn = sn*np.sum((y - x @ self.gibbs_w[i-1])/tau)
#             self.gibbs_b[i] = np.random.normal(cn, sn)
            
#             # sample from weights (w)
            
#             Psin_inv = self.Psi0_inv + x.T@(x/tau[:,None])
#             Psin = np.linalg.inv(Psin_inv)
#             wn = Psin @ np.sum(x*((y-self.gibbs_b[i])/tau)[:,None], axis = 0)             
#             self.gibbs_w[i] = np.random.multivariate_normal(wn, Psin)
#         # ------
#         # burn and mean!
#         self.gibbs_s = self.gibbs_s[-self.n_gibbs:]
#         self.gibbs_b = self.gibbs_b[-self.n_gibbs:]
#         self.gibbs_w = self.gibbs_w[-self.n_gibbs:]
        
#         self.s = np.mean(self.gibbs_s)
#         self.b = np.mean(self.gibbs_b)
#         self.w = np.mean(self.gibbs_w, axis = 0)
        
#     def posterior_predictive(self, x, **kwargs):
#         '''
#         x: numpy (m, p) array
#         '''            
#         assert x.ndim == 2, "x must be a matrix with the features!"
#         m = self.b + x @ self.w        
#         return m, 2*self.s*self.b*np.ones_like(m)




def wls_centered_intercept(X, y, w, eps=1e-12):
    """
    Weighted least squares with an explicit intercept implemented via centering.
    Returns (b, beta).
    """
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    # weighted centering
    wsum = w.sum() + eps
    xbar = (w[:, None] * X).sum(axis=0) / wsum
    ybar = (w * y).sum() / wsum
    Xc = X - xbar
    yc = y - ybar

    # WLS via QR on sqrt-weighted, requires w >= 0
    rw = np.sqrt(np.maximum(w, 0.0))
    Xw = rw[:, None] * Xc
    yw = rw * yc

    Q, R = np.linalg.qr(Xw, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ yw)
    b = ybar - xbar @ beta
    return b, beta


class LaplaceRegr:
    def __init__(self, n_iter=200, tol=1e-6, eps=1e-6, verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.eps = eps
        self.verbose = verbose
        self.b = None
        self.w = None
        self.s = None

    def estimate(self, y, x, **kwargs):
        """
        y: (n,) or (n,1)
        x: (n,p)
        """
        y = np.asarray(y, dtype=float).ravel()
        x = np.asarray(x, dtype=float)
        n, p = x.shape
        assert y.size == n

        # init: OLS (robust starts also fine), s from abs residuals
        X1 = np.c_[np.ones(n), x]
        beta0, *_ = np.linalg.lstsq(X1, y, rcond=None)
        self.b = float(beta0[0])
        self.w = beta0[1:]
        r = y - (self.b + x @ self.w)
        self.s = np.mean(np.abs(r)) + self.eps  # avoid zero

        for t in range(self.n_iter):
            # E-step: weights w_i = 1 / (s * |r_i|)
            wts = 1.0 / (np.maximum(np.abs(r), self.eps) * max(self.s, self.eps))

            # M-step: weighted least squares for (b,w)
            new_b, new_w = wls_centered_intercept(x, y, wts)

            # Update residuals and s using the NEW params
            r = y - (new_b + x @ new_w)
            new_s = np.mean(np.abs(r))

            # Convergence checks (relative changes)
            db = abs(new_b - self.b)
            dw = np.linalg.norm(new_w - self.w)
            denom = self.eps + abs(self.b) + np.linalg.norm(self.w)
            rel_param = (db + dw) / denom
            rel_s = abs(new_s - self.s) / (self.s + self.eps)

            if self.verbose:
                lad = np.sum(np.abs(r))
                print(f"iter {t+1:3d} | LAD {lad:.6f} | s {new_s:.6f} | dθ {rel_param:.3e} | ds {rel_s:.3e}")

            # commit
            self.b, self.w, self.s = float(new_b), new_w, float(new_s)

            if rel_param < self.tol and rel_s < self.tol:
                break

        return self

    def posterior_predictive(self, x, **kwargs):
        """
        Returns (mean, variance) under Laplace noise model.
        Var(eps) = 2 s^2
        """
        x = np.asarray(x, dtype=float)
        m = self.b + x @ self.w
        v = 2.0 * (self.s ** 2)
        return m, np.full(m.shape, v)

    def view(self, plot = False, **kwargs):
        print('** Laplace Regression **')
        print('Bias   :', self.b)
        print('Weights:', self.w)
        print('Scale s:', self.s)

        if plot:
            plt.title('Regression weights')
            plt.plot(np.hstack(([self.b], self.w)) , '.-')
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    n = 1000
    b = 0.5
    w = 0.1
    s = 0.25
    x = np.random.normal(0,1,n)
    y = b+w*x+np.random.laplace(0,s,n)
    plt.plot(x, y, '.')
    plt.show()
    regr = LaplaceRegr(1000)
    regr.estimate(y = y, x = x[:,None])
    regr.view()


