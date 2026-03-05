import numpy as np
import matplotlib.pyplot as plt
from tm.base import BaseModel
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

        # -----------------------
        # new version
        # pre calculations (no diag matrices)
        wgt = 1.0 / v                          # (n,)
        XT_Dinv_X = x.T @ (x * wgt[:, None])   # (p,p)
        XT_Dinv_y = x.T @ (y * wgt)            # (p,)

        eps = 1e-12

        for i in range(1, self.n_iter):
            if self.ard:
                A = np.diag(self.eb_a[i-1])
            else:
                A = self.eb_a[i-1] * I

            b = self.eb_b[i-1]
            K = A + b * XT_Dinv_X

            # posterior mean
            wn = np.linalg.solve(K, b * XT_Dinv_y)

            if self.ard:
                # need diag(Sn): compute Sn (could be optimized via Cholesky if needed)
                Sn = np.linalg.inv(K)
                gamma = 1.0 - self.eb_a[i-1] * np.diag(Sn)
                tmp = gamma / (wn * wn + eps)
                tmp = np.minimum(tmp, self.LARGE)

                # optionally: do not ARD-penalize intercept
                if self.intercept:
                    tmp[0] = self.eb_a[i-1, 0]

                self.eb_a[i] = tmp

                if compute_b:
                    res = y - x @ wn
                    sse_w = np.dot(res * wgt, res)   # weighted SSE = res.T D^{-1} res
                    self.eb_b[i] = (n - np.sum(gamma)) / max(sse_w, eps)
            else:
                tmp = p / max(np.dot(wn, wn), eps)
                self.eb_a[i] = min(tmp, self.LARGE)

                if compute_b:
                    res = y - x @ wn
                    sse_w = np.dot(res * wgt, res)
                    self.eb_b[i] = n / max(sse_w, eps)

            # convergence
            diff = np.abs(self.eb_a[i] - self.eb_a[i-1])
            d = diff if np.isscalar(diff) else diff.max()
            if d < self.tol:
                self.eb_a = self.eb_a[:i+1]
                self.eb_b = self.eb_b[:i+1]
                break
        # -----------------------

        # XT_Dinv_X = x.T @ np.diag(1/v) @ x
        # XT_Dinv_y = x.T @ np.diag(1/v) @ y            
            
        # # 
        # # Empirical Bayes
        # # if ard, then we have a different a for each variable
        # if self.ard:
        #     self.eb_a = np.zeros((self.n_iter, p))
        # else:
        #     self.eb_a = np.zeros(self.n_iter)
        
        # self.eb_b = np.ones(self.n_iter, dtype = np.float64)
        
        # self.eb_a[0] = self.a
        # self.eb_b[0] = self.b
                
        # for i in range(1, self.n_iter):
        #     if self.ard:
        #         A = np.diag(self.eb_a[i-1])
        #     else:
        #         A = self.eb_a[i-1]*I

        #     b = self.eb_b[i-1]
            
        #     #Sn = np.linalg.inv(A + b*XT_Dinv_X)
        #     #wn = b * Sn @ XT_Dinv_y

        #     K = A + b * XT_Dinv_X
        #     wn = np.linalg.solve(K, b * XT_Dinv_y)

            
        #     # update
        #     if self.ard:
        #         Sn = np.linalg.inv(K)  
        #         gamma = 1 - self.eb_a[i-1]*np.diag(Sn)

        #         tmp = gamma / (wn*wn + 1e-12)
        #         # control explosion in a for irrelevant features
        #         tmp = np.minimum(tmp, self.LARGE)

                
        #         #tmp = gamma / np.power(wn, 2)                
        #         #tmp[tmp>self.LARGE] = self.LARGE
                
        #         self.eb_a[i] = tmp            
        #         if compute_b:
        #             self.eb_b[i] = (n-np.sum(gamma)) / np.sum(np.power(y - x@wn,2))
        #     else:
        #         tmp = p / np.dot(wn, wn)
        #         self.eb_a[i] = min(tmp, self.LARGE)
        #         if compute_b:
        #             self.eb_b[i] = n / np.sum(np.power(y - x@wn,2))
        
        #     # evaluate convergence (use convergence in a!)
        #     d = np.mean(np.abs(self.eb_a[i] - self.eb_a[i-1]))
        #     # d = np.abs(self.eb_b[i]-self.eb_b[i-1])            
        #     if (d < self.tol):
        #         self.eb_a = self.eb_a[:i+1]
        #         self.eb_b = self.eb_b[:i+1]
        #         break
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
                 ard = True,  
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

    def post_estimate(self, y, x, **kwargs):
        if self.post_w_norm:
            m, v = self.core_regr.posterior_predictive(x)
            w = m / (v + m*m)
            self.w_norm = np.quantile(np.abs(w), self.w_quantile)





class BayesianLinearRegression:
    """
    Bayesian linear regression with evidence updates for:
      - OLS (flat-ish prior): w from weighted LS, beta from residuals, S for predictive variance
      - Ridge: scalar alpha, learned by evidence fixed-point updates
      - ARD: per-weight alpha_j, learned by evidence fixed-point updates

    Supports optional feature standardization (X only) and optional y-centering.

    Noise model with optional heteroskedastic *relative* scales v:
      y ~ N(Xw, (1/beta) * diag(v))

    Notes on scaling:
      - If standardize=True, training is performed in standardized feature space,
        but parameters (w, S) are unscaled back to ORIGINAL feature units before storing.
      - Therefore, posterior_predictive/predict expect x in ORIGINAL units (unstandardized).
    """

    def __init__(self,
                 intercept: bool = True,
                 prior: str = 'ridge',                 # 'ols', 'ridge', 'ard'
                 n_iter: int = 5000,
                 tol: float = 1e-6,
                 rho: float = 0.3,
                 jitter: float = 1e-10,
                 eps: float = 1e-12,
                 a_min: float = 1e-12,
                 a_max: float = 1e18,
                 b_min: float = 1e-12,
                 b_max: float = 1e12,
                 ard_intercept: bool = False,
                 standardize: bool = True,
                 center_y: bool = True,
                 max_jitter_tries: int = 1,
                 jitter_mult: float = 10.0,
                 adaptive_rho: bool = True,
                 rho_min: float = 0.01,
                 # minor additions
                 auto_center_y_if_no_intercept: bool = True,
                 store_standardized_posterior: bool = False):
        assert prior in ['ols', 'ridge', 'ard']
        self.prior = prior
        self.intercept = intercept
        self.n_iter = n_iter
        self.tol = tol
        self.rho = float(rho)
        self.jitter = float(jitter)
        self.eps = float(eps)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.b_min = float(b_min)
        self.b_max = float(b_max)
        self.ard_intercept = ard_intercept

        self.standardize = standardize
        self.center_y = center_y
        self.auto_center_y_if_no_intercept = auto_center_y_if_no_intercept

        self.max_jitter_tries = int(max_jitter_tries)
        self.jitter_mult = float(jitter_mult)
        self.adaptive_rho = adaptive_rho
        self.rho_min = float(rho_min)

        # learned (original units)
        self.w = None          # includes intercept if intercept=True
        self.S = None          # posterior covariance in original units (if computed)
        self.a = None          # alpha scalar (ridge) or vector (ard)
        self.b = None          # beta precision (original y-units; y is only centered, not scaled)

        # optional storage of standardized-space posterior (debugging)
        self.store_standardized_posterior = store_standardized_posterior
        self.w_s_ = None
        self.S_s_ = None

        # traces
        self.em_a = None
        self.em_b = None

        # scaling params
        self.x_mean_ = None
        self.x_scale_ = None
        self.y_mean_ = None

    def view(self):
        print("** BayesianLinearRegression **")
        print("w:", self.w)
        print("beta:", self.b)
        print("alpha:", self.a)

    @staticmethod
    def _diag_inv_from_cholesky(L: np.ndarray) -> np.ndarray:
        # diag(K^{-1}) where K = L L^T
        I = np.eye(L.shape[0], dtype=L.dtype)
        Linv = np.linalg.solve(L, I)
        return np.sum(Linv * Linv, axis=0)

    @staticmethod
    def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(L.T, np.linalg.solve(L, b))

    def _robust_cholesky(self, K: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Try Cholesky with increasing jitter until it succeeds.
        Returns (L, used_jitter).
        """
        I = np.eye(K.shape[0], dtype=K.dtype)
        base = self.jitter * (np.trace(K) / K.shape[0] + 1.0)
        jit = max(float(base), self.eps)
        for _ in range(self.max_jitter_tries):
            try:
                L = np.linalg.cholesky(K + jit * I)
                return L, jit
            except np.linalg.LinAlgError:
                jit = max(jit * self.jitter_mult, self.eps)
        # final try to surface the error if still failing
        L = np.linalg.cholesky(K + jit * I)
        return L, jit

    def _unscale_params(self, w_s: np.ndarray, S_s: np.ndarray | None):
        """
        Convert weights/covariance from standardized feature space back to original units.
        w_s includes intercept if intercept=True.
        """
        if not self.standardize:
            w = w_s.copy()
            S = None if S_s is None else S_s.copy()
        else:
            p = w_s.shape[0]

            if self.intercept:
                # Exact linear transform (without y_mean, which is affine):
                # w0_orig = w0_s - sum_j (mu_j/s_j) * wj_s
                # wj_orig = wj_s / s_j
                T = np.eye(p, dtype=np.float64)
                T[0, 1:] = -(self.x_mean_ / self.x_scale_)
                T[1:, 1:] = np.diag(1.0 / self.x_scale_)
                w = T @ w_s
                # Add y mean into intercept if we centered y
                w[0] += self.y_mean_
                S = None if S_s is None else (T @ S_s @ T.T)
            else:
                # No intercept term available. Just scale coefficients back.
                T = np.diag(1.0 / self.x_scale_)
                w = T @ w_s
                S = None if S_s is None else (T @ S_s @ T.T)
                # Can't bake y_mean_ without an intercept; handle in prediction.
        return w, S

    def _prepare_data(self, y: np.ndarray, X: np.ndarray, v: np.ndarray | None):
        # y -> (n,)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 2:
            assert y.shape[1] == 1
            y = y[:, 0]
        y = y.reshape(-1)

        X = np.asarray(X, dtype=np.float64)
        assert X.ndim == 2
        assert X.shape[0] == y.size

        n = y.size
        p0 = X.shape[1]

        # v -> (n,)
        if v is None:
            v = np.ones(n, dtype=np.float64)
        else:
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 2:
                assert v.shape[1] == 1
                v = v[:, 0]
            v = v.reshape(-1)
            assert v.size == n
            v = np.maximum(v, self.eps)

        # standardize X columns (caller should NOT include intercept in X)
        if self.standardize:
            self.x_mean_ = X.mean(axis=0)
            self.x_scale_ = X.std(axis=0)
            self.x_scale_ = np.maximum(self.x_scale_, self.eps)
            Xs = (X - self.x_mean_) / self.x_scale_
        else:
            self.x_mean_ = np.zeros(p0, dtype=np.float64)
            self.x_scale_ = np.ones(p0, dtype=np.float64)
            Xs = X

        # auto-center y if no intercept (recommended)
        center_y = self.center_y
        if (not self.intercept) and self.auto_center_y_if_no_intercept:
            center_y = True

        if center_y:
            self.y_mean_ = float(np.mean(y))
            ys = y - self.y_mean_
        else:
            self.y_mean_ = 0.0
            ys = y

        # add intercept column if requested
        if self.intercept:
            Xs = np.hstack([np.ones((n, 1), dtype=np.float64), Xs])

        return ys, Xs, v

    @staticmethod
    def _rel_log_delta(new, old, eps):
        """
        Relative change in log-space: |log(new)-log(old)|.
        Works well when values span many orders of magnitude.
        """
        new = np.asarray(new, dtype=np.float64)
        old = np.asarray(old, dtype=np.float64)
        return np.max(np.abs(np.log(new + eps) - np.log(old + eps)))

    def estimate(self, y: np.ndarray, x: np.ndarray, v: np.ndarray = None, **kwargs):
        ys, Xs, v = self._prepare_data(y, x, v)
        n = ys.size
        p = Xs.shape[1]
        I = np.eye(p, dtype=np.float64)

        # weights: W = diag(1/v)
        wgt = 1.0 / v

        # weighted normal equations parts
        XT_Dinv_X = Xs.T @ (Xs * wgt[:, None])   # (p,p)
        XT_Dinv_y = Xs.T @ (ys * wgt)            # (p,)

        # ---------- OLS ----------
        if self.prior == 'ols':
            K = XT_Dinv_X.copy()
            L, _ = self._robust_cholesky(K)
            w_s = self._chol_solve(L, XT_Dinv_y)

            res = ys - Xs @ w_s
            sse_w = float(np.dot(res * wgt, res))
            dof = max(n - p, 1)
            sigma2 = sse_w / max(dof, 1)
            b = float(np.clip(1.0 / max(sigma2, self.eps), self.b_min, self.b_max))

            # Posterior covariance in standardized space: S = (beta X^T W X)^-1
            Kp = b * XT_Dinv_X
            Lp, _ = self._robust_cholesky(Kp)
            Linv = np.linalg.solve(Lp, I)
            S_s = Linv.T @ Linv

            self.b = b
            self.a = None

            if self.store_standardized_posterior:
                self.w_s_ = w_s.copy()
                self.S_s_ = S_s.copy()

            self.w, self.S = self._unscale_params(w_s, S_s)
            return self

        # ---------- Ridge / ARD ----------
        # init beta from y variance
        y_var = float(np.var(ys)) + self.eps
        b0 = float(np.clip(1.0 / y_var, self.b_min, self.b_max))

        if self.prior == 'ridge':
            a0 = float(np.clip(1.0, self.a_min, self.a_max))
            self.em_a = np.zeros(self.n_iter, dtype=np.float64)
        else:
            a0 = np.ones(p, dtype=np.float64)
            if self.intercept and (not self.ard_intercept):
                a0[0] = 0.0  # truly unpenalized intercept
            self.em_a = np.zeros((self.n_iter, p), dtype=np.float64)

        self.em_b = np.zeros(self.n_iter, dtype=np.float64)
        self.em_a[0] = a0
        self.em_b[0] = b0

        rho = self.rho
        prev_metric = None
        unpenalize_intercept = (self.prior == 'ard' and self.intercept and (not self.ard_intercept))

        for i in range(1, self.n_iter):
            a_prev = self.em_a[i - 1]
            b_prev = float(self.em_b[i - 1])

            b_prev = float(np.clip(b_prev, self.b_min, self.b_max))

            if self.prior == 'ridge':
                a_prev = float(np.clip(a_prev, self.a_min, self.a_max))
                A_diag = a_prev * np.ones(p, dtype=np.float64)
            else:
                a_prev = a_prev.copy()
                if unpenalize_intercept:
                    a_prev[0] = 0.0
                    a_prev[1:] = np.clip(a_prev[1:], self.a_min, self.a_max)
                else:
                    a_prev = np.clip(a_prev, self.a_min, self.a_max)
                A_diag = a_prev

            # Build K = A + beta X^T W X ; avoid forming diag(...) (minor perf improvement)
            K = b_prev * XT_Dinv_X.copy()
            # add diagonal A (and eps for truly unpenalized dims)
            diag_add = A_diag + (A_diag == 0.0) * self.eps
            K.flat[::p + 1] += diag_add

            L, _ = self._robust_cholesky(K)
            w_s = self._chol_solve(L, b_prev * XT_Dinv_y)

            diagSn = self._diag_inv_from_cholesky(L)
            trSn = float(np.sum(diagSn))

            if self.prior == 'ard':
                # zero intercept gamma early to avoid confusing diagnostics
                gamma = 1.0 - a_prev * diagSn
                gamma = np.clip(gamma, 0.0, 1.0)
                if unpenalize_intercept:
                    gamma[0] = 0.0  # exclude intercept from ARD updates

                a_new = gamma / (w_s * w_s + self.eps)

                if unpenalize_intercept:
                    a_new[0] = 0.0
                    a_new[1:] = np.clip(a_new[1:], self.a_min, self.a_max)
                    a_next = a_prev.copy()
                    a_next[1:] = (1.0 - rho) * a_prev[1:] + rho * a_new[1:]
                    a_next[1:] = np.clip(a_next[1:], self.a_min, self.a_max)
                    a_next[0] = 0.0
                else:
                    a_new = np.clip(a_new, self.a_min, self.a_max)
                    a_next = (1.0 - rho) * a_prev + rho * a_new
                    a_next = np.clip(a_next, self.a_min, self.a_max)

                res = ys - Xs @ w_s
                sse_w = float(np.dot(res * wgt, res))
                eff = max(n - float(np.sum(gamma)), self.eps)
                b_new = eff / max(sse_w, self.eps)
                b_new = float(np.clip(b_new, self.b_min, self.b_max))
                b_next = (1.0 - rho) * b_prev + rho * b_new

                # log-space convergence (more stable across huge magnitudes)
                if unpenalize_intercept:
                    da = self._rel_log_delta(a_next[1:], a_prev[1:], self.eps)
                else:
                    da = self._rel_log_delta(a_next, a_prev, self.eps)
                db = self._rel_log_delta(b_next, b_prev, self.eps)

            else:
                # ridge
                gamma = float(p - float(a_prev) * trSn)
                gamma = float(np.clip(gamma, 0.0, float(p)))

                a_new = float(p) / max(float(np.dot(w_s, w_s)) + trSn, self.eps)
                a_new = float(np.clip(a_new, self.a_min, self.a_max))
                a_next = (1.0 - rho) * float(a_prev) + rho * a_new
                a_next = float(np.clip(a_next, self.a_min, self.a_max))

                res = ys - Xs @ w_s
                sse_w = float(np.dot(res * wgt, res))
                eff = max(n - gamma, self.eps)
                b_new = eff / max(sse_w, self.eps)
                b_new = float(np.clip(b_new, self.b_min, self.b_max))
                b_next = (1.0 - rho) * b_prev + rho * b_new

                da = self._rel_log_delta(a_next, a_prev, self.eps)
                db = self._rel_log_delta(b_next, b_prev, self.eps)

            # adaptive damping
            if self.adaptive_rho:
                metric = float(da + db)
                if prev_metric is not None and metric > 1.05 * prev_metric and rho > self.rho_min:
                    rho = max(self.rho_min, 0.5 * rho)
                prev_metric = metric

            self.em_a[i] = a_next
            self.em_b[i] = b_next

            if da < self.tol and db < self.tol:
                self.em_a = self.em_a[:i + 1]
                self.em_b = self.em_b[:i + 1]
                break

        if i == self.n_iter - 1:
            print("Evidence Approximation did not converge... "
                  f"(last log-da={da:.3e}, log-db={db:.3e})")

        # final posterior
        self.a = self.em_a[-1]
        self.b = float(self.em_b[-1])

        if self.prior == 'ridge':
            A_diag = float(self.a) * np.ones(p, dtype=np.float64)
        else:
            A_diag = self.a.copy()
            if unpenalize_intercept:
                A_diag[0] = 0.0
                A_diag[1:] = np.clip(A_diag[1:], self.a_min, self.a_max)
            else:
                A_diag = np.clip(A_diag, self.a_min, self.a_max)

        # final K, again without np.diag
        K = self.b * XT_Dinv_X.copy()
        diag_add = A_diag + (A_diag == 0.0) * self.eps
        K.flat[::p + 1] += diag_add

        L, _ = self._robust_cholesky(K)
        Linv = np.linalg.solve(L, I)
        S_s = Linv.T @ Linv
        w_s = self._chol_solve(L, self.b * XT_Dinv_y)

        if self.store_standardized_posterior:
            self.w_s_ = w_s.copy()
            self.S_s_ = S_s.copy()

        self.w, self.S = self._unscale_params(w_s, S_s)
        return self

    def posterior_predictive(self, x: np.ndarray, v: np.ndarray = None, **kwargs):
        """
        Posterior predictive for inputs x in ORIGINAL feature units.

        Returns:
          mean: (m,)
          var:  (m,) total predictive variance = noise + parameter uncertainty
        """
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2
        m = x.shape[0]

        if self.intercept:
            X = np.hstack([np.ones((m, 1), dtype=np.float64), x])
        else:
            X = x

        mean = X @ self.w

        # If we centered y but have no intercept, add mean back here
        if (not self.intercept) and self.center_y:
            mean = mean + self.y_mean_

        # parameter uncertainty term
        if self.S is not None:
            var_p = np.einsum('ij,jk,ik->i', X, self.S, X)
        else:
            var_p = 0.0

        # noise term
        if v is None:
            v = np.ones(m, dtype=np.float64)
        else:
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            assert v.size == m
            v = np.maximum(v, self.eps)

        var = (v / self.b) + var_p
        return mean, var







if __name__ == '__main__':

    pass