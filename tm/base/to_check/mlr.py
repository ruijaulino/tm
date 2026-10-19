from tm.base import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import logsumexp

def wls_centered_intercept(X, y, w):
    """
    Weighted least squares with a centered intercept.
    Returns (b, beta, sigma2, xbar, ybar, n_eff).
    """
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    wsum = w.sum() + 1e-12
    xbar = (w[:, None] * X).sum(axis=0) / wsum
    ybar = (w * y).sum() / wsum
    Xc = X - xbar
    yc = y - ybar

    rw = np.sqrt(w)
    Xw = rw[:, None] * Xc
    yw = rw * yc

    # QR solve (stable)
    Q, R = np.linalg.qr(Xw, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ yw)

    b = ybar - xbar @ beta

    resid = y - (b + X @ beta)
    rss = (w * resid**2).sum()
    w2sum = (w**2).sum() + 1e-12
    n_eff = (wsum**2) / w2sum
    dof = max(n_eff - (X.shape[1] + 1), 1.0)
    sigma2 = rss / dof
    return b, beta, sigma2, xbar, ybar, n_eff


class MLR(BaseModel):
    """
    Mixture of linear regression experts viewed as the conditional of a Gaussian mixture on (x,y).

    Gate:  x ~ N(m_k, C_k)          (C_k full or diagonal)
    Expert: y | x ~ N(b_k + x^T w_k, sigma_k^2)  (intercept handled by centering)

    Parameters
    ----------
    n_experts : int
    cov_type  : {'full','diag'}  gate covariance type
    n_init_kmeans : int          kmeans restarts for init
    n_iter : int                 EM iterations cap
    tol : float                  stopping tolerance on sigma^2
    random_state : int|None
    jitter : float               PSD jitter for C_k
    ridge_wls : float            tiny ridge for WLS if needed
    """
    def __init__(self, n_experts=2, cov_type='full', n_init_kmeans=10,
                 n_iter=200, tol=1e-6, random_state=42, jitter=1e-8):
        assert n_experts > 1
        assert cov_type in ('full', 'diag')
        self.K = n_experts
        self.cov_type = cov_type
        self.n_init_kmeans = n_init_kmeans
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.jitter = jitter

        # learned params
        self.pis = None          # (K,)
        self.ms = None           # (K, d)
        self.Cs = None           # (K, d, d) if full; None if diag
        self.Ds = None           # (K, d)     if diag; None if full
        self.bs = None           # (K,)
        self.ws = None           # (K, d)
        self.sigma2s = None      # (K,)
        self.loglik_ = None

    def view(self, plot = False, **kwargs):
        print('** Mixture of Linear Regression Experts **')
        for k in range(self.K):
            print()
            print(f'  Expert {k+1}')
            print('z probability: ', self.pis[k])
            print('variance: ', self.sigma2s[k])
            print('x center: ', self.ms[k])
            print('x covariance')
            if self.cov_type == 'full':
                print(self.Cs[k])
            else:
                print(self.Ds[k])
            print('x scale')
            if self.cov_type == 'full':
                print(np.sqrt(np.diag(self.Cs[k])))
            else:
                print(np.sqrt(self.Ds[k]))                
            if plot:
                plt.plot(self.ws[k])
                plt.show()
            print('weights: ', self.ws[k])
            print('bias: ', self.bs[k])
        print()
        print()        
        
    # ---------- initialization via k-means on [X | y]
    def _initialize(self, x, y):
        n, d = x.shape
        Z = np.hstack([x, y.reshape(-1, 1)])
        km = KMeans(n_clusters=self.K, n_init=self.n_init_kmeans,
                    random_state=self.random_state)
        km.fit(Z)
        labels = km.labels_

        self.pis = np.zeros(self.K)
        self.ms = np.zeros((self.K, d))
        if self.cov_type == 'full':
            self.Cs = np.zeros((self.K, d, d))
            self.Ds = None
        else:
            self.Ds = np.zeros((self.K, d))
            self.Cs = None
        self.bs = np.zeros(self.K)
        self.ws = np.zeros((self.K, d))
        self.sigma2s = np.ones(self.K)

        for k in range(self.K):
            idx = (labels == k)
            nk = idx.sum()
            self.pis[k] = max(nk, 1) / n

            xk = x[idx] if nk > 0 else X
            yk = y[idx] if nk > 0 else y

            # gate mean
            mk = xk.mean(axis=0) if nk > 0 else x.mean(axis=0)
            self.ms[k] = mk

            # gate covariance
            xc = xk - mk
            if self.cov_type == 'full':
                Ck = (xc.T @ xc) / max(nk - 1, 1)
                Ck += self.jitter * np.eye(d)
                self.Cs[k] = Ck
            else:
                dk = (xc**2).sum(axis=0) / max(nk - 1, 1)
                dk = np.maximum(dk, self.jitter)
                self.Ds[k] = dk

            # expert init via centered WLS
            b, w, sig2, *_ = wls_centered_intercept(xk, yk, np.ones_like(yk))
            self.bs[k] = b
            self.ws[k] = w
            self.sigma2s[k] = float(sig2)

    # ---------- log gate density per point and component
    def _log_gate(self, x):
        n, d = x.shape
        logp = np.empty((n, self.K))
        if self.cov_type == 'full':
            for k in range(self.K):
                Ck = self.Cs[k] + self.jitter * np.eye(d)
                sign, logdet = np.linalg.slogdet(Ck)
                if sign <= 0:
                    Ck = Ck + 10 * self.jitter * np.eye(d)
                    sign, logdet = np.linalg.slogdet(Ck)
                xm = x - self.ms[k]
                # Solve Ck^{-1} * Xm^T and take rowwise dot
                sol = np.linalg.solve(Ck, xm.T).T
                qf = np.sum(xm * sol, axis=1)
                logp[:, k] = -0.5 * (qf + d * np.log(2 * np.pi) + logdet)
        else:
            for k in range(self.K):
                dk = np.maximum(self.Ds[k], self.jitter)
                xm = x - self.ms[k]
                qf = np.sum((xm**2) / dk[None, :], axis=1)
                logdet = np.sum(np.log(dk))
                logp[:, k] = -0.5 * (qf + d * np.log(2 * np.pi) + logdet)
        return logp

    # ---------- log expert density per point and component
    def _log_expert(self, x, y):
        n = x.shape[0]
        logp = np.empty((n, self.K))
        for k in range(self.K):
            mu = self.bs[k] + x @ self.ws[k]
            s2 = max(self.sigma2s[k], 1e-12)
            logp[:, k] = -0.5 * (((y - mu) ** 2) / s2 + np.log(2 * np.pi) + np.log(s2))
        return logp

    # ---------- one E-step
    def _e_step(self, x, y):
        log_prior = np.log(np.maximum(self.pis, 1e-16))[None, :]
        log_gate = self._log_gate(x)
        log_expert = self._log_expert(x, y)
        log_r = log_prior + log_gate + log_expert
        ll = logsumexp(log_r, axis=1)
        r = np.exp(log_r - ll[:, None])
        return r, ll.sum()

    # ---------- one M-step
    def _m_step(self, x, y, r):
        n, d = x.shape
        nk = r.sum(axis=0) + 1e-12
        self.pis = nk / n

        # Gates
        for k in range(self.K):
            rk = r[:, k]
            wsum = nk[k]
            mk = (rk[:, None] * x).sum(axis=0) / wsum
            self.ms[k] = mk
            xc = x - mk
            if self.cov_type == 'full':
                Ck = (xc.T * rk) @ xc / wsum
                Ck += self.jitter * np.eye(d)
                self.Cs[k] = Ck
            else:
                dk = (rk[:, None] * (xc**2)).sum(axis=0) / wsum
                dk = np.maximum(dk, self.jitter)
                self.Ds[k] = dk

        # Experts (centered-intercept WLS)
        for k in range(self.K):
            rk = r[:, k]
            b, w, sig2, *_ = wls_centered_intercept(x, y, rk)
            self.bs[k] = b
            self.ws[k] = w
            self.sigma2s[k] = float(max(sig2, 1e-12))

    # ---------- public API
    def estimate(self, y, x, **kwargs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = x.shape
        self._initialize(x, y)

        prev_sig2 = self.sigma2s.copy()
        for _ in range(self.n_iter):
            r, ll = self._e_step(x, y)
            self._m_step(x, y, r)
            if np.mean(np.abs(self.sigma2s - prev_sig2)) < self.tol:
                self.loglik_ = ll
                break
            prev_sig2 = self.sigma2s.copy()
        else:
            self.loglik_ = ll  # last

        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        # conditional gate weights p(z=k | x) (i.e., gating network at test time)
        log_prior = np.log(np.maximum(self.pis, 1e-16))[None, :]
        log_gate = self._log_gate(x)
        log_gamma = log_prior + log_gate
        return np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))


    def predict_var(self, X):
        """
        Predictive variance via law of total variance:
        Var[y|x] = sum_k gamma_k(x) [sigma_k^2 + (mu_k - sum_j gamma_j mu_j)^2]
        """
        X = np.asarray(X, dtype=float)
        gamma = self.predict_proba(X)
        mus = np.column_stack([self.bs[k] + X @ self.ws[k] for k in range(self.K)])
        mix_mean = (gamma * mus).sum(axis=1, keepdims=True)
        comp_var = np.array([self.sigma2s[k] for k in range(self.K)])[None, :]
        return (gamma * (comp_var + (mus - mix_mean)**2)).sum(axis=1)

    def posterior_predictive(self, x, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        x = np.asarray(x, dtype=float)
        gamma = self.predict_proba(x)
        means = np.column_stack([self.bs[k] + x @ self.ws[k] for k in range(self.K)])
        m = (gamma * means).sum(axis=1)
        comp_var = np.array([self.sigma2s[k] for k in range(self.K)])[None, :]
        v = (gamma * (comp_var + (means - m[:,None])**2)).sum(axis=1)
        return m, v