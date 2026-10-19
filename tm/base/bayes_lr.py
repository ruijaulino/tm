import numpy as np
import warnings

class BayesianLinearRegression:
    """Bayesian regression with optional per-feature quantile clipping.

    clip_quantiles=None preserves the original behavior. A pair such as
    (0.01, 0.99) clips all features to their training quantiles, before
    centering and scaling. Quantiles are unweighted, computed along axis 0
    using NumPy's default linear interpolation. Targets are never clipped.
    The fitted clip_lower_ and clip_upper_ bounds are reused for both
    prediction methods and recomputed on every fit. Input arrays are not
    modified. Avoid clipping binary indicators or other categorical features.
    """
    def __init__(self, intercept=True, prior='ridge', n_iter=1000,
                 tol=1e-6, rho=0.5, standardize=True,
                 precision_bounds=(1e-12, 1e12), clip_quantiles=None):
        if prior not in ('ols', 'ridge', 'ard'):
            raise ValueError("prior must be 'ols', 'ridge', or 'ard'")
        if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
            raise ValueError('n_iter must be a positive integer')
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError('tol must be positive and finite')
        if not np.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError('rho must be in (0, 1]')
        lo, hi = map(float, precision_bounds)
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            raise ValueError('precision_bounds must satisfy 0 < lower < upper')
        self.intercept, self.standardize = bool(intercept), bool(standardize)
        self.prior, self.n_iter, self.tol, self.rho = prior, n_iter, tol, rho
        self.precision_bounds = (lo, hi)
        self.clip_quantiles = self._validate_clip_quantiles(clip_quantiles)
        self.clip_lower_ = self.clip_upper_ = None
        self.w = self.S = self.a = self.b = None

    @staticmethod
    def _validate_clip_quantiles(quantiles):
        if quantiles is None:
            return None
        try:
            quantiles = np.asarray(quantiles, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('clip_quantiles must be None or a pair with 0 <= lower < upper <= 1') from exc
        if (quantiles.shape != (2,) or not np.isfinite(quantiles).all()
                or not 0 <= quantiles[0] < quantiles[1] <= 1):
            raise ValueError('clip_quantiles must be None or a pair with 0 <= lower < upper <= 1')
        return tuple(float(q) for q in quantiles)

    @staticmethod
    def _matrix(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('x must be a finite two-dimensional array')
        return x

    @staticmethod
    def _vector(y, n, name):
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1 or y.size != n or not np.isfinite(y).all():
            raise ValueError(f'{name} must be a finite vector of length {n}')
        return y

    @classmethod
    def _variances(cls, v, n):
        v = np.ones(n) if v is None else cls._vector(v, n, 'v')
        if np.any(v <= 0):
            raise ValueError('v must contain strictly positive relative variances')
        return v

    @staticmethod
    def _ard_posterior(G, h, a, b):
        # Diagonal equilibration improves conditioning without adding a prior.
        K = b * G
        K.flat[::a.size + 1] += a
        d = np.sqrt(np.diag(K))
        B = K / d[:, None] / d[None, :]
        try:
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                'Posterior precision is numerically singular; remove nearly '
                'duplicate features or increase the lower precision bound.'
            ) from exc
        F = np.linalg.solve(L, np.eye(a.size)) / d[None, :]
        # S = F.T @ F, but only its diagonal and its action on h are needed.
        return b * (F.T @ (F @ h)), np.sum(F * F, axis=0), F

    def estimate(self, y, x, v=None, **kwargs):
        X = self._matrix(x)
        n, p = X.shape
        if n == 0 or p == 0:
            raise ValueError('x must have at least one row and one feature')
        y = self._vector(y, n, 'y')
        v = self._variances(v, n)
        n_eff = n - int(self.intercept)
        if n_eff <= 0:
            raise ValueError('Not enough observations to estimate noise')
        quantiles = self._validate_clip_quantiles(self.clip_quantiles)
        clip_lower = clip_upper = None
        if quantiles is not None:
            clip_lower, clip_upper = np.quantile(X, quantiles, axis=0)
            X = np.clip(X, clip_lower, clip_upper)
        q = 1.0 / v
        qsum = q.sum()
        mx = (q @ X) / qsum if self.intercept else np.zeros(p)
        my = float(q @ y / qsum) if self.intercept else 0.0
        xc, yc = X - mx, y - my
        scale = np.ones(p)
        if self.standardize:
            # Center only for estimating spread when fitting through the origin.
            dx = xc if self.intercept else X - (q @ X) / qsum
            scale = np.sqrt(q @ (dx * dx) / qsum)
            scale[scale == 0] = 1.0
        H = (xc / scale) * np.sqrt(q)[:, None]
        t = yc * np.sqrt(q)
        # Augment by t to obtain Q.T @ t and the orthogonal residual without
        # constructing the large Q matrix. The extra column costs one QR step.
        reduced = np.linalg.qr(np.column_stack((H, t)), mode='r')
        k = min(n, p)
        R, z = reduced[:k, :p], reduced[:k, p]
        rss_floor = float(reduced[k:, p] @ reduced[k:, p])
        del H
        tiny = np.finfo(float).tiny
        lo, hi = self.precision_bounds
        clip = lambda value: np.clip(value, lo, hi)
        self.converged_ = self.prior == 'ols'
        self.n_iter_ = 0

        if self.prior in ('ols', 'ridge'):
            U, s, Vt = np.linalg.svd(R, full_matrices=True)
            eigenvalues = np.zeros(p)
            eigenvalues[:s.size] = s * s
            projected = np.zeros(p)
            projected[:s.size] = s * (U[:, :s.size].T @ z)
            # Orthogonal residual decomposition avoids cancellation at perfect fits.
            uz = U.T @ z
            spectral_floor = rss_floor + float(uz[s.size:] @ uz[s.size:])

        if self.prior == 'ols':
            rank = int(np.count_nonzero(s > np.finfo(float).eps * max(n, p) * s[0]))
            if rank != p or n_eff <= p:
                raise ValueError('OLS requires full column rank and n > p + intercept')
            ws = Vt.T @ (projected / eigenvalues)
            r = z - R @ ws
            rss = rss_floor + float(r @ r)
            b = float(clip((n_eff - p) / max(rss, (n_eff - p) / hi, tiny)))
            Ss = (Vt.T * (1.0 / (b * eigenvalues))) @ Vt
            a = None
            self.em_a = None
            self.em_b = np.array([b])
            gamma_total = float(p)
        else:
            a = float(clip(1.0)) if self.prior == 'ridge' else np.full(p, clip(1.0))
            b = float(clip(n_eff / max(float(t @ t), n_eff / hi, tiny)))
            history_a, history_b = [np.copy(a)], [b]
            if self.prior == 'ard':
                G, h = R.T @ R, R.T @ z
            for iteration in range(1, self.n_iter + 1):
                if self.prior == 'ridge':
                    inv = 1.0 / (a + b * eigenvalues)
                    w_eigen = b * projected * inv
                    gamma_total = float(np.sum(b * eigenvalues * inv))
                    target_a = gamma_total / max(float(w_eigen @ w_eigen), gamma_total / hi, tiny)
                    r = uz[:s.size] - s * w_eigen[:s.size]
                    rss = spectral_floor + float(r @ r)
                else:
                    ws, diagS, F = self._ard_posterior(G, h, a, b)
                    gamma = np.clip(1.0 - a * diagS, 0.0, 1.0)
                    gamma_total = float(gamma.sum())
                    target_a = gamma / np.maximum.reduce([ws * ws, gamma / hi, np.full(p, tiny)])
                    r = z - R @ ws
                    rss = rss_floor + float(r @ r)
                noise_dof = max(n_eff - gamma_total, tiny)
                target_b = noise_dof / max(rss, noise_dof / hi, tiny)
                next_a = clip((1 - self.rho) * a + self.rho * clip(target_a))
                next_b = float(clip((1 - self.rho) * b + self.rho * clip(target_b)))
                delta = max(float(np.max(np.abs(np.log(next_a) - np.log(a)))),
                            abs(np.log(next_b) - np.log(b)))
                a, b = next_a, next_b
                history_a.append(np.copy(a))
                history_b.append(b)
                if delta < self.tol:
                    self.converged_ = True
                    break
            self.n_iter_ = iteration
            self.em_a, self.em_b = np.asarray(history_a), np.asarray(history_b)
            if not self.converged_:
                warnings.warn('Evidence updates reached n_iter without convergence',
                              RuntimeWarning, stacklevel=2)
            if self.prior == 'ridge':
                inv = 1.0 / (a + b * eigenvalues)
                ws = Vt.T @ (b * projected * inv)
                Ss = (Vt.T * inv) @ Vt
                gamma_total = float(np.sum(b * eigenvalues * inv))
            else:
                ws, diagS, F = self._ard_posterior(G, h, a, b)
                Ss = F.T @ F
                gamma_total = float(np.clip(1 - a * diagS, 0, 1).sum())

        self.clip_lower_, self.clip_upper_ = clip_lower, clip_upper
        self.n_features_in_ = p
        self.x_mean_, self.x_scale_, self.y_mean_ = mx, scale, my
        self.coef_ = ws / scale
        self.intercept_ = float(my - mx @ self.coef_) if self.intercept else 0.0
        slope_cov = Ss / scale[:, None] / scale[None, :]
        if self.intercept:
            cross = -(slope_cov @ mx)
            S = np.empty((p + 1, p + 1))
            S[1:, 1:] = slope_cov
            S[0, 1:] = S[1:, 0] = cross
            S[0, 0] = 1.0 / (b * qsum) - mx @ cross
            self.w = np.r_[self.intercept_, self.coef_]
        else:
            S = slope_cov
            self.w = self.coef_.copy()
        self.S, self.a, self.b = (S + S.T) * 0.5, a, b
        self._slope_cov = slope_cov
        self._intercept_var = 1.0 / (b * qsum) if self.intercept else 0.0
        self.effective_dof_ = gamma_total + int(self.intercept)
        return self

    def fit(self, x, y, v=None):
        return self.estimate(y, x, v)

    def _prediction_x(self, x):
        if self.w is None:
            raise RuntimeError('Fit the model before predicting')
        x = self._matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError('x has a different number of features than training data')
        if self.clip_lower_ is not None:
            x = np.clip(x, self.clip_lower_, self.clip_upper_)
        return x

    def predict(self, x, **kwargs):
        x = self._prediction_x(x)
        return (x - self.x_mean_) @ self.coef_ + self.y_mean_

    def posterior_predictive(self, x, v=None, **kwargs):
        x = self._prediction_x(x)
        v = self._variances(v, x.shape[0])
        dx = x - self.x_mean_
        mean = dx @ self.coef_ + self.y_mean_
        parameter_var = np.einsum('ij,ij->i', dx @ self._slope_cov, dx)
        var = v / self.b + self._intercept_var + np.maximum(parameter_var, 0.0)
        return mean, var

    def view(self, **kwargs):
        print(f'BayesianLinearRegression(prior={self.prior!r})')
        print('w:', self.w, '\nbeta:', self.b, '\nalpha:', self.a)



if __name__ == '__main__':

    pass