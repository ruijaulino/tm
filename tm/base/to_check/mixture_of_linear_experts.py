import numpy as np


# ============================================================
# General utilities
# ============================================================

def add_intercept(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


def logsumexp(a, axis=1, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(
        np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    )

    if not keepdims:
        out = np.squeeze(out, axis=axis)

    return out


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)

    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# ============================================================
# Weighted linear regression
# ============================================================

def weighted_ridge_regression(X, y, weights, ridge=1e-6):
    """
    Solve

        min_beta sum_i w_i (y_i - x_i' beta)^2
                 + ridge * ||beta||^2

    The intercept, assumed to be column 0, is not penalized.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0.0

    XTWX = X.T @ (weights[:, None] * X)
    XTWy = X.T @ (weights * y)

    return np.linalg.solve(
        XTWX + ridge * penalty,
        XTWy,
    )


# ============================================================
# Soft-label multinomial logistic regression
# ============================================================

def softmax_probabilities(X, alpha):
    """
    Parameters
    ----------
    X:
        Shape (n, p).

    alpha:
        Shape (K - 1, p).

    The final class is the reference class, with coefficients zero.
    """
    n = X.shape[0]

    logits = np.column_stack([
        X @ alpha.T,
        np.zeros(n),
    ])

    return softmax(logits)


def fit_softmax_newton(
    X,
    targets,
    alpha=None,
    ridge=1e-4,
    max_iter=10,
    tol=1e-7,
):
    """
    Fit multinomial logistic regression with soft targets.

    Parameters
    ----------
    X:
        Shape (n, p).

    targets:
        Shape (n, K). Each row sums to one.

    alpha:
        Initial coefficients, shape (K - 1, p).

    Returns
    -------
    alpha:
        Fitted coefficients for the non-reference classes.
    """
    n, p = X.shape
    K = targets.shape[1]
    q = K - 1

    if alpha is None:
        alpha = np.zeros((q, p))
    else:
        alpha = alpha.copy()

    penalty = np.ones((q, p))
    penalty[:, 0] = 0.0

    for _ in range(max_iter):
        pi = softmax_probabilities(X, alpha)
        pi_nonref = pi[:, :q]

        # Gradient of the negative log-likelihood
        gradient = (pi_nonref - targets[:, :q]).T @ X
        gradient += ridge * penalty * alpha

        gradient_flat = gradient.ravel()

        if np.linalg.norm(gradient_flat) < tol:
            break

        # Hessian of the negative log-likelihood
        hessian = np.zeros((q * p, q * p))

        for k in range(q):
            for j in range(q):
                if k == j:
                    weights = pi_nonref[:, k] * (
                        1.0 - pi_nonref[:, k]
                    )
                else:
                    weights = (
                        -pi_nonref[:, k]
                        * pi_nonref[:, j]
                    )

                block = X.T @ (weights[:, None] * X)

                row = slice(k * p, (k + 1) * p)
                col = slice(j * p, (j + 1) * p)

                hessian[row, col] = block

        # Ridge Hessian
        hessian += ridge * np.diag(penalty.ravel())

        step = np.linalg.solve(
            hessian + 1e-8 * np.eye(q * p),
            gradient_flat,
        )

        alpha -= step.reshape(q, p)

        if np.linalg.norm(step) < tol:
            break

    return alpha


# ============================================================
# Mixture of linear experts
# ============================================================

class MixtureOfLinearExperts:

    def __init__(
        self,
        n_experts=2,
        variance_mode="expert",
        expert_ridge=1e-8,
        gate_ridge=1e-8,
        variance_floor=1e-8,
        max_iter=100,
        gate_steps=5,
        tol=1e-6,
        random_state=None,
    ):
        '''
        variance_mode: shared, expert, fixed, 
        '''
        self.K = n_experts
        self.variance_mode = variance_mode

        self.expert_ridge = expert_ridge
        self.gate_ridge = gate_ridge
        self.variance_floor = variance_floor

        self.max_iter = max_iter
        self.gate_steps = gate_steps
        self.tol = tol

        self.random_state = random_state

    def _variance_matrix(self, n, observation_variance=None):
        """
        Return an (n, K) matrix of component variances.
        """
        if self.variance_mode == "fixed":
            return np.repeat(
                observation_variance[:, None],
                self.K,
                axis=1,
            )

        if self.variance_mode == "shared":
            return np.full(
                (n, self.K),
                self.shared_variance_,
            )

        if self.variance_mode == "expert":
            return np.broadcast_to(
                self.expert_variance_[None, :],
                (n, self.K),
            )

        raise ValueError("Unknown variance mode")

    def _component_log_probabilities(
        self,
        X,
        y,
        observation_variance=None,
    ):
        pi = softmax_probabilities(X, self.alpha_)
        means = X @ self.beta_.T

        variances = self._variance_matrix(
            len(y),
            observation_variance,
        )

        residuals = y[:, None] - means

        log_normal = (
            -0.5 * np.log(2.0 * np.pi * variances)
            -0.5 * residuals**2 / variances
        )

        return np.log(pi + 1e-300) + log_normal

    def _e_step(
        self,
        X,
        y,
        observation_variance=None,
    ):
        log_components = self._component_log_probabilities(
            X,
            y,
            observation_variance,
        )

        log_density = logsumexp(
            log_components,
            axis=1,
            keepdims=True,
        )

        responsibilities = np.exp(
            log_components - log_density
        )

        log_likelihood = log_density.sum()

        return responsibilities, log_likelihood

    def _update_experts(
        self,
        X,
        y,
        responsibilities,
        observation_variance=None,
    ):
        variances = self._variance_matrix(
            len(y),
            observation_variance,
        )

        for k in range(self.K):
            weights = (
                responsibilities[:, k]
                / variances[:, k]
            )

            self.beta_[k] = weighted_ridge_regression(
                X,
                y,
                weights,
                ridge=self.expert_ridge,
            )

    def _update_variance(
        self,
        X,
        y,
        responsibilities,
    ):
        if self.variance_mode == "fixed":
            return

        means = X @ self.beta_.T
        squared_errors = (y[:, None] - means) ** 2

        if self.variance_mode == "shared":
            self.shared_variance_ = np.sum(
                responsibilities * squared_errors
            ) / len(y)

            self.shared_variance_ = max(
                self.shared_variance_,
                self.variance_floor,
            )

        elif self.variance_mode == "expert":
            numerator = np.sum(
                responsibilities * squared_errors,
                axis=0,
            )

            denominator = np.sum(
                responsibilities,
                axis=0,
            )

            self.expert_variance_ = (
                numerator / denominator
            )

            self.expert_variance_ = np.maximum(
                self.expert_variance_,
                self.variance_floor,
            )

    def fit(
        self,
        X,
        y,
        observation_variance=None,
    ):
        X = add_intercept(X)
        y = np.asarray(y, dtype=float)

        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        if self.variance_mode == "fixed":
            observation_variance = np.asarray(
                observation_variance,
                dtype=float,
            )

        # Random initial responsibilities
        responsibilities = rng.random((n, self.K))
        responsibilities /= responsibilities.sum(
            axis=1,
            keepdims=True,
        )

        # Initial experts
        self.beta_ = np.zeros((self.K, p))

        for k in range(self.K):
            self.beta_[k] = weighted_ridge_regression(
                X,
                y,
                responsibilities[:, k],
                ridge=self.expert_ridge,
            )

        # Initial gate
        self.alpha_ = np.zeros((self.K - 1, p))

        # Initial variance
        initial_variance = max(
            np.var(y),
            self.variance_floor,
        )

        if self.variance_mode == "shared":
            self.shared_variance_ = initial_variance

        elif self.variance_mode == "expert":
            self.expert_variance_ = np.full(
                self.K,
                initial_variance,
            )

        # previous_ll = -np.inf
        self.log_likelihood_history_ = []
        previous_ll = None

        for iteration in range(self.max_iter):

            responsibilities, ll = self._e_step(
                X,
                y,
                observation_variance,
            )

            self._update_experts(
                X,
                y,
                responsibilities,
                observation_variance,
            )

            self.alpha_ = fit_softmax_newton(
                X=X,
                targets=responsibilities,
                alpha=self.alpha_,
                ridge=self.gate_ridge,
                max_iter=self.gate_steps,
            )

            self._update_variance(
                X,
                y,
                responsibilities,
            )

            # Likelihood under the newly updated parameters
            responsibilities, new_ll = self._e_step(
                X,
                y,
                observation_variance,
            )

            self.log_likelihood_history_.append(new_ll)

            if previous_ll is not None:
                relative_change = abs(
                    new_ll - previous_ll
                ) / (1.0 + abs(previous_ll))

                if relative_change < self.tol:
                    break

            previous_ll = new_ll

        self.n_iter_ = iteration + 1
        self.log_likelihood_ = new_ll
        self.responsibilities_ = responsibilities

        return self


    def gate_probabilities(self, X):
        X = add_intercept(X)
        return softmax_probabilities(X, self.alpha_)

    def expert_means(self, X):
        X = add_intercept(X)
        return X @ self.beta_.T

    def predict(self, X):
        pi = self.gate_probabilities(X)
        means = self.expert_means(X)

        return np.sum(pi * means, axis=1)
    
    def predict_mean_variance(
        self,
        X,
        observation_variance=None,
    ):
        X_raw = np.asarray(X, dtype=float)

        # gate_probabilities adds the intercept internally
        pi = self.gate_probabilities(X_raw)

        # Add it once here for the expert regressions
        X_design = add_intercept(X_raw)
        means = X_design @ self.beta_.T

        mean = np.sum(pi * means, axis=1)

        variances = self._variance_matrix(
            len(X_raw),
            observation_variance,
        )

        within_variance = np.sum(
            pi * variances,
            axis=1,
        )

        between_variance = np.sum(
            pi * (means - mean[:, None])**2,
            axis=1,
        )

        total_variance = within_variance + between_variance

        return mean, total_variance
    
    
    def posterior_probabilities(
        self,
        X,
        y,
        observation_variance=None,
    ):
        X = add_intercept(X)
        y = np.asarray(y)

        responsibilities, _ = self._e_step(
            X,
            y,
            observation_variance,
        )

        return responsibilities

    # to match API
    def view(self, **kwargs):
        pass

    def estimate(self, y, x, **kwargs):
            
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]
        assert x.ndim == 2, "x must be a matrix with the features!"
        assert y.size == x.shape[0], "y and x must have the same number of observations"

        self.fit(X = x, y = y)

    def posterior_predictive(self, x, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        assert x.ndim == 2, "x must be a matrix with the features!"
        return self.predict_mean_variance(X = x)




def test_core():
    rng = np.random.default_rng(123)

    n = 3000
    X = rng.normal(size=(n, 2))

    gate_probability = 1.0 / (
        1.0 + np.exp(
            -(0.5 + 1.5 * X[:, 0])
        )
    )

    z = rng.binomial(1, gate_probability)

    mean_0 = (
        1.0
        + 2.0 * X[:, 0]
        - 0.5 * X[:, 1]
    )

    mean_1 = (
        -1.0
        - 1.0 * X[:, 0]
        + 1.5 * X[:, 1]
    )

    y = np.where(
        z == 0,
        mean_0,
        mean_1,
    )

    y += rng.normal(scale=0.5, size=n)


    model = MixtureOfLinearExperts(
        n_experts=2,
        variance_mode="expert",
        max_iter=100,
        gate_steps=3,
        random_state=123,
    )

    model.fit(X, y)

    print("Expert coefficients")
    print(model.beta_)

    print("Gate coefficients")
    print(model.alpha_)

    print("variance")
    print(model.expert_variance_)

    print("Log-likelihood")
    print(model.log_likelihood_)



if __name__ == '__main__':
    pass

