import numpy as np
from tm.base import BaseModel

class StateGaussian(BaseModel):
    def __init__(self, min_points = 10, zero_states = [], diagonal_cov = False):
        self.min_points = min_points
        self.states_distribution = {}
        self.zero_states = zero_states
        self.diagonal_cov = diagonal_cov
        self.default_mean = 0
        self.default_var = 1
        self.w_norm = 1
        self.p = 1

    def view(self, plot_hist = True):
        print('StateGaussian')
        for k, v in self.states_distribution.items():
            print(f"State z={k}")
            print(v)
            print()

    def estimate(self, y, z, **kwargs):
        
        assert isinstance(z, np.ndarray), "z must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"        
        assert z.ndim == 1, "z must be a vector"
        assert y.ndim == 2, "y must be a matrix"
        assert y.shape[0] == z.size, "y and z must have the same number of observations"

        n, self.p = y.shape        
        states = np.unique(z)
        for state in states:                        
            if state in self.zero_states:
                self.states_distribution.update({state: {'m':np.zeros(self.p), 'c':np.eye(self.p)}})
            else:
                idx = np.where(z == state)[0]
                if idx.size > self.min_points:
                    m = np.mean(y[idx], axis = 0)
                    c = np.atleast_2d(np.cov(y[idx].T))
                    m2 = c + np.outer(m, m)
                    if self.diagonal_cov:
                        c = np.diag(np.diag(c))
                    self.states_distribution.update({state: {'m':m, 'c':c}})
                else:
                    self.states_distribution.update({state: {'m':np.zeros(self.p), 'c':np.eye(self.p)}})
        
    def posterior_predictive(self, y=None, x=None, t=None, z=None, msidx=None, is_live=False, **kwargs):
        """
        Returns:
            mean: (n, p)
            cov:  (n, p, p)
        """

        n = z.shape[0]

        if z is None:
            mean = np.full((n, self.p), self.default_mean)
            cov = np.repeat((self.default_var * np.eye(self.p))[None, :, :], n, axis=0)
            return mean, cov

        assert isinstance(z, np.ndarray), "z must be a numpy array"
        assert z.ndim == 1, "z must be a vector"
        assert z.size == n, "y and z must have the same number of observations"

        mean = np.zeros((n, self.p))
        cov = np.zeros((n, self.p, self.p))

        default_cov = self.default_var * np.eye(self.p)

        for state in np.unique(z):
            idx = np.where(z == state)[0]

            dist = self.states_distribution.get(
                state,
                {
                    "m": np.full(self.p, self.default_mean),
                    "c": default_cov
                }
            )

            m = dist["m"]
            c = dist["c"]

            if self.diagonal_cov:
                c = np.diag(np.diag(c))

            mean[idx] = m
            cov[idx] = c

        return mean, cov


if __name__ == '__main__':
    y = np.random.normal(0,0.01,((50,2)))
    z = np.random.choice([0,1], size = 50)
    model = StateGaussian(diagonal_cov = True)

    model.estimate(y =  y, z = z)
    model.view()
    print(model.posterior_predictive(z = np.array([0,0, 1])))




