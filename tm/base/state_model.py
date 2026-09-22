"""Conditional state means and probability mixtures of state models."""
import copy
import numpy as np
from .base import BaseModel


def _vector(value, name, numeric=False):
    value = np.asarray(value, dtype=float if numeric else None)
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError(f'{name} must have shape (n,) or (n, 1)')
    if numeric or np.issubdtype(value.dtype, np.number):
        if not np.isfinite(value).all():
            raise ValueError(f'{name} must contain finite values')
    return value


class StateModel(BaseModel):
    """Estimate one mean and observation variance per discrete state.

    With use_var=True, positive relative variances precision-weight the mean.
    The returned variance is the unweighted mean squared deviation around that
    fitted mean, floored at var_floor; it is not uncertainty in the mean.
    Unseen, suppressed, and undersampled states return (0, 1).
    """

    def __init__(self, min_points=10, zero_states=None, use_var=False,
                 var_floor=1e-6):
        if isinstance(min_points, bool) or not isinstance(min_points, (int, np.integer)) or min_points < 1:
            raise ValueError('min_points must be a positive integer')
        if not np.isfinite(var_floor) or var_floor <= 0:
            raise ValueError('var_floor must be positive and finite')
        self.min_points = min_points
        self.zero_states = tuple(() if zero_states is None else zero_states)
        self.use_var = use_var
        self.var_floor = var_floor
        self.states_distribution = {}
        self.default_mean, self.default_var = 0, 1
        self.w_norm = self.p = 1

    def view(self, plot=False, **kwargs):
        print('** State Model **')
        for state, distribution in self.states_distribution.items():
            print(f'State z = {state}: {distribution}')

    def estimate(self, y, z, var=None, **kwargs):
        y, z = _vector(y, 'y', True), _vector(z, 'z')
        if len(y) != len(z):
            raise ValueError('y and z must have the same number of observations')
        variances = np.ones(len(y))
        if self.use_var and var is not None:
            variances = _vector(var, 'var', True)
            if len(variances) != len(y) or np.any(variances <= 0):
                raise ValueError('var must contain one positive variance per observation')
        distributions = {}
        for state in np.unique(z):
            values = y[z == state]
            m, v = self.default_mean, self.default_var
            if state not in self.zero_states and len(values) >= self.min_points:
                relative = variances[z == state]
                weights = relative.min() / relative
                m = float(np.sum(values * (weights / weights.sum())))
                v = max(float(np.mean((values - m) ** 2)), self.var_floor)
            distributions[state] = {'m': m, 'v': v}
        self.states_distribution = distributions
        return self

    def posterior_predictive(self, z, **kwargs):
        z = _vector(z, 'z')
        m = np.full(len(z), self.default_mean, dtype=float)
        v = np.full(len(z), self.default_var, dtype=float)
        for state in np.unique(z):
            if state not in self.zero_states and state in self.states_distribution:
                distribution = self.states_distribution[state]
                m[z == state], v[z == state] = distribution['m'], distribution['v']
        return m, v


class EnsembleStateModel(BaseModel):
    """Probability mixture: one StateModel per column of z.

    Predictive variance includes within-model variance and disagreement between
    model means, using the law of total variance.
    """

    def __init__(self, model_weights=None, min_points=10, use_var=False, base_models=None):
        if base_models is not None and (
            not isinstance(base_models, list) or
            not all(isinstance(model, StateModel) for model in base_models)
        ):
            raise ValueError('base_models must be a list of StateModel instances')
        self.base_model = StateModel(min_points=min_points, use_var=use_var)
        self.predefined_base_models = copy.deepcopy(base_models)
        self.base_models = None
        self.min_points = min_points
        self.model_weights = None if model_weights is None else np.array(model_weights, dtype=float, copy=True)

    @staticmethod
    def _states(z):
        z = np.asarray(z)
        if z.ndim != 2 or z.shape[1] == 0:
            raise ValueError('z must be a matrix with at least one state column')
        return z

    def view(self, plot=False, **kwargs):
        if self.base_models is None:
            raise RuntimeError('Fit the ensemble before viewing it')
        for i, model in enumerate(self.base_models):
            print('Model for state:', i)
            model.view(plot=plot, **kwargs)

    def estimate(self, y, z, var=None, **kwargs):
        z = self._states(z)
        p = z.shape[1]
        weights = np.ones(p) if self.model_weights is None else np.array(self.model_weights, dtype=float, copy=True)
        if weights.shape != (p,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError('model_weights must be finite, nonnegative, match state columns, and have positive total')
        weights /= weights.max()
        weights /= weights.sum()
        if self.predefined_base_models is not None and len(self.predefined_base_models) != p:
            raise ValueError('base_models must match the number of state columns')
        models = []
        for i in range(p):
            template = self.base_model if self.predefined_base_models is None else self.predefined_base_models[i]
            model = copy.deepcopy(template)
            model.estimate(y=y, z=z[:, i], var=var, **kwargs)
            models.append(model)
        self.base_models, self.model_weights_ = models, weights
        return self

    def posterior_predictive(self, z, **kwargs):
        if self.base_models is None:
            raise RuntimeError('Fit the ensemble before predicting')
        z = self._states(z)
        if z.shape[1] != len(self.base_models):
            raise ValueError('z must have the same number of columns as training data')
        predictions = [model.posterior_predictive(z=z[:, i], **kwargs)
                       for i, model in enumerate(self.base_models)]
        means = np.stack([prediction[0] for prediction in predictions])
        variances = np.stack([prediction[1] for prediction in predictions])
        mean = self.model_weights_ @ means
        variance = self.model_weights_ @ (variances + (means - mean) ** 2)
        return mean, variance


class ThresholdStateModel(BaseModel):
    """Convert one feature into states x > th and x <= th."""

    def __init__(self, min_points=10, th=0):
        if not np.isfinite(th):
            raise ValueError('th must be finite')
        self.th = th
        self.state_model = StateModel(min_points=min_points)

    def view(self, plot=False, **kwargs):
        self.state_model.view(plot=plot, **kwargs)

    def estimate(self, y, x, **kwargs):
        self.state_model.estimate(y, _vector(x, 'x', True) > self.th, **kwargs)
        return self

    def posterior_predictive(self, x, **kwargs):
        return self.state_model.posterior_predictive(_vector(x, 'x', True) > self.th, **kwargs)


toStateModel = ThresholdStateModel
