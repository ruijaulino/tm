"""Exponentially weighted rolling models and risk wrappers.

Variance/covariance helpers estimate uncentered moments, not centered variance.
Legacy class names remain aliases for compatibility.
"""
import numpy as np
from .base import BaseModel
from .state_model import StateModel, EnsembleStateModel
import numpy as np

def _filter(values, f):
    values, f = np.asarray(values, dtype=float), np.asarray(f, dtype=float)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError('values must be a finite array with a time axis')
    if f.ndim != 1 or f.size == 0 or not np.isfinite(f).all() or np.any(f < 0) or not np.any(f > 0):
        raise ValueError('filter must be finite, nonnegative and have positive mass')
    if values.size == 0:
        return np.zeros_like(values)
    f = f / f.max()
    f = f / f.sum()
    return np.apply_along_axis(lambda v: np.convolve(v, f, mode='full')[:len(v)], 0, values)


apply_filter_vector = apply_filter_matrix = apply_filter_tensor = _filter


def rollvar(y, f):
    """Filtered uncentered second moment, assuming a zero mean."""
    y = np.asarray(y, dtype=float)
    return _filter(y * y, f)


def rollmean(y, f):
    return _filter(y, f)


def rollcov(y, f):
    """Filtered uncentered cross moment, assuming a zero mean."""
    y = np.asarray(y, dtype=float)
    return _filter(np.einsum('ni,nj->nij', y, y), f)


def _shift(values, lag, initial):
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError('lag must be a nonnegative integer')
    result = np.empty_like(values)
    result[...] = initial
    shift = lag + 1
    if shift < len(values):
        result[shift:] = values[:-shift]
    return result


def predictive_rollvar(y, f, lag=0):
    return _shift(rollvar(y, f), lag, 1.0)


def predictive_rollcov(y, f, lag=0):
    c = rollcov(y, f)
    c = _shift(c, lag, np.eye(c.shape[1]))
    idx = np.arange(c.shape[1])
    c[:, idx, idx] = np.where(c[:, idx, idx] < 1e-10, 1.0, c[:, idx, idx])
    return c


def predictive_rollmean(y, f, lag=0):
    return _shift(rollmean(y, f), lag, 0.0)


def diagonalize_covs(cov):
    '''
    keeps only the diagonal part of every entry of cov (nXdXd)
    outputs (nXdXd) as well
    '''
    diag = np.diagonal(cov, axis1=1, axis2=2)      # (n, d)
    out = np.zeros_like(cov)
    idx = np.arange(cov.shape[1])
    out[:, idx, idx] = diag    
    return out

class RollingMean(BaseModel):
    def __init__(self, phi = 0.95, phi_frac_cover = 0.95, reversion = False, min_points = 10, side = None, use_var = False, lag = 0):
        self.phi = phi
        self.phi_frac_cover = min(phi_frac_cover, 0.9999)
        self.min_points = min_points
        self.reversion = reversion
        self.side = side
        self.lag = lag
        self.use_var = use_var

    def view(self, **kwargs):
        pass

    def estimate(self, **kwargs):
        pass

    def posterior_predictive(self, y, is_live = False, **kwargs):
        '''
        works for multivariate y...
        '''
        #if y.ndim == 2:
        #    assert y.shape[1] == 1, "y must contain a single target for RollingMean model"
        #    y = y[:, 0]          

        if y.size != 0:
            # create filter
            k_f = np.log(1-self.phi_frac_cover)/np.log(self.phi) - 1
            f = (1-self.phi)*np.power(self.phi, np.arange(int(k_f)+1))
            m = predictive_rollmean(y, f, lag = self.lag)
            if self.reversion:
                m*=-1
            if self.side is not None:
                m[m*self.side < 0] = 0
            # burn some observations
            if is_live and y.shape[0] < f.size:
                print('Data is not enough for live. Return zero weight...')
            m[:min(f.size,self.min_points)] = 0
            return m, np.ones_like(y)        
        else:
            return np.zeros_like(y), np.ones_like(y)



class RollingVariance(BaseModel):
    def __init__(self, 
                 base_model:BaseModel,
                 phi = 0.95,  
                 phi_frac_cover = 0.95,
                 min_points = 10,
                 lag = 0
                ):
        self.phi = phi
        self.phi_frac_cover = min(phi_frac_cover, 0.9999)
        self.min_points = min_points
        self.base_model = base_model
        self.lag = lag
        try:
            if self.base_model.min_points != self.min_points:
                print('Warning: setting min_points in RollingVariance different than in base_model')
        except:
            pass 
    
    def view(self, plot = False, **kwargs):
        self.base_model.view(plot = plot)

    def estimate(self, y = None, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''

        # compute roll variance for y
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target for a RollingVariance model"
            y = y[:, 0]  

        k_f = np.log(1-self.phi_frac_cover)/np.log(self.phi) - 1
        f = (1-self.phi)*np.power(self.phi, np.arange(int(k_f)+1))
        if y.size < f.size+1:
            var = np.ones_like(y)
        else:
            var = rollvar(y, f) 
            var[var==0] = 1e8
        self.base_model.estimate(y = y, x = x, t = t, z = z, msidx = msidx, v = var)

    def posterior_predictive(self, y = None, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target for a RollingVariance model"
            y = y[:, 0]          
        if y.size != 0:
            k_f = np.log(1-self.phi_frac_cover)/np.log(self.phi) - 1
            f = (1-self.phi)*np.power(self.phi, np.arange(int(k_f)+1))
            v = predictive_rollvar(y, f, lag = self.lag)
            v[v == 0] = 1e8
            m, _ = self.base_model.posterior_predictive(y = y[:,None], x = x, t = t, z = z, msidx = msidx, v = v)                
            # create filter
            # burn some observations
            if is_live and y.size < f.size+1:
                print('Data is not enough for live. Return zero weight...')
            # m[:f.size] = 0
            v[:min(f.size,self.min_points)] = 1

            return m, v
        else:
            return np.zeros_like(y), np.ones_like(y)



class RollingPrecisionWeightedMean(BaseModel):
    """Exponentially weighted mean with observation-specific inverse variance.

    ``phi`` discounts mean observations; ``var_phi`` discounts squared returns.
    For each observation s, v_s includes y_s (the post's covariance-first update).
    The filtered mean is sum(phi**age * y_s / v_s) / sum(phi**age / v_s).
    Both outputs are shifted by 1 + lag, so row t uses data through t-1-lag.
    Variances are uncentered second moments, assuming small expected returns.

    Accepts (n,) or (n, d), preserving shape and treating columns independently.
    Returns observation variance, not uncertainty in the estimated mean.
    ``use_var=False`` returns unit variances but still precision-weights the mean.
    ``var_floor`` is a positive floor in squared-return units.
    """

    def __init__(self, phi = 0.95, var_phi = 0.95, phi_frac_cover = 0.95,
                 reversion = False, min_points = 10, side = None,
                 use_var = True, lag = 0, var_floor = 1e-12):
        for name, value in (("phi", phi), ("var_phi", var_phi),
                            ("phi_frac_cover", phi_frac_cover)):
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, value in (("lag", lag), ("min_points", min_points)):
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not np.isfinite(var_floor) or var_floor <= 0:
            raise ValueError("var_floor must be finite and positive")
        self.phi = phi
        self.phi_frac_cover = phi_frac_cover
        self.reversion = reversion
        self.min_points = min_points
        self.side = side
        self.use_var = use_var
        self.lag = lag
        self.var_phi = var_phi
        self.var_floor = var_floor

    def estimate(self, **kwargs):
        pass

    def posterior_predictive(self, y, is_live = False, **kwargs):
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target for a RollingVariance model"
            y = y[:, 0]    

        y = np.asarray(y, dtype=float)
        if y.ndim not in (1, 2) or not np.all(np.isfinite(y)):
            raise ValueError("y must be a finite array with shape (n,) or (n, d)")
        m, v = np.zeros_like(y), np.ones_like(y)
        if y.size == 0:
            return m, v
        values = y[:, None] if y.ndim == 1 else y

        def make_filter(phi):
            k_f = np.log(1-self.phi_frac_cover) / np.log(phi) - 1
            return (1-phi) * np.power(phi, np.arange(max(1, int(k_f)+1)))

        f = make_filter(self.phi)
        vf = make_filter(self.var_phi)
        # Normalize startup mass so an incomplete window does not imply low risk.
        mass = apply_filter_matrix(np.ones((len(values), 1)), vf)
        variance = apply_filter_matrix(values * values, vf) / mass
        variance = np.maximum(variance, self.var_floor)
        # The common floor multiplier cancels from the ratio and bounds weights.
        precision = self.var_floor / variance
        denominator = apply_filter_matrix(precision, f)
        filtered_mean = apply_filter_matrix(precision * values, f) / denominator

        shift = 1 + self.lag
        if shift < len(values):
            means = filtered_mean[:-shift]
            variances = variance[:-shift]
            m[shift:] = means[:, 0] if y.ndim == 1 else means
            if self.use_var:
                v[shift:] = variances[:, 0] if y.ndim == 1 else variances
        if self.reversion:
            m *= -1
        if self.side is not None:
            m[m*self.side < 0] = 0
        # Require min_points available observations, including any extra lag.
        burn = max(shift, self.min_points + self.lag)
        m[:burn] = 0
        v[:burn] = 1
        if is_live and len(values) < max(f.size, vf.size) + self.lag:
            print('Data is not enough for live. Return zero weight...')
            m[:] = 0

        return m, v




















class RollingInverseVolatility(BaseModel):
    def __init__(self, 
                 phi = 0.95,  
                 phi_frac_cover = 0.95,
                 min_points = 10,
                 lag = 0
                ):
        self.phi = phi
        self.phi_frac_cover = min(phi_frac_cover, 0.9999)
        self.min_points = min_points
        self.use_m2 = False
        self.lag = lag # lag to consider observations only up to self.lag days 
        self.mu = 0
        self.scale = 1e8

    def view(self, plot = False, **kwargs):
        pass

    def estimate(self, y = None, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''

        # compute roll variance for y
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target for a RollingVariance model"
            y = y[:, 0]  
        if y.size > 50:
            self.mu = np.abs(np.mean(y))
            self.scale = np.mean(y)



    def posterior_predictive(self, y = None, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target for a RollingVariance model"
            y = y[:, 0]          
        if y.size != 0:
            # create filter
            k_f = np.log(1-self.phi_frac_cover)/np.log(self.phi) - 1
            f = (1-self.phi)*np.power(self.phi, np.arange(int(k_f)+1))
            v = predictive_rollvar(y, f, lag = self.lag)
            scale = np.sqrt(v)
            scale[scale == 0] = 1e8
            # burn some observations
            if is_live and y.size < f.size:
                print('Data is not enough for live. Return zero weight...')
            # m[:f.size] = 0
            scale[:min(f.size,self.min_points)] = 1
            mu = np.ones_like(y)
            mu[:min(f.size,self.min_points)] = 0
            
            # return scale, scale*scale
            return mu, scale
            #return self.mu*scale/self.scale, scale*scale
        else:
            return np.zeros_like(y), np.ones_like(y)




class RollingCovariance(BaseModel):
    def __init__(self, 
                 base_model:BaseModel,
                 phi = 0.95,  
                 phi_frac_cover = 0.95,
                 min_points = 10,
                 lag = 0,
                 diagonalize = False,
                 reg_corr = 1

                ):
        self.phi = phi
        self.phi_frac_cover = min(phi_frac_cover, 0.9999)
        self.min_points = min_points
        self.base_model = base_model
        self.lag = lag
        self.diagonalize = diagonalize
        self.reg_corr = reg_corr

        try:
            if self.base_model.min_points != self.min_points:
                print('Warning: setting min_points in RollingVariance different than in base_model')
        except:
            pass 
    
    def view(self, plot = False, **kwargs):
        self.base_model.view(plot = plot)

    def estimate(self, y = None, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''
        self.base_model.estimate(y = y, x = x, t = t, z = z, msidx = msidx)

    def posterior_predictive(self, y = None, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if y.shape[0] != 0:
            m, _ = self.base_model.posterior_predictive(y = y, x = x, t = t, z = z, msidx = msidx) 
            
            # create filter
            k_f = np.log(1-self.phi_frac_cover)/np.log(self.phi) - 1
            f = (1-self.phi)*np.power(self.phi, np.arange(int(k_f)+1))
            cov = predictive_rollcov(y, f, lag = self.lag)
            if self.diagonalize:
                cov = diagonalize_covs(cov)
            # v[v == 0] = 1e8
            # burn some observations
            if is_live and y.size < f.size:
                print('Data is not enough for live. Return zero weight...')
            # m[:f.size] = 0
            cov[:min(f.size,self.min_points)] = np.eye(y.shape[1])

            std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))   # (n, d)
            std[std < 1e-10] = 1e-10
            scales = np.zeros_like(cov)
            idx = np.arange(cov.shape[1])
            scales[:, idx, idx] = std
            denom = std[:, :, None] * std[:, None, :]   # (n, d, d)
            R = cov / denom
            R *= self.reg_corr
            # make sure these entries are one
            R[:, idx, idx] = 1.0


            return m, scales@R@scales

        else:
            return np.zeros_like(y), np.repeat(np.eye(y.shape[1])[None, :, :], y.shape[0], axis=0)



class RollingInverseMultivariateVolatility(BaseModel):
    def __init__(self, 
                 phi = 0.95,  
                 phi_frac_cover = 0.95,
                 min_points = 10,
                 lag = 0,
                 diagonalize = False,
                 corr_mult = 1
                ):
        self.phi = phi
        self.phi_frac_cover = min(phi_frac_cover, 0.9999)
        self.min_points = min_points
        self.lag = lag
        self.diagonalize = diagonalize
        self.use_m2 = False
        self.corr_mult = corr_mult
    
    def view(self, plot = False, **kwargs):
        pass

    def estimate(self, y = None, x = None, t = None, z = None, msidx = None, **kwargs):   
        '''
        estimate without penalizing with varying variance...
        we can add that but maybe it's too much unjustified complexity        
        '''
        pass

    def posterior_predictive(self, y = None, x = None, t = None, z = None, msidx = None, is_live = False, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        if y.shape[0] != 0:
            
            # create filter
            k_f = np.log(1-self.phi_frac_cover)/np.log(self.phi) - 1
            f = (1-self.phi)*np.power(self.phi, np.arange(int(k_f)+1))
            cov = predictive_rollcov(y, f, lag = self.lag)
            # burn some observations
            if is_live and y.size < f.size:
                print('Data is not enough for live. Return zero weight...')
            # m[:f.size] = 0


            # print(np.diagonal(cov, axis1=1, axis2=2))
            # view of all diagonal entries: shape (n, p)
            # tmp = cov[:, np.arange(cov.shape[1]), np.arange(cov.shape[1])]
            # replace zeros with 1
            #tmp[tmp == 0] = 1
            #print(np.diagonal(cov, axis1=1, axis2=2))
            #print(sdfsdf)
            cov[:min(f.size,self.min_points)] = np.eye(y.shape[1])
            if self.diagonalize:
                cov = diagonalize_covs(cov)
            # extract correlation and scales

            std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))   # (n, d)
            scales = np.zeros_like(cov)
            idx = np.arange(cov.shape[1])
            scales[:, idx, idx] = std
            denom = std[:, :, None] * std[:, None, :]   # (n, d, d)
            # denom[denom == 0] = 1e8
            # cov[cov == 0] = 1e8
            R = cov / denom
            R *= self.corr_mult
            R[:, idx, idx] = 1.0
            # we should output SR -> when inverted gives S^{-1} R^{-1}
            return np.ones_like(y), scales@R

        else:
            return np.zeros_like(y), np.repeat(np.eye(y.shape[1])[None, :, :], y.shape[0], axis=0)





class RollingVarianceLinearRegression(RollingVariance):
    def __init__(self, phi = 0.95, phi_frac_cover = 0.95, intercept = True):
        from .to_check.lr import LinRegr
        super().__init__(LinRegr(intercept = intercept), phi = phi, phi_frac_cover = phi_frac_cover)

class RollingVarianceStateModel(RollingVariance):
    def __init__(self, phi = 0.95, phi_frac_cover = 0.95, min_points = 10, zero_states = None, use_var = False):
        super().__init__(StateModel(min_points = min_points, zero_states = zero_states, use_var = use_var), phi = phi, phi_frac_cover = phi_frac_cover, min_points = min_points)


class RollingVarianceEnsembleStateModel(RollingVariance):
    def __init__(self, phi = 0.95, phi_frac_cover = 0.95, min_points = 10, model_weights = None, use_var = False, base_models = None):
        super().__init__(EnsembleStateModel(model_weights = model_weights, min_points = min_points, use_var = use_var, base_models = base_models), phi = phi, phi_frac_cover = phi_frac_cover, min_points = min_points)

# Compatibility with existing callers and saved model references.
RollVarEnsembleStateModel = RollingVarianceEnsembleStateModel
RollVarStateModel = RollingVarianceStateModel
RollVarLinRegr = RollingVarianceLinearRegression
RollVarMean = RollingPrecisionWeightedMean
RollInvMultiVol = RollingInverseMultivariateVolatility
RollInvVol = RollingInverseVolatility
RollMean = RollingMean
RollCov = RollingCovariance
RollVar = RollingVariance

__all__ = ['RollingVarianceEnsembleStateModel', 'RollingVarianceStateModel', 'RollingVarianceLinearRegression', 'RollingPrecisionWeightedMean', 'RollingInverseMultivariateVolatility', 'RollingInverseVolatility', 'RollingMean', 'RollingCovariance', 'RollingVariance', 'RollVarEnsembleStateModel', 'RollVarStateModel', 'RollVarLinRegr', 'RollVarMean', 'RollInvMultiVol', 'RollInvVol', 'RollMean', 'RollCov', 'RollVar', 'apply_filter_vector', 'apply_filter_matrix', 'apply_filter_tensor', 'rollvar', 'rollmean', 'rollcov', 'predictive_rollvar', 'predictive_rollcov', 'predictive_rollmean', 'diagonalize_covs']
