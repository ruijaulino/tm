from __future__ import annotations

from tm.containers import Dataset
from abc import ABC, abstractmethod
import numpy as np

from tm.workflows import cvbt_path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tm.model import ModelSet
    
class EnsembleModel(ABC):

    @abstractmethod
    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()
        pass

    @abstractmethod
    def get(self, key:str) -> float:
        """Subclasses must implement this method"""
        # returns the portfolio weight for a key
        pass

    def view(self, **kwargs):
        print("EnsembleModel")
        for k, v in self.pws.items():
            print(f'Portfolio Weight for {k} = {v}')



class IdleEnsembleModel(EnsembleModel):
    
    def __init__(self, v = 1, normalize:bool = True):
        self.v = 1
        self.normalize = normalize
        self.pws = {}

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()
        s = 0
        if self.normalize:
            s = self.v*len(dataset)
        if s == 0: s = 1        
        for k, _ in dataset.items():
            self.pws.update({k: self.v / s})

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)      



class ParametricEnsembleModel(EnsembleModel):
    
    def __init__(self, pws):
        self.pws = pws

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        pass

    def get(self, k:str) -> float:
        assert k in self.pws, f"ParametricEnsembleModel does not contain key {k}"
        return self.pws.get(k, 1)      






class InvVolEnsembleModel(EnsembleModel):
    
    def __init__(self):
        self.pws = {}

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()
        
        keys = []
        vols = []
        for k, data in dataset.items():
            if data.n > 5:
                keys.append(k)
                v = np.mean(np.std(data.y, axis = 0))
                if v == 0: v = 1e10
                vols.append(v)
            else:
                keys.append(k)
                vols.append(1e10)
        vols = 1 / np.array(vols)
        vols /= np.sum(vols)
        self.pws = dict(zip(keys, vols))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)        



class InvVolStratFilterEnsembleModel(EnsembleModel):
    def __init__(self, strat_filter_statistic = 'sharpe', k_folds:int = 4, seq_path:bool = False, burn_fraction:float = 0.1, min_burn_points:int = 3):
        self.strat_filter_statistic = strat_filter_statistic
        self.k_folds = k_folds
        self.seq_path = seq_path
        self.burn_fraction = burn_fraction
        self.min_burn_points = min_burn_points
        self.pws = {}

    def _compute_filter_stat(self, s):        
        if self.strat_filter_statistic == 'sharpe':
            scale = np.std(s)
            if scale != 0:
                return np.mean(s) / scale
            else:
                return 0            
        elif self.strat_filter_statistic == 'mean':
            return np.mean(s)
        elif self.strat_filter_statistic == 'gr':
            return np.mean(s) - 0.5*np.mean(np.power(s, 2))
        else:
            raise Exception('Unknown strat_filter_statistic')

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()

        # maybe copies not necessary
        dataset_ = cvbt_path(
                    dataset = dataset.copy(), 
                    modelset = model_set.copy(),
                    k_folds = self.k_folds, 
                    seq_path = self.seq_path, 
                    start_fold = 0, 
                    burn_fraction = self.burn_fraction, 
                    min_burn_points = self.min_burn_points
                    )
        keys = []
        f = []
        w = []
        for k, data in dataset_.items():
            if data.n > 5:
                keys.append(k)
                f.append(self._compute_filter_stat(data.s))
                w.append(1 / np.mean(np.std(data.y, axis = 0)))
            else:
                keys.append(k)
                f.append(-1)
                w.append(1 / 1e10)

        f = np.array(f)
        w = np.array(w)
        w[f<=0] = 0

        d = np.sum(w)        
        if d == 0: w = np.ones_like(w)       
        w /= np.sum(w)
        self.pws = dict(zip(keys, w))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)      



class EqWStratFilterEnsembleModel(EnsembleModel):
    def __init__(self, strat_filter_statistic = 'sharpe', k_folds:int = 4, seq_path:bool = False, burn_fraction:float = 0.1, min_burn_points:int = 3):
        self.strat_filter_statistic = strat_filter_statistic
        self.k_folds = k_folds
        self.seq_path = seq_path
        self.burn_fraction = burn_fraction
        self.min_burn_points = min_burn_points
        self.pws = {}

    def _compute_filter_stat(self, s):        
        if self.strat_filter_statistic == 'sharpe':
            scale = np.std(s)
            if scale != 0:
                return np.mean(s) / scale
            else:
                return 0            
        elif self.strat_filter_statistic == 'mean':
            return np.mean(s)
        elif self.strat_filter_statistic == 'gr':
            return np.mean(s) - 0.5*np.mean(np.power(s, 2))
        else:
            raise Exception('Unknown strat_filter_statistic')

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()

        # maybe copies not necessary
        dataset_ = cvbt_path(
                    dataset = dataset.copy(), 
                    modelset = model_set.copy(),
                    k_folds = self.k_folds, 
                    seq_path = self.seq_path, 
                    start_fold = 0, 
                    burn_fraction = self.burn_fraction, 
                    min_burn_points = self.min_burn_points
                    )
        keys = []
        f = []
        w = []
        for k, data in dataset_.items():
            if data.n > 5:
                keys.append(k)
                f.append(self._compute_filter_stat(data.s))
                w.append(1.)
            else:
                keys.append(k)
                f.append(-1)
                w.append(1.)

        f = np.array(f)
        w = np.array(w)
        w[f<=0] = 0

        d = np.sum(w)        
        if d == 0: w = np.ones_like(w)       
        w /= np.sum(w)
        self.pws = dict(zip(keys, w))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)      




class StratStatEnsembleModel(EnsembleModel):
    
    def __init__(self, statistic = 'invvol', k_folds:int = 4, seq_path:bool = False, burn_fraction:float = 0.1, min_burn_points:int = 3):
        self.statistic = statistic        
        self.k_folds = k_folds
        self.seq_path = seq_path
        self.burn_fraction = burn_fraction
        self.min_burn_points = min_burn_points
        self.pws = {}

    def _compute_stat(self, s):
        if self.statistic == 'invvol':
            scale = np.std(s)
            if scale != 0:
                return np.sign(np.mean(s)) / scale
            else:
                return 0
        
        elif self.statistic == 'sharpe':
            scale = np.std(s)
            if scale != 0:
                return np.mean(s) / scale
            else:
                return 0      

        elif self.statistic == 'sharpe2':
            scale = np.std(s)
            if scale != 0:
                return np.power(np.mean(s) / scale, 2)
            else:
                return 0      

        
        elif self.statistic == 'mean':
            return np.mean(s)

        elif self.statistic == 'kelly':
            v = np.mean(s*s)
            if v != 0:
                return np.mean(s) / v
            else:
                return 0     

        elif self.statistic == 'gr':
            return np.mean(s) - 0.5*np.mean(np.power(s, 2))

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()

        # maybe copies not necessary
        dataset_ = cvbt_path(
                    dataset = dataset.copy(), 
                    modelset = model_set.copy(),
                    k_folds = self.k_folds, 
                    seq_path = self.seq_path, 
                    start_fold = 0, 
                    burn_fraction = self.burn_fraction, 
                    min_burn_points = self.min_burn_points
                    )
        keys = []
        w = []
        for k, data in dataset_.items():
            keys.append(k)
            w.append(self._compute_stat(data.s))
        w = np.array(w)
        w[w <= 0] = 0
        d = np.sum(w)        
        if d == 0: w = np.ones_like(w)       
        w /= np.sum(w)
        self.pws = dict(zip(keys, w))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)        




class StratAllocEnsembleModel(EnsembleModel):
    
    def __init__(self, beta = 0.1, filter_mean:bool = True, auto_beta:bool = False, auto_beta_min_w_f:float = 0.1, k_folds:int = 4, seq_path:bool = False, burn_fraction:float = 0.1, min_burn_points:int = 3):
        self.beta = beta
        self.filter_mean = filter_mean
        self.auto_beta = auto_beta
        self.auto_beta_min_w_f = auto_beta_min_w_f
        self.auto_beta_min_w_f = min(max(0, self.auto_beta_min_w_f), 0.99)
        self.k_folds = k_folds
        self.seq_path = seq_path
        self.burn_fraction = burn_fraction
        self.min_burn_points = min_burn_points
        self.pws = {}

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()

        # maybe copies not necessary
        dataset_ = cvbt_path(
                    dataset = dataset.copy(), 
                    modelset = model_set.copy(),
                    k_folds = self.k_folds, 
                    seq_path = self.seq_path, 
                    start_fold = 0, 
                    burn_fraction = self.burn_fraction, 
                    min_burn_points = self.min_burn_points
                    )
        keys = []
        m = []
        scale = []
        for k, data in dataset_.items():
            keys.append(k)
            m.append(max(np.mean(data.s),0))
            scale.append(np.std(data.s))
        
        m = np.array(m)
        scale = np.array(scale)
        
        
        if self.filter_mean:
            valid = m>0
        else:
            valid = np.ones(m.size, dtype = bool)

        # if non are valid just put all to zero
        if np.sum(valid) == 0:
            w = np.zeros_like(m)
            self.pws = dict(zip(keys, w))    
        else:                
            w = np.ones_like(m)
            w[~valid] = 0
            w /= np.sum(w)        
            
            v = scale*scale

            if self.auto_beta:
                f = (m[valid]-np.mean(m[valid]) + (np.mean(v[valid])-v[valid])/w[valid].size)
                minf = np.min(f)
                if minf < 0:
                    self.beta = -1 * f.size * minf / (1 - self.auto_beta_min_w_f)

            w[valid] += (m[valid]-np.mean(m[valid]) + (np.mean(v[valid])-v[valid])/w[valid].size) / self.beta        
            w[w<0] = 0 # for security
            self.pws = dict(zip(keys, w))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1) 



