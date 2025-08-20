from abc import ABC, abstractmethod

# BaseModel class
# probabilistic base models must implement this
class BaseModel(ABC):

    # then this can be overridden
    def view(self, **kwargs):
        pass

    @abstractmethod
    def estimate(self, y, x, z, t, msidx, **kwargs):
        """fit model parameters"""
        pass

    @abstractmethod
    def posterior_predictive(self, y, x, z, t, **kwargs):
        """posterior predictive distribution"""
        pass
