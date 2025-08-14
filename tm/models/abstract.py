from abc import ABC, abstractmethod

# Model class
# models must implement this
class Model(ABC):

    # then this can be overridden
    def view(self, **kwargs):
        pass

    @abstractmethod
    def estimate(self, y, x, z, t, msidx, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def get_weight(self, y, x, t, z, xq, zq, prev_w, *args, **kwargs):
        """Subclasses must implement this method"""
        # xq, zq must be seen as the last element in x, z!
        # need to implement this on the model!
        # also, the multisequence idx is not passed as the workflow 
        # takes care of it
        pass
