import numpy as np
from tm.transforms import Transform



# Transform class
class ScaleTransform(Transform):
    
    def __init__(self, demean = False):
        self.demean = demean
        self.scale = 1
        self.mean = 0

    def view(self):
        print('** Scale Transform **')
        print('scale: ', self.scale)
        if self.demean:
            print('mean: ', self.mean)

    def cost_scale(self):
        return self.scale

    def estimate(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""        
        if arr.shape[0] != 0:
            self.scale = np.std(arr, axis = 0)
            if self.demean:
                self.mean = np.mean(arr, axis = 0)
            else:
                self.mean = self.mean*np.ones_like(self.scale)

    def transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        if self.demean:
            return (arr - self.mean) / self.scale
        else:
            return arr / self.scale
    
    def inverse_transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        if self.demean:
            return arr * self.scale + self.mean
        else:
            return arr * self.scale
    
    def scale_back_moments(self, mu, cov):
        """Subclasses must implement this method"""
        mu = mu * self.scale + self.mean
        cov = cov * self.scale[:, None] * self.scale[None, :]
        return mu, cov


# Transform class
class DemeanTransform(Transform):
    
    def __init__(self):
        self.mean = 0

    def view(self):
        print('** Demean Transform **')
        print('mean: ', self.mean)

    def cost_scale(self):
        return 1

    def estimate(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""        
        if arr.shape[0] != 0:
            self.mean = np.mean(arr, axis = 0)

    def transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        return arr - self.mean
    
    def inverse_transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        return arr + self.mean

    def scale_back_moments(self, mu, cov):
        """Subclasses must implement this method"""
        mu = mu + self.mean
        return mu, cov        

if __name__ == '__main__':
    pass