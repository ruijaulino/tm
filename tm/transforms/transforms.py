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

if __name__ == '__main__':
    pass