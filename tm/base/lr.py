import numpy as np
import matplotlib.pyplot as plt
from tm.base import BaseModel

class LinRegr(BaseModel):
    def __init__(self, intercept = True):
        self.intercept = intercept
        self.w = None
        self.v = 1

    def view(self, plot = False, **kwargs):
        print('** Linear Regression **')
        print('Weights: ', self.w)
        print('Variance: ', self.v)
        if plot:
            plt.title('Regression weights')
            plt.plot(self.w, '.-')
            plt.grid(True)
            plt.show()

    def estimate(self, y, x, **kwargs):
        '''
        y: numpy (n, ) array with targets
        x: numpy (n, p) array with features
        '''
        if y.ndim == 2:
            assert y.shape[1] == 1, "y must contain a single target"
            y = y[:, 0]
        assert x.ndim == 2, "x must be a matrix with the features!"
        assert y.size == x.shape[0], "y and x must have the same number of observations"
        n = y.size
        if self.intercept:
            x = np.hstack((np.ones((n, 1)), x))
        Q, R = np.linalg.qr(x)
        self.w = np.linalg.solve(R, Q.T @ y)
        self.v = np.var(y - x @ self.w)

    def posterior_predictive(self, x, **kwargs):
        '''
        x: numpy (m, p) array
        '''            
        assert x.ndim == 2, "x must be a matrix with the features!"
        n = x.shape[0]
        if self.intercept:
            x = np.hstack((np.ones((n, 1)), x))
        m = x @ self.w        
        return m, self.v*np.ones_like(m)


def dev():
    
    np.random.seed(0)

    y = np.random.normal(0,1,500)


    phi = 0.93
    n = y.size
    window = 100


    v0 = np.zeros_like(y)
    v1 = np.zeros_like(y)
    
    v1[:window] = np.var(y[:window])


    f = np.ones(window) / window
    v01 = predictive_rollvar(y, f)

    f = (1-phi)*np.power(phi, np.arange(window))
    #print(f)

    v11 = predictive_rollvar(y, f)
    v1[:window] = v11[:window]

    for i in range(window, y.size):
        v0[i] = np.mean(y[i-window:i]*y[i-window:i])
        v1[i] = (1-phi)*(y[i-1]**2) + phi*v1[i-1]

    v0[:window] = v0[window] 

    plt.plot(v0, color = 'b')
    plt.plot(v01, color = 'r')
    
    plt.plot(v1, color = 'g')
    plt.plot(v11, color = 'm')
    
    plt.show()




def test():
    np.random.seed(0)
    y = np.random.normal(0, 1, 1000)

    phi = 0.93
    n = y.size
    window = 50

    # No need to reverse — np.convolve handles it
    f = (1 - phi) * np.power(phi, np.arange(window))

    

    print('window size: ', )

    print('sum of f: ', np.sum(f))


    plt.plot(f)
    plt.title("Exponential filter")
    plt.show()

    v1 = rollvar(y, f)

    v2 = np.zeros(n)
    for i in range(1, n):
        v2[i] = (1 - phi) * (y[i] ** 2) + phi * v2[i - 1]

    plt.plot(v1, label='v1 (convolution)')
    plt.plot(v2, label='v2 (recursive)', linestyle='--')
    plt.legend()
    plt.title("Rolling Variance Comparison")
    plt.show()




if __name__ == '__main__':
    #test()
    #dev()
    #exit(0)
    
    n = 1000
    a=0.1
    b=0.5
    scale = 0.01
    x=np.random.normal(0,scale,n)
    y=a+b*x+np.random.normal(0,scale,n)
    linregr = LinRegr()
    linregr.estimate(y = y, x = x[:,None])
    linregr.view()
    print(linregr.posterior_predictive(y, x[:,None]))
    pass


