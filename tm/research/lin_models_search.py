'''
Fast simple linear models research
'''
import numpy as np
import matplotlib.pyplot as plt

def linreg(x, y, calc_s:bool = False, fee:float = 0., use_qr:bool = True, min_valid_points:int = 10):    
    '''
    Compute oos linear regression strategy results with LOOCV
    '''
    assert x.ndim == 1, "x must be a vector"
    assert y.ndim == 1, "y must be a vector"
    assert x.size == y.size, "x and y must have the same length"
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    n = x.size
    if n < min_valid_points: 
        return None, None, None
    X = np.column_stack((np.ones(n), x))
    if calc_s:
        if use_qr:
            # Compute b using QR decomposition
            Q, R = np.linalg.qr(X)
            b = np.linalg.solve(R, Q.T @ y)
            h = np.sum(Q**2, axis=1)
            w = (X @ b - y*h)/(1-h) # LOOCV weights
            s = y*w - fee*np.abs(w)
        else:
            tmp = np.linalg.pinv(X.T @ X) @ X.T
            b = tmp @ y
            h = np.diag(X @ tmp)
            w = (X @ b - y*h)/(1-h)
            s = y*w - fee*np.abs(w)
    else:
        b = np.linalg.pinv(X.T @ X) @ X.T @ y
        s = None
    return b, s, n

def linreg_oos_sharpe(x, y, fee, sharpe_mult = np.sqrt(260)):
    b, s, n = linreg(
        x = x, 
        y = y, 
        calc_s = True, 
        fee = fee
        )
    if b is not None:
        return sharpe_mult * np.mean(s) / np.std(s), n
    return -10000, 0
        

if __name__ == '__main__':
    

    n = 500
    x = np.random.normal(0, 1, n)
    y = 0.1*x + np.random.normal(0, 0.1, n)
    plt.plot(x, y, '.')
    plt.show()
    
    b, s, n = linreg(x, y, calc_s = True, fee = 0., use_qr = True, min_valid_points = 10)
    if s is not None:
        plt.plot(np.cumsum(s))
        plt.show()


    print(linreg_oos_sharpe(x, y, fee = 0, sharpe_mult = np.sqrt(260)))



    pass