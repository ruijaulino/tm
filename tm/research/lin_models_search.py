'''
Fast simple linear models research
'''
import numpy as np
import matplotlib.pyplot as plt

def base_linreg_w(x, y, min_valid_points = 10):
    if x.ndim == 1: x = x[:,None]
    assert y.ndim == 1, "y must be a vector"
    assert x.shape[0] == y.size, "x and y must have the same length"

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_all = x.shape[0]
    
    w = np.zeros(n_all)
    valid = ~(np.isnan(x).any(axis = 1) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    n = x.shape[0]
    if n < min_valid_points: 
        return None
    X = np.hstack((np.ones(n)[:,None], x))
    # Compute b using QR decomposition
    Q, R = np.linalg.qr(X)
    b = np.linalg.solve(R, Q.T @ y)
    h = np.sum(Q**2, axis=1)
    w[valid] = (X @ b - y*h)/(1-h) # LOOCV weights
    return w

def linreg_ensemble(x, y, fee:float = 0, min_valid_points:int = 10):

    '''
    ensemble average of multiple individual models 
    '''

    if x.ndim == 1: x = x[:,None]
    assert y.ndim == 1, "y must be a vector"
    assert x.shape[0] == y.size, "x and y must have the same length"
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n, p = x.shape
    w = np.zeros_like(x)
    for j in range(p):
        w_ = base_linreg_w(x[:,j], y, min_valid_points)
        if w_ is not None: w[:,j] = w_
    y[np.isnan(y)] = 0
    w = np.mean(w, axis = 1)
    n_valid = (w!=0).sum() # proxy for valid predictions
    s = y*w - fee*np.abs(w)
    return s, n_valid


def linreg(x, y, calc_s:bool = False, fee:float = 0., use_qr:bool = True, min_valid_points:int = 10):    
    '''
    Compute oos linear regression strategy results with LOOCV
    '''
    # assert x.ndim == 1, "x must be a vector"
    if x.ndim == 1: x = x[:,None]
    assert y.ndim == 1, "y must be a vector"
    assert x.shape[0] == y.size, "x and y must have the same length"
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x).any(axis = 1) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    n = x.shape[0]
    if n < min_valid_points: 
        return None, None, None
    X = np.hstack((np.ones(n)[:,None], x))
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
    

def test_linreg_w():

    n = 500
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    y = 0.1*x1 - 0.5*x2 + np.random.normal(0, 0.1, n)
    plt.plot(x1, y, '.')
    plt.show()
    plt.plot(x2, y, '.')
    plt.show()
    
    x = np.hstack((x1[:,None], x2[:,None]))
    s, n = linreg_ensemble(x, y, fee = 0, min_valid_points = 10)

    if s is not None:
        plt.plot(np.cumsum(s))
        plt.show()

if __name__ == '__main__':


    test_linreg_w()
    exit(0)


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

    



    exit(0)

    # also works with matrice x

    n = 500
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    y = 0.1*x1 - 0.5*x2 + np.random.normal(0, 0.1, n)
    plt.plot(x1, y, '.')
    plt.show()
    plt.plot(x2, y, '.')
    plt.show()
    
    x = np.hstack((x1[:,None], x2[:,None]))

    b, s, n = linreg(x, y, calc_s = True, fee = 0., use_qr = True, min_valid_points = 10)
    if s is not None:
        plt.plot(np.cumsum(s))
        plt.show()


    print(linreg_oos_sharpe(x, y, fee = 0, sharpe_mult = np.sqrt(260)))



    pass