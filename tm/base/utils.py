import numpy as np

def rollvar(y, f):
    ysq = y * y
    # Ensure f is normalized to sum to 1
    f = f / f.sum()
    # np.convolve flips filter internally
    v = np.convolve(ysq, f, mode='full')[:len(ysq)]
    return v

def predictive_rollvar(y, f):
    v = rollvar(y, f)
    v = np.hstack((v[0], v[:-1]))
    return v
