import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm

def msidx_to_table(msidx:np.ndarray):
    '''
    idx: numpy (n,) array like 0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,...
        with the indication where the subsequences are
        creates an array like
        [
        [0,3],
        [3,7],
        ...
        ]
        with the indexes where the subsequences start and end
    '''
    aux = msidx[1:] - msidx[:-1]
    aux = np.where(aux != 0)[0]
    aux += 1
    aux_left = np.hstack(([0], aux))
    aux_right = np.hstack((aux, [msidx.size]))
    return np.hstack((aux_left[:,None], aux_right[:,None]))

if __name__ == '__main__':
    msidx = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
    t = msidx_to_table(msidx)
    print(t)
    for i in range(t.shape[0]):
        print(msidx[t[i][0]:t[i][1]])
