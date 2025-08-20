import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from tm.containers import Data, Dataset
from tm.constants import *


def calculate_fees(s, weights, seq_fees, pct_fee):
    '''
    s: numpy (n,k) array with several return sequences
    seq_fees: bool with the indication that the fees are sequential
    pct_fees: scalar or array with the fees to be considered
    returns
    s_fees: numpy (n_fees,n,k) with the returns sequences for different fees
    ''' 
    if seq_fees:
        dw = np.abs(weights[1:] - weights[:-1])
        dw = np.vstack(([np.zeros_like(dw[0])], dw))
        dw = np.sum(dw, axis = 1)
    else:
        dw = np.sum(np.abs(weights), axis = 1)
    return s - pct_fee*dw


def equity_curve(s, ts, color = 'g', pct_fee = 0, title:str = 'Equity curve'):
    title = title
    s_df = pd.DataFrame(np.cumsum(s,axis = 0), index = ts)
    s_df.plot(color = color, title = title, legend = False) 
    plt.grid(True)
    plt.show()


def returns_distribution(s, pct_fee = 0, bins = 50):
    title='Strategy returns distribution'
    plt.title(title)
    plt.hist(s.ravel(), bins = bins, density = True)
    plt.grid(True)
    plt.show()          



def visualize_weights(w, ts, cols = None):
    aux = pd.DataFrame(np.sum(w, axis = 1), index = ts)
    aux.plot(title = 'Weights sum', legend = False)
    plt.grid(True)
    plt.show()

    aux = pd.DataFrame(np.sum(np.abs(w), axis = 1), index = ts)
    aux.plot(title = 'Total Leverage', legend = False)
    plt.grid(True)
    plt.show()

    p = w.shape[1]
    if p > 1:
        for i in range(p):
            title='Weight for asset %s'%(i+1) if cols is None else 'Weight for '+cols[i]
            aux = pd.DataFrame(w[:,i,:], index = ts)
            aux.plot(title = title, legend = False)
            plt.grid(True)
            plt.show()  

def bootstrap_sharpe(s, n_boot = 1000):
    '''
    bootstrat samples of sharpe ratio for an array of returns s ~ (n,)
    '''
    l = s.size
    idx = np.arange(l, dtype = int)
    idx_ = np.random.choice(idx,(l, n_boot), replace = True)
    s_ = s[idx_]
    boot_samples = np.mean(s_, axis = 0) / np.std(s_, axis = 0)
    return boot_samples


def valid_strategy(s, n_boot, sr_mult, pct_fee = 0, view = True):
    '''
    check if the paths represent a strategy with a positive
    sharpe ratio via bootstrap from the worst path
    s: numpy (n,k) array with strategy returns
    '''
    paths_sr = sr_mult*np.mean(s, axis = 0) / np.std(s, axis = 0)
    idx_lowest_sr = np.argmin(paths_sr)
    b_samples = bootstrap_sharpe(s[:,idx_lowest_sr], n_boot = n_boot)
    b_samples *= sr_mult
    valid = False
    if np.sum(b_samples < 0) == 0:
        valid = True
    if valid:
        txt='** ACCEPT STRATEGY **' 
        if view: print(txt)     
    else:
        txt='** REJECT STRATEGY **' 
        if view: print(txt)     
    if s.shape[1]!=1:
        txt='Distribution of paths SHARPE' 
        if view:
            plt.title(txt)
            plt.hist(paths_sr,density=True)
            plt.grid(True)
            plt.show()
    if view:
        plt.title('(Worst path) SR bootstrap distribution')
        plt.hist(b_samples,density=True)
        plt.grid(True)
        plt.show() 
    return valid


def performance_summary(s, sr_mult, pct_fee = 0):
    print()
    txt='** PERFORMANCE SUMMARY **' 
    print(txt)
    print()
    print('Return: ', np.power(sr_mult, 2) * np.mean(s))
    print('Standard deviation: ', sr_mult * np.std(s))
    print('Sharpe: ', sr_mult * np.mean(s) / np.std(s))
    print()


# this is just a list of Datasets
# to be used as results from cvbt
class Paths(list):
    
    def add(self, dataset:Union[Data, Dataset]):
        
        # convert into a dataset to make the code easier!
        if isinstance(dataset, Data):
            tmp = Dataset()
            tmp.add('data', dataset)
            self.append(tmp)
        else:
            self.append(tmp)

    # add post process methods
    def post_process(self, pct_fee = 0., seq_fees = False, sr_mult = np.sqrt(250), n_boot = 1000, key = None, start_date = '', end_date = ''):

        if len(self) == 0:
            print('No paths to process!')
            return
        
        
        keys = list(self[0].keys())

        # by default use the results for the first dataframe used as input
        # this will work by default because, in general, there is only one
        key = key if key is not None else keys[0]
        
        print(f'Post process for key {key}')

        # get and joint results for key
        s=[]
        w=[]    

        for dataset in self:
            s.append(dataset[key].s[:,None])
            w.append(dataset[key].w)

        if len(s)==0:
            print('No results to process!')
            return
        
        ts = dataset[key].index()

        # stack arrays
        s = np.hstack(s)
        w = np.stack(w, axis = 2)
        s = calculate_fees(s, w, seq_fees, pct_fee)

        # post processing        
        equity_curve(s, ts, color = 'g', pct_fee = pct_fee)    
        
        returns_distribution(s,pct_fee=pct_fee,bins=50)
        
        visualize_weights(w,ts)

        valid_strategy(s,n_boot,sr_mult,pct_fee=pct_fee)

        performance_summary(s,sr_mult,pct_fee=pct_fee)


    def portfolio_post_process(self, pct_fee = 0., seq_fees = False, sr_mult = np.sqrt(250), n_boot = 1000, view_weights = True, use_pw = True, multiplier = 1, start_date = '', end_date = ''):
     
    
        if len(self) == 0:
            print('No paths to process!')
            return
        
        keys = list(self[0].keys())
        if not isinstance(pct_fee, dict):
            pct_fee = {k:pct_fee for k in keys}


        paths_s=[]
        paths_pw=[]
        paths_leverage=[]
        paths_net_leverage = []
        paths_n_datasets=[]

        paths_s_datasets_boot = []
        
        for dataset in self:

            path_s=[]
            path_pw=[]
            path_w_abs_sum=[] # to build leverage
            path_w_sum = []
            for key, data in dataset.items():
                s = data.s
                w = data.w
                pw = data.pw   
                ts = data.index()
                if not use_pw:
                    pw = np.ones_like(pw) 

                s = pd.DataFrame(calculate_fees(s[:,None], w[:,:,None], seq_fees, pct_fee.get(key, 0)), columns = [key], index=ts)
                path_s.append(s)
                path_pw.append(pd.DataFrame(pw[:,None], columns = [key], index = ts))
                w = pd.DataFrame(w, columns = data.w_cols, index = ts)
                path_w_abs_sum.append(pd.DataFrame(w.abs().sum(axis=1), columns=[key]))
                path_w_sum.append(pd.DataFrame(w.abs().sum(axis=1),columns = [key]))

            path_s = pd.concat(path_s, axis = 1)
            path_pw = pd.concat(path_pw, axis = 1)
            path_w_abs_sum = pd.concat(path_w_abs_sum, axis = 1)
            path_w_sum = pd.concat(path_w_sum, axis = 1)
            path_s.columns = keys
            path_pw.columns = keys
            path_w_abs_sum.columns = keys
            path_w_sum.columns = keys
            # fill na with zero
            path_s = path_s.fillna(0)
            path_count_non_zero = path_pw.copy(deep = True)
            path_count_non_zero = path_count_non_zero.fillna(0)
            
            path_pw = path_pw.ffill() #fillna(method = 'ffill')
            path_w_abs_sum = path_w_abs_sum.fillna(0)
            path_w_sum = path_w_sum.fillna(0)

            path_pw /= np.sum(np.abs(path_pw), axis = 1).values[:,None]
            path_pw *= multiplier
            non_zero_counts = path_count_non_zero.apply(lambda row: (row != 0).sum(), axis=1)
        
            path_s = pd.DataFrame(np.sum(path_s*path_pw, axis = 1), columns=['s'])

            paths_s.append(path_s)
            paths_pw.append(path_pw)
            paths_net_leverage.append(pd.DataFrame(np.sum(path_w_sum*path_pw, axis = 1), columns=['s']))
            paths_leverage.append(pd.DataFrame(np.sum(path_w_abs_sum*path_pw, axis = 1), columns=['s']))
            paths_n_datasets.append(pd.DataFrame(non_zero_counts, columns = ['n']))
                
        
        s = pd.concat(paths_s, axis = 1)
        w = np.stack([e.values for e in paths_pw], axis = 2)
        lev = pd.concat(paths_leverage, axis = 1)
        net_lev = pd.concat(paths_net_leverage, axis = 1)
        n_datasets = pd.concat(paths_n_datasets, axis = 1)

        out = s.copy(deep = True)
        out.columns = [f'path_{i+1}' for i in range(len(out.columns))]

        ts=s.index
        s=s.values

        equity_curve(s, ts, color = 'g', pct_fee = pct_fee)

        returns_distribution(s, pct_fee = pct_fee, bins = 50)
        
        if view_weights: visualize_weights(w, ts, keys)
        lev.plot(legend = False, title = 'Paths Leverage')
        plt.grid(True)
        plt.show()
        
        net_lev.plot(legend = False, title = 'Paths Net Leverage')
        plt.grid(True)
        plt.show()


        n_datasets.plot(legend = False, title = 'Number of datasets')
        plt.grid(True)
        plt.show()  

        valid_strategy(s, n_boot, sr_mult, pct_fee = pct_fee)

        performance_summary(s, sr_mult, pct_fee = pct_fee)


