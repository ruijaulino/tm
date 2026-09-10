import numpy as np
import pandas as pd
from scipy.stats import norm
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


def block_bootstrap_sharpe(s, n_boot=1000, block_size=10):
    """
    Moving-block-bootstrap Sharpe samples without materializing the sampled paths.

    The sampled Sharpe only depends on the first two raw moments.  Instead of
    constructing an ``(n, n_boot)`` array of sampled returns, compute the sum and
    squared-sum of each sampled block from prefix sums.  This reduces both memory
    and work by roughly a factor of ``block_size``.
    """
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError("s must be 1D")

    n = s.size
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must satisfy 1 <= block_size <= len(s)")

    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    # Keep the same draw shape/distribution as the previous implementation.
    starts = np.random.randint(
        0, max_start + 1, size=(n_blocks, n_boot)
    )

    # All blocks are full length except the last one, which is truncated so the
    # bootstrap sample has exactly n observations.
    lengths = np.full(n_blocks, block_size, dtype=np.int64)
    lengths[-1] = n - (n_blocks - 1) * block_size
    ends = starts + lengths[:, None]

    # Prefix sums give O(1) sum and squared-sum for every sampled block.
    csum = np.empty(n + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(s, dtype=np.float64, out=csum[1:])

    csum2 = np.empty(n + 1, dtype=np.float64)
    csum2[0] = 0.0
    np.cumsum(np.square(s, dtype=np.float64), dtype=np.float64, out=csum2[1:])

    total = np.sum(csum[ends] - csum[starts], axis=0)
    total2 = np.sum(csum2[ends] - csum2[starts], axis=0)

    mu = total / n
    var = total2 / n - mu * mu
    # Protect against tiny negative values caused by floating-point cancellation.
    var = np.maximum(var, 0.0)
    sigma = np.sqrt(var)

    return mu / sigma







def valid_strategy(s, n_boot, sr_mult, alpha = 0.05, alpha_n = 1000, pct_fee = 0, block_size = 20, view = True):
    '''
    check if the paths represent a strategy with a positive
    sharpe ratio via bootstrap from the worst path
    s: numpy (n,k) array with strategy returns
    '''
    paths_sr = sr_mult*np.mean(s, axis = 0) / np.std(s, axis = 0)
    idx_lowest_sr = np.argmin(paths_sr)
    b_samples = block_bootstrap_sharpe(s[:,idx_lowest_sr], n_boot = n_boot, block_size = block_size)
    b_samples *= sr_mult
    valid = False
    #if np.sum(b_samples < 0) == 0:
    #    valid = True
    
    mean_b = np.mean(b_samples)
    scale_b = np.std(b_samples)
    
    # bonferroni
    valid = norm.cdf(0, loc=mean_b, scale=scale_b) < alpha/alpha_n
    
    if valid:
        txt=f'** ACCEPT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha/alpha_n}]**' 
        if view: print(txt)     
    else:
        txt=f'** REJECT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha/alpha_n}]**' 
        if view: print(txt)     
    
    #if s.shape[1]!=1:
    #    txt='Distribution of paths SHARPE' 
    #    if view:
    #        plt.title(txt)
    #        plt.hist(paths_sr,density=True)
    #        plt.grid(True)
    #        plt.show()
    if view:
        plt.title('(Worst path) SR bootstrap distribution')

        x_min, x_max = 0.5*np.min(b_samples), np.max(b_samples)*1.5
        x_pdf = np.linspace(x_min, x_max, 250)
        pdf = np.exp(-0.5*np.power((x_pdf-mean_b)/scale_b,2))/np.sqrt(2*np.pi*scale_b*scale_b)
        plt.hist(b_samples,density=True, label = 'Histogram')
        plt.plot(x_pdf, pdf, label = 'Gaussian Fit')
        plt.axvline(0)
        plt.grid(True)
        plt.legend()
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
            self.append(dataset)

    # add post process methods
    def post_process(self, pct_fee = 0., seq_fees = False, sr_mult = np.sqrt(250), n_boot = 1000, block_size = 20, alpha = 0.05, alpha_n = 1000, key = None, start_date = '', end_date = '', simple_view = False):

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
        
        if not simple_view:

            returns_distribution(s,pct_fee=pct_fee,bins=50)
            
            visualize_weights(w,ts)

        valid_strategy(s,n_boot,sr_mult,alpha = alpha, alpha_n = alpha_n,pct_fee=pct_fee, block_size = block_size)

        performance_summary(s,sr_mult,pct_fee=pct_fee)

        out = pd.DataFrame(s, index = ts)
        out.columns = [f'path_{i+1}' for i in range(len(out.columns))]
        return out

    def portfolio_post_process(self, pct_fee = 0., seq_fees = False, sr_mult = np.sqrt(250), n_boot = 1000, block_size = 20, alpha = 0.05, alpha_n = 1000, view_weights = True, use_pw = True, multiplier = 1, start_date = '', end_date = '', resample_to = 'B'):
        """
        Post-process a set of portfolio paths.

        The expensive part of the original implementation was repeatedly building and
        resampling pandas objects for s, pw and the full weight matrix.  Here we
        resample only an integer row-position Series, then use those positions to
        index the NumPy arrays directly.  This keeps the exact pandas resampling bins
        while avoiding conversion/resampling of the n x p weight matrix.

        Notes
        -----
        The fast resampling path assumes s, pw and w contain finite values.  This is
        the normal output of the backtest pipeline.  If NaNs are intentionally stored
        inside these arrays, pandas ``resample(...).last()`` has column-wise NaN
        semantics that are not identical to selecting the last physical row.
        """

        if len(self) == 0:
            print('No paths to process!')
            return

        keys = list(self[0].keys())
        if not isinstance(pct_fee, dict):
            pct_fee = {k: pct_fee for k in keys}

        paths_s = []
        paths_pw = []
        paths_leverage = []
        paths_net_leverage = []
        paths_n_datasets = []

        for dataset in self:
            parts = {}

            for key in keys:
                data = dataset[key]
                w = data.w
                n = data.n

                # Use pandas only to determine the output bins and the last source
                # row in each bin.  This is much cheaper than resampling w itself.
                index = pd.to_datetime(data.ts, unit='s')
                if resample_to is None:
                    out_index = index
                    valid = np.ones(n, dtype=bool)
                    pos = np.arange(n, dtype=np.intp)
                else:
                    pos_s = pd.Series(
                        np.arange(n, dtype=np.int64),
                        index=index,
                        copy=False,
                    ).resample(resample_to).last()
                    out_index = pos_s.index
                    valid = pos_s.notna().to_numpy()
                    pos = pos_s[valid].to_numpy(dtype=np.intp, copy=False)

                m = len(out_index)
                s_out = np.zeros(m, dtype=np.result_type(data.s, np.float64))
                pw_out = np.zeros(m, dtype=np.result_type(data.pw, np.float64))
                gross_out = np.zeros(m, dtype=np.float64)
                net_out = np.zeros(m, dtype=np.float64)

                if pos.size:
                    w_last = w[pos]
                    gross_last = np.sum(np.abs(w_last), axis=1)
                    net_last = np.sum(w_last, axis=1)

                    fee = pct_fee.get(key, 0)
                    if np.ndim(fee) != 0:
                        raise ValueError('pct_fee values must be scalars')

                    if fee == 0:
                        s_last = data.s[pos]
                    elif seq_fees:
                        # calculate_fees() is applied before resampling in the old
                        # implementation.  Therefore turnover at a retained row t is
                        # |w[t] - w[t-1]|, not the change from the previous retained row.
                        turnover = np.zeros(pos.size, dtype=np.float64)
                        has_prev = pos > 0
                        if np.any(has_prev):
                            p = pos[has_prev]
                            turnover[has_prev] = np.sum(
                                np.abs(w[p] - w[p - 1]), axis=1
                            )
                        s_last = data.s[pos] - fee * turnover
                    else:
                        s_last = data.s[pos] - fee * gross_last

                    if use_pw:
                        pw_last = data.pw[pos]
                    else:
                        pw_last = np.ones(pos.size, dtype=pw_out.dtype)

                    s_out[valid] = s_last
                    pw_out[valid] = pw_last
                    gross_out[valid] = gross_last
                    net_out[valid] = net_last

                parts[key] = pd.DataFrame(
                    {
                        's': s_out,
                        'pw': pw_out,
                        'gross': gross_out,
                        'net': net_out,
                    },
                    index=out_index,
                )

            # One alignment/concat per path instead of four separate concat passes.
            path = pd.concat(parts, axis=1)

            path_s = path.xs('s', level=1, axis=1).fillna(0)
            raw_pw = path.xs('pw', level=1, axis=1)
            path_gross = path.xs('gross', level=1, axis=1).fillna(0)
            path_net = path.xs('net', level=1, axis=1).fillna(0)

            # Vectorized replacement for DataFrame.apply(..., axis=1).
            non_zero_counts = raw_pw.fillna(0).ne(0).sum(axis=1)

            # Preserve the original behavior: forward-fill only values introduced
            # by alignment across datasets; missing resample bins were already zero.
            path_pw = raw_pw.ffill() * multiplier
            pw_values = path_pw.to_numpy(copy=False)

            # np.nansum matches pandas' row-wise sum(skipna=True) for leading NaNs.
            path_s_values = np.nansum(
                path_s.to_numpy(copy=False) * pw_values, axis=1
            )
            gross_values = np.nansum(
                path_gross.to_numpy(copy=False) * pw_values, axis=1
            )
            net_values = np.nansum(
                path_net.to_numpy(copy=False) * pw_values, axis=1
            )

            paths_s.append(pd.Series(path_s_values, index=path.index, name='s'))
            paths_pw.append(path_pw)
            paths_leverage.append(pd.Series(gross_values, index=path.index, name='s'))
            paths_net_leverage.append(pd.Series(net_values, index=path.index, name='s'))
            paths_n_datasets.append(pd.Series(non_zero_counts, index=path.index, name='n'))

        s = pd.concat(paths_s, axis=1)
        lev = pd.concat(paths_leverage, axis=1)
        net_lev = pd.concat(paths_net_leverage, axis=1)
        n_datasets = pd.concat(paths_n_datasets, axis=1)

        # Apply date filtering before constructing the potentially large 3D weight
        # array used only for visualization.
        mask = np.ones(len(s), dtype=bool)
        if start_date != '':
            mask &= s.index > pd.Timestamp(start_date)
        if end_date != '':
            mask &= s.index <= pd.Timestamp(end_date)

        if not np.all(mask):
            s = s.loc[mask]
            lev = lev.loc[mask]
            net_lev = net_lev.loc[mask]
            n_datasets = n_datasets.loc[mask]

        out = s.copy()
        out.columns = [f'path_{i+1}' for i in range(out.shape[1])]

        ts = s.index
        s_values = s.to_numpy(copy=False)

        equity_curve(s_values, ts, color='g', pct_fee=pct_fee)
        returns_distribution(s_values, pct_fee=pct_fee, bins=50)

        if view_weights:
            # Align each path to the final filtered index.  This also fixes the old
            # start_date mismatch between w and ts.
            w = np.stack(
                [pw.reindex(ts).to_numpy(copy=False) for pw in paths_pw],
                axis=2,
            )
            visualize_weights(w, ts, keys)

        lev.plot(legend=False, title='Paths Leverage')
        plt.grid(True)
        plt.show()

        net_lev.plot(legend=False, title='Paths Net Leverage')
        plt.grid(True)
        plt.show()

        n_datasets.plot(legend=False, title='Number of datasets')
        plt.grid(True)
        plt.show()

        valid_strategy(
            s_values,
            n_boot,
            sr_mult,
            alpha=alpha,
            alpha_n=alpha_n,
            pct_fee=pct_fee,
            block_size=block_size,
        )
        performance_summary(s_values, sr_mult, pct_fee=pct_fee)

        return out
