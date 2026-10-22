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
        dw = np.abs(np.diff(weights, axis=0, prepend=np.zeros_like(weights[:1])))
        dw = np.sum(dw, axis = 1)
    else:
        dw = np.sum(np.abs(weights), axis = 1)
    return s - pct_fee*dw


def equity_curve(s, ts, color = 'g', pct_fee = 0, title:str = 'Equity curve', *, return_kind='additive'):
    title = title
    if return_kind not in ('additive', 'simple'):
        raise ValueError("return_kind must be 'additive' or 'simple'")
    values = pd.DataFrame(s, index=ts)
    s_df = values.cumsum() if return_kind == 'additive' else (1. + values).cumprod() - 1.
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
    Vectorized moving block bootstrap Sharpe samples for a 1D return series.

    Parameters
    ----------
    s : array-like, shape (n,)
        1D array of returns.
    n_boot : int
        Number of bootstrap samples.
    block_size : int
        Length of each block.
    ddof : int
        Degrees of freedom for std.

    Returns
    -------
    boot_samples : np.ndarray, shape (n_boot,)
        Bootstrap Sharpe samples.
    """
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError("s must be 1D")

    n = s.size
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must satisfy 1 <= block_size <= len(s)")

    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    # shape: (n_blocks, n_boot)
    starts = np.random.randint(0, max_start + 1, size=(n_blocks, n_boot))

    # shape: (block_size,)
    offsets = np.arange(block_size)

    # shape: (n_blocks, n_boot, block_size)
    idx = starts[..., None] + offsets
    
    # flatten first and last axes -> total sampled length >= n
    # shape after reshape: (n_blocks*block_size, n_boot)
    idx = idx.transpose(0, 2, 1).reshape(n_blocks * block_size, n_boot)

    # truncate to length n
    idx = idx[:n, :]

    # sampled returns, shape: (n, n_boot)
    s_boot = s[idx]

    mu = np.mean(s_boot, axis=0)
    sigma = np.std(s_boot, axis=0)

    boot_samples = mu / sigma
    
    return boot_samples







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
    def post_process(self, pct_fee = 0., seq_fees = False, sr_mult = np.sqrt(250), n_boot = 1000, block_size = 20, alpha = 0.05, alpha_n = 1000, key = None, start_date = '', end_date = '', simple_view = False, *, view=True, return_kind='additive'):

        if len(self) == 0:
            print('No paths to process!')
            return
        
        
        keys = list(self[0].keys())

        # by default use the results for the first dataframe used as input
        # this will work by default because, in general, there is only one
        key = key if key is not None else keys[0]
        
        if view: print(f'Post process for key {key}')

        # get and joint results for key
        s=[]
        w=[]    

        for dataset in self:
            s.append(dataset[key].s[:,None])
            w.append(dataset[key].w)

        if len(s)==0:
            print('No results to process!')
            return
        
        ts = pd.DatetimeIndex(self[0][key].index())
        if any(not pd.DatetimeIndex(d[key].index()).equals(ts) for d in self):
            raise ValueError('All paths must have matching timestamps')

        # stack arrays
        s = np.hstack(s)
        w = np.stack(w, axis = 2)
        s = calculate_fees(s, w, seq_fees, pct_fee)

        mask = np.ones(len(ts), dtype=bool)
        if start_date not in ('', None):
            mask &= ts > pd.Timestamp(start_date)
        if end_date not in ('', None):
            mask &= ts <= pd.Timestamp(end_date)
        if not mask.any():
            raise ValueError('No observations in the requested window')
        s, w, ts = s[mask], w[mask], ts[mask]

        # post processing
        if view: equity_curve(s, ts, color = 'g', pct_fee = pct_fee, return_kind=return_kind)    
        
        if view and not simple_view:

            returns_distribution(s,pct_fee=pct_fee,bins=50)
            
            visualize_weights(w,ts)

        if n_boot: valid_strategy(s,n_boot,sr_mult,alpha = alpha, alpha_n = alpha_n,pct_fee=pct_fee, block_size = block_size, view=view)

        if view: performance_summary(s,sr_mult,pct_fee=pct_fee)

        out = pd.DataFrame(s, index = ts)
        out.columns = [f'path_{i+1}' for i in range(len(out.columns))]
        return out

    def portfolio_post_process(self, pct_fee=0., seq_fees=False, sr_mult=np.sqrt(250),
                               n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000,
                               view_weights=True, use_pw=True, multiplier=1,
                               start_date='', end_date='', resample_to='B', *,
                               normalize_pw=False, return_kind='additive',
                               allocation_timing='effective', weight_attr='pw', view=True):
        """Build portfolio returns at original timestamps, then resample.

        Keeps the original arguments and returns the same path-return DataFrame.
        Defaults retain additive returns and allocations effective on each row.
        Use return_kind='simple' for percentage returns (compound after mixing).
        allocation_timing='previous' shifts pw by one source observation only;
        s must already reflect w effective for its return period. Log returns
        are not supported. Use weight_attr='sw' if your Data uses that name.

        Normalization is optional and targets unit gross strategy allocation,
        not unit asset exposure. Missing allocation updates carry forward;
        missing returns/exposures while allocated raise instead of silently
        reallocating capital. Supply explicit zero rows for inactivity.

        Fees apply to allocation * asset weights. Sequential fees include initial
        entry from flat and allocation changes. This target-exposure approximation
        excludes passive weight drift, fill prices, financing and cross-strategy
        netting. Returns must be pre-fee. No automatic final liquidation occurs.
        Nonsequential fees retain the original per-period gross-exposure model.

        Date filtering precedes resampling but follows turnover calculation.
        Empty bins stay NaN. Set n_boot=0 to skip bootstrap and view=False for
        no plots. Set sr_mult for the OUTPUT frequency. Performance summary
        reports arithmetic annualized mean, volatility and Sharpe, not CAGR.
        """
        if return_kind not in ('simple', 'additive'):
            raise ValueError("return_kind must be 'simple' or 'additive'")
        if allocation_timing not in ('effective', 'previous'):
            raise ValueError("allocation_timing must be 'effective' or 'previous'")
        if not np.isfinite(multiplier):
            raise ValueError('multiplier must be finite')
        paths = list(self)
        if not paths:
            raise ValueError('paths must not be empty')
        output = []
        allocations = []
        for dataset in paths:
            if not dataset:
                raise ValueError('Each path must contain at least one strategy')
            parts = {}
            for key, d in dataset.items():
                index = pd.DatetimeIndex(d.index())
                if index.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
                    raise ValueError(f'{key}: timestamps must be nonempty, unique and sorted')
                s, w = np.asarray(d.s, dtype=float), np.asarray(d.w, dtype=float)
                a = np.asarray(getattr(d, weight_attr), dtype=float) if use_pw else np.ones(len(index))
                if s.shape != (len(index),) or a.shape != s.shape or w.ndim != 2 or w.shape[0] != len(index):
                    raise ValueError(f'{key}: incompatible return, allocation or asset-weight shapes')
                if not all(np.isfinite(x).all() for x in (s, w, a)):
                    raise ValueError(f'{key}: inputs must be finite; supply explicit inactivity rows')
                allocation = pd.Series(a, index=index)
                if allocation_timing == 'previous':
                    allocation = allocation.shift(1, fill_value=0.)
                parts[key] = (pd.Series(s, index=index), pd.DataFrame(w, index=index), allocation)
            index = next(iter(parts.values()))[0].index
            for r, _, _ in parts.values():
                index = index.union(r.index)
            index = index.sort_values()
            keys = list(parts)
            rates = np.array([pct_fee.get(k, 0.) if isinstance(pct_fee, dict) else pct_fee for k in keys], dtype=float)
            if not np.isfinite(rates).all() or (rates < 0).any():
                raise ValueError('fee rates must be finite and nonnegative')
            # 1. Align allocation updates. Before a strategy starts, hold cash.
            weights = pd.DataFrame({k: parts[k][2].reindex(index).ffill().fillna(0.) for k in keys})
            # 2. Normalize at each SOURCE timestamp, before return aggregation.
            if normalize_pw:
                denominator = weights.abs().sum(axis=1).replace(0., np.nan)
                weights = weights.div(denominator, axis=0).fillna(0.)
            weights = weights * multiplier
            gross = pd.Series(0., index=index)
            fees = pd.Series(0., index=index)
            leverage = pd.Series(0., index=index)
            net_leverage = pd.Series(0., index=index)
            # 3. Compute each strategy's contribution and effective asset positions.
            for j, key in enumerate(keys):
                returns, asset_weights, _ = parts[key]
                allocation = weights[key]
                missing = ~index.isin(returns.index)
                if (missing & allocation.ne(0.)).any():
                    raise ValueError(f'{key}: missing returns/exposures while allocated; align source calendars explicitly')
                returns = returns.reindex(index).fillna(0.)
                positions = asset_weights.reindex(index).fillna(0.).mul(allocation, axis=0)
                gross += allocation * returns
                leverage += positions.abs().sum(axis=1)
                net_leverage += positions.sum(axis=1)
                # Start flat. A change in allocation itself creates turnover.
                trades = positions - positions.shift(1, fill_value=0.)
                basis = trades.abs().sum(axis=1) if seq_fees else positions.abs().sum(axis=1)
                fees += rates[j] * basis
            frame = pd.DataFrame({'strategy': gross - fees, 'gross_returns': gross,
                                  'fees': fees, 'lev': leverage, 'net_lev': net_leverage,
                                  'counts': weights.ne(0.).sum(axis=1)})
            if return_kind == 'simple' and (frame[['strategy', 'gross_returns']] < -1.).any().any():
                raise ValueError('Simple returns below -100% cannot be compounded under this model')
            # 4. Select the requested window WITHOUT resetting holdings at its start.
            mask = np.ones(len(index), dtype=bool)
            if start_date not in ('', None):
                mask &= index > pd.Timestamp(start_date)
            if end_date not in ('', None):
                mask &= index <= pd.Timestamp(end_date)
            frame, weights = frame.loc[mask], weights.loc[mask]
            if frame.empty:
                raise ValueError('No observations in the requested window')
            # 5. Aggregate completed portfolio returns, never raw strategy returns.
            if resample_to is not None:
                grouped = frame.resample(resample_to)
                result = grouped.last()
                for column in ('strategy', 'gross_returns'):
                    if return_kind == 'simple':
                        result[column] = (1. + frame[column]).resample(resample_to).prod(min_count=1) - 1.
                    else:
                        result[column] = grouped[column].sum(min_count=1)
                result['fees'] = grouped['fees'].sum(min_count=1)
                # Empty bins remain unknown, rather than fabricated zero returns.
                frame = result
                weights = weights.resample(resample_to).last()
            output.append(frame)
            allocations.append(weights)
        names = [f'path_{i+1}' for i in range(len(paths))]
        result = {column: pd.concat([f[column] for f in output], axis=1, keys=names)
                  for column in output[0].columns}
        result['weights'] = pd.concat(allocations, axis=1, keys=names, names=['path', 'strategy'])
        # Preserve the original return value: one return column per path.
        out = result['strategy']
        ts = out.index
        if view:
            equity_curve(out.to_numpy(), ts, pct_fee=pct_fee, return_kind=return_kind)
            returns_distribution(out.to_numpy(), pct_fee=pct_fee)
            if view_weights:
                for name, allocation in zip(names, allocations):
                    visualize_weights(allocation.to_numpy()[:, :, None], allocation.index,
                                      [str(k) + ' — ' + name for k in allocation.columns])
            for field, title in [('lev', 'Paths gross leverage'),
                                 ('net_lev', 'Paths net leverage'),
                                 ('counts', 'Number of allocated strategies')]:
                result[field].plot(title=title)
                plt.grid(True)
                plt.show()
            for name in out:
                print(name)
                performance_summary(out[name].dropna().to_numpy(), sr_mult, pct_fee=pct_fee)
        if n_boot:
            if not np.isfinite(out.to_numpy()).all():
                raise ValueError('Bootstrap requires a common complete return calendar; use n_boot=0 to inspect gaps')
            if (out.std(ddof=0) == 0).any():
                raise ValueError('Bootstrap requires nonzero return variance; use n_boot=0 for constant paths')
            valid_strategy(out.to_numpy(), n_boot, sr_mult, alpha=alpha,
                           alpha_n=alpha_n, pct_fee=pct_fee, block_size=block_size, view=view)
        return out
