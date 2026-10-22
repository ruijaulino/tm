"""Backtest plotting, bootstrap validation and portfolio post-processing.

Portfolio mixing occurs before additive return resampling. Public signatures,
random draws and return formats are preserved. Known legacy
conventions are documented at the relevant methods rather than silently changed.
All implementation is contained in this file; no other post_process module is used.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from tm.containers import Data, Dataset
from tm.constants import *

def calculate_fees(s, weights, seq_fees, pct_fee):
    """Subtract strategy-level fees from returns.

    s has shape (time, paths); weights has shape (time, assets, paths).
    Sequential fees use absolute changes in asset weights, with no entry fee
    on the first row. Otherwise fees use gross exposure on every row.
    pct_fee must broadcast against the resulting (time, paths) fee basis.
    Returns an array with the same shape as s. Allocations are not applied here."""
    if seq_fees:
        # The initial observation is deliberately exempt from sequential fees.
        # Reduce assets first, avoiding a second full asset-weight-sized array.
        changes = np.abs(weights[1:] - weights[:-1]).sum(axis=1)
        dw = np.concatenate((np.zeros_like(weights[:1].sum(axis=1)), changes), axis=0)
    else:
        dw = np.sum(np.abs(weights), axis=1)
    return s - pct_fee * dw

def equity_curve(s, ts, color='g', pct_fee=0, title: str='Equity curve'):
    """Plot cumulative additive returns for each path.

    pct_fee is retained for call compatibility; fees must already be in s."""
    s_df = pd.DataFrame(np.cumsum(s, axis=0), index=ts)
    s_df.plot(color=color, title=title, legend=False)
    plt.grid(True)
    plt.show()

def returns_distribution(s, pct_fee=0, bins=50):
    """Plot the pooled return distribution across all paths.

    pct_fee is retained for call compatibility and is not applied here."""
    title = 'Strategy returns distribution'
    plt.title(title)
    plt.hist(s.ravel(), bins=bins, density=True)
    plt.grid(True)
    plt.show()

def visualize_weights(w, ts, cols=None, view_individual_weights=True):
    """Plot signed and absolute weight totals, then individual assets.

    w has shape (time, assets, paths). ts supplies the time index; cols optionally
    supplies asset labels. A single asset only produces the two total plots.
    Set view_individual_weights=False to show only the two aggregate plots."""
    aux = pd.DataFrame(np.sum(w, axis=1), index=ts)
    aux.plot(title='Weights sum', legend=False)
    plt.grid(True)
    plt.show()
    aux = pd.DataFrame(np.sum(np.abs(w), axis=1), index=ts)
    aux.plot(title='Total Leverage', legend=False)
    plt.grid(True)
    plt.show()
    p = w.shape[1]
    if view_individual_weights and p > 1:
        for i in range(p):
            title = 'Weight for asset %s' % (i + 1) if cols is None else 'Weight for ' + cols[i]
            aux = pd.DataFrame(w[:, i, :], index=ts)
            aux.plot(title=title, legend=False)
            plt.grid(True)
            plt.show()

def bootstrap_sharpe(s, n_boot=1000):
    """Draw independent observations with replacement from a 1D return array.

    Return n_boot unannualized mean/std ratios (population std, ddof=0).
    Uses NumPy's global random state; seed it externally for reproducibility."""
    l = s.size
    idx = np.arange(l, dtype=int)
    idx_ = np.random.choice(idx, (l, n_boot), replace=True)
    s_ = s[idx_]
    boot_samples = np.mean(s_, axis=0) / np.std(s_, axis=0)
    return boot_samples

def block_bootstrap_sharpe(s, n_boot=1000, block_size=10):
    """Draw moving-block bootstrap Sharpe ratios from a 1D return array.

    Each replicate concatenates randomly selected contiguous blocks and truncates
    to the original sample length. Return n_boot unannualized mean/std ratios
    using population standard deviation. Uses NumPy's global random state."""
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError('s must be 1D')
    n = s.size
    if block_size < 1 or block_size > n:
        raise ValueError('block_size must satisfy 1 <= block_size <= len(s)')
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    # One block start for each block and bootstrap replicate.
    starts = np.random.randint(0, max_start + 1, size=(n_blocks, n_boot))
    offsets = np.arange(block_size)
    idx = starts[..., None] + offsets
    # Concatenate blocks down the time axis, retaining replicate columns.
    idx = idx.transpose(0, 2, 1).reshape(n_blocks * block_size, n_boot)
    idx = idx[:n, :]
    s_boot = s[idx]
    mu = np.mean(s_boot, axis=0)
    sigma = np.std(s_boot, axis=0)
    boot_samples = mu / sigma
    return boot_samples

def valid_strategy(s, n_boot, sr_mult, alpha=0.05, alpha_n=1000, pct_fee=0, block_size=20, view=True):
    """Test the path with the lowest annualized Sharpe via block bootstrap.

    Fit a normal approximation to its bootstrap Sharpes and compare the
    probability below zero with alpha / alpha_n (Bonferroni adjustment).
    Return the acceptance boolean. view controls printing and plotting only.
    pct_fee is retained for compatibility; s must already include fees."""
    paths_sr = sr_mult * np.mean(s, axis=0) / np.std(s, axis=0)
    idx_lowest_sr = np.argmin(paths_sr)
    b_samples = block_bootstrap_sharpe(s[:, idx_lowest_sr], n_boot=n_boot, block_size=block_size)
    b_samples *= sr_mult
    mean_b = np.mean(b_samples)
    scale_b = np.std(b_samples)
    valid = norm.cdf(0, loc=mean_b, scale=scale_b) < alpha / alpha_n
    if valid:
        txt = f'** ACCEPT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha / alpha_n}]**'
        if view:
            print(txt)
    else:
        txt = f'** REJECT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha / alpha_n}]**'
        if view:
            print(txt)
    if view:
        plt.title('(Worst path) SR bootstrap distribution')
        x_min, x_max = (0.5 * np.min(b_samples), np.max(b_samples) * 1.5)
        x_pdf = np.linspace(x_min, x_max, 250)
        pdf = np.exp(-0.5 * np.power((x_pdf - mean_b) / scale_b, 2)) / np.sqrt(2 * np.pi * scale_b * scale_b)
        plt.hist(b_samples, density=True, label='Histogram')
        plt.plot(x_pdf, pdf, label='Gaussian Fit')
        plt.axvline(0)
        plt.grid(True)
        plt.legend()
        plt.show()
    return valid

def performance_summary(s, sr_mult, pct_fee=0):
    """Print arithmetic annualized mean, volatility and Sharpe.

    Statistics pool all elements of s, including multiple paths. sr_mult is
    the square root of periods per year. pct_fee is not applied here."""
    print()
    txt = '** PERFORMANCE SUMMARY **'
    print(txt)
    print()
    print('Return: ', np.power(sr_mult, 2) * np.mean(s))
    print('Standard deviation: ', sr_mult * np.std(s))
    print('Sharpe: ', sr_mult * np.mean(s) / np.std(s))
    print()

class Paths(list):
    """List of backtest paths, each represented by a Dataset."""

    def add(self, dataset: Union[Data, Dataset]):
        """Append a Dataset, or wrap a Data instance under the key 'data'."""
        if isinstance(dataset, Data):
            tmp = Dataset()
            tmp.add('data', dataset)
            self.append(tmp)
        else:
            self.append(dataset)

    def post_process(self, pct_fee=0.0, seq_fees=False, sr_mult=np.sqrt(250), n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000, key=None, start_date='', end_date='', simple_view=False):
        """Analyze one dataset key across all paths; return a path-return DataFrame.

    Defaults to the first key. Fees are applied before plots and bootstrap.
    Assumes matching array lengths and uses the LAST path's timestamps.
    For compatibility, start_date and end_date are accepted but unused.
    simple_view suppresses only distribution and asset-weight plots."""
        if len(self) == 0:
            print('No paths to process!')
            return
        keys = list(self[0].keys())
        key = key if key is not None else keys[0]
        print(f'Post process for key {key}')
        s = []
        w = []
        for dataset in self:
            s.append(dataset[key].s[:, None])
            w.append(dataset[key].w)
        if len(s) == 0:
            print('No results to process!')
            return
        ts = dataset[key].index()
        s = np.hstack(s)
        w = np.stack(w, axis=2)
        s = calculate_fees(s, w, seq_fees, pct_fee)
        equity_curve(s, ts, color='g', pct_fee=pct_fee)
        if not simple_view:
            returns_distribution(s, pct_fee=pct_fee, bins=50)
            visualize_weights(w, ts)
        valid_strategy(s, n_boot, sr_mult, alpha=alpha, alpha_n=alpha_n, pct_fee=pct_fee, block_size=block_size)
        performance_summary(s, sr_mult, pct_fee=pct_fee)
        out = pd.DataFrame(s, index=ts)
        out.columns = [f'path_{i + 1}' for i in range(len(out.columns))]
        return out

    def portfolio_post_process(self, pct_fee=0.0, seq_fees=False, sr_mult=np.sqrt(250), n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000, view_weights=True, use_pw=True, multiplier=1, start_date='', end_date='', resample_to='B', normalize_sw=False, view_individual_weights=False):
        """Combine strategy returns and allocations within each path.

    Apply strategy fees and align calendars at the original observation frequency.
    Missing returns/exposures become zero; allocation updates carry forward.
    Mix returns using the allocation on each original row, then SUM the portfolio
    returns in each output bin (additive returns/P&L, not compounded percentages).
    Allocation, leverage and count plots use the LAST snapshot in each bin.
    Empty return bins are zero; empty snapshot bins are zero.
    With normalize_sw=True, divide each row's allocations by their absolute sum
    after alignment and forward-filling, independently for each path. Signs are
    preserved; rows with no allocation remain zero. Then apply multiplier.
    The default False preserves the original allocation magnitudes. With
    use_pw=False, normalization produces equal weights among nonzero allocations.
    view_individual_weights=False hides individual strategy weight charts but
    retains aggregate weight charts. view_weights=False hides both kinds.

    Return a DataFrame with columns path_1, path_2, etc. start_date filters returns
    and leverage with a strict lower bound on output-bin labels; end_date remains
    unused for compatibility. Allocation plots use the same selected output index.

    Compatibility note: both leverage plots use absolute asset exposures times
    SIGNED strategy allocations. The plot named net leverage therefore retains
    the original calculation; this cleanup does not change that convention."""
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
            path_s = []
            path_pw = []
            path_w_abs_sum = []
            for key, data in dataset.items():
                s = data.s
                w = data.w
                pw = data.sw
                ts = data.index()
                if not use_pw:
                    pw = np.ones_like(pw)
                s = pd.DataFrame(calculate_fees(s[:, None], w[:, :, None], seq_fees, pct_fee.get(key, 0)), columns=[key], index=ts)
                path_s.append(s)
                pw = pd.DataFrame(pw[:, None], columns=[key], index=ts)
                path_pw.append(pw)
                # Reduce the existing NumPy array once. nansum preserves pandas'
                # skip-NaN behavior. Both legacy leverage charts use this basis.
                gross_exposure = pd.DataFrame(np.nansum(np.abs(w), axis=1), index=ts, columns=[key])
                path_w_abs_sum.append(gross_exposure)
            # Align original observations before applying strategy allocations.
            path_s = pd.concat(path_s, axis=1)
            path_pw = pd.concat(path_pw, axis=1)
            path_w_abs_sum = pd.concat(path_w_abs_sum, axis=1)
            path_s.columns = keys
            path_pw.columns = keys
            path_w_abs_sum.columns = keys
            path_s = path_s.fillna(0)
            # Count observed nonzero allocations BEFORE forward-filling gaps.
            path_count_non_zero = path_pw.copy(deep=True)
            path_count_non_zero = path_count_non_zero.fillna(0)
            # Carry allocation updates forward; explicit zero allocations remain zero.
            path_pw = path_pw.ffill()
            path_w_abs_sum = path_w_abs_sum.fillna(0)
            # Normalize gross strategy allocation, not underlying asset leverage.
            if normalize_sw:
                total_weight = path_pw.abs().sum(axis=1).replace(0, np.nan)
                path_pw = path_pw.div(total_weight, axis=0).fillna(0)
            path_pw *= multiplier
            non_zero_counts = path_count_non_zero.ne(0).sum(axis=1)
            path_s = pd.DataFrame(np.sum(path_s * path_pw, axis=1), columns=['s'])
            path_leverage = pd.DataFrame(np.sum(path_w_abs_sum * path_pw, axis=1), columns=['s'])
            path_net_leverage = path_leverage.copy()
            path_counts = pd.DataFrame(non_zero_counts, columns=['n'])

            # Aggregate completed portfolio returns, so fold changes inside a
            # bin apply only to the returns belonging to their original rows.
            if resample_to is not None:
                path_s = path_s.resample(resample_to).sum()
                path_pw = path_pw.resample(resample_to).last().fillna(0)
                path_leverage = path_leverage.resample(resample_to).last().fillna(0)
                path_net_leverage = path_net_leverage.resample(resample_to).last().fillna(0)
                path_counts = path_counts.resample(resample_to).last().fillna(0)

            paths_s.append(path_s)
            paths_pw.append(path_pw)
            paths_net_leverage.append(path_net_leverage)
            paths_leverage.append(path_leverage)
            paths_n_datasets.append(path_counts)
        s = pd.concat(paths_s, axis=1)
        lev = pd.concat(paths_leverage, axis=1)
        net_lev = pd.concat(paths_net_leverage, axis=1)
        n_datasets = pd.concat(paths_n_datasets, axis=1)
        if start_date != '':
            s = s[s.index > start_date]
            lev = lev[lev.index > start_date]
            net_lev = net_lev[net_lev.index > start_date]
            n_datasets = n_datasets[n_datasets.index > start_date]
        # Align allocation snapshots to the plotted return dates across paths.
        if view_weights:
            w = np.stack([e.reindex(index=s.index, columns=keys).to_numpy() for e in paths_pw], axis=2)
        out = s.copy(deep=True)
        out.columns = [f'path_{i + 1}' for i in range(len(out.columns))]
        ts = s.index
        s = s.values
        equity_curve(s, ts, color='g', pct_fee=pct_fee)
        returns_distribution(s, pct_fee=pct_fee, bins=50)
        if view_weights:
            visualize_weights(w, ts, keys, view_individual_weights=view_individual_weights)
        lev.plot(legend=False, title='Paths Leverage')
        plt.grid(True)
        plt.show()
        net_lev.plot(legend=False, title='Paths Net Leverage')
        plt.grid(True)
        plt.show()
        n_datasets.plot(legend=False, title='Number of datasets')
        plt.grid(True)
        plt.show()
        valid_strategy(s, n_boot, sr_mult, alpha=alpha, alpha_n=alpha_n, pct_fee=pct_fee, block_size=block_size)
        performance_summary(s, sr_mult, pct_fee=pct_fee)
        return out





######################################
######################################
######################################

# """Backtest plotting, bootstrap validation and portfolio post-processing.

# Portfolio mixing occurs before additive return resampling. Public signatures,
# random draws and return formats are preserved. Known legacy
# conventions are documented at the relevant methods rather than silently changed.
# All implementation is contained in this file; no other post_process module is used.
# """
# import numpy as np
# import pandas as pd
# from scipy.stats import norm
# import matplotlib.pyplot as plt
# from typing import List, Union, Dict
# from tm.containers import Data, Dataset
# from tm.constants import *

# def calculate_fees(s, weights, seq_fees, pct_fee):
#     """Subtract strategy-level fees from returns.

#     s has shape (time, paths); weights has shape (time, assets, paths).
#     Sequential fees use absolute changes in asset weights, with no entry fee
#     on the first row. Otherwise fees use gross exposure on every row.
#     pct_fee must broadcast against the resulting (time, paths) fee basis.
#     Returns an array with the same shape as s. Allocations are not applied here."""
#     if seq_fees:
#         # The initial observation is deliberately exempt from sequential fees.
#         dw = np.abs(weights[1:] - weights[:-1])
#         dw = np.vstack(([np.zeros_like(dw[0])], dw))
#         dw = np.sum(dw, axis=1)
#     else:
#         dw = np.sum(np.abs(weights), axis=1)
#     return s - pct_fee * dw

# def equity_curve(s, ts, color='g', pct_fee=0, title: str='Equity curve'):
#     """Plot cumulative additive returns for each path.

#     pct_fee is retained for call compatibility; fees must already be in s."""
#     s_df = pd.DataFrame(np.cumsum(s, axis=0), index=ts)
#     s_df.plot(color=color, title=title, legend=False)
#     plt.grid(True)
#     plt.show()

# def returns_distribution(s, pct_fee=0, bins=50):
#     """Plot the pooled return distribution across all paths.

#     pct_fee is retained for call compatibility and is not applied here."""
#     title = 'Strategy returns distribution'
#     plt.title(title)
#     plt.hist(s.ravel(), bins=bins, density=True)
#     plt.grid(True)
#     plt.show()

# def visualize_weights(w, ts, cols=None):
#     """Plot signed and absolute weight totals, then individual assets.

#     w has shape (time, assets, paths). ts supplies the time index; cols optionally
#     supplies asset labels. A single asset only produces the two total plots."""
#     aux = pd.DataFrame(np.sum(w, axis=1), index=ts)
#     aux.plot(title='Weights sum', legend=False)
#     plt.grid(True)
#     plt.show()
#     aux = pd.DataFrame(np.sum(np.abs(w), axis=1), index=ts)
#     aux.plot(title='Total Leverage', legend=False)
#     plt.grid(True)
#     plt.show()
#     p = w.shape[1]
#     if p > 1:
#         for i in range(p):
#             title = 'Weight for asset %s' % (i + 1) if cols is None else 'Weight for ' + cols[i]
#             aux = pd.DataFrame(w[:, i, :], index=ts)
#             aux.plot(title=title, legend=False)
#             plt.grid(True)
#             plt.show()

# def bootstrap_sharpe(s, n_boot=1000):
#     """Draw independent observations with replacement from a 1D return array.

#     Return n_boot unannualized mean/std ratios (population std, ddof=0).
#     Uses NumPy's global random state; seed it externally for reproducibility."""
#     l = s.size
#     idx = np.arange(l, dtype=int)
#     idx_ = np.random.choice(idx, (l, n_boot), replace=True)
#     s_ = s[idx_]
#     boot_samples = np.mean(s_, axis=0) / np.std(s_, axis=0)
#     return boot_samples

# def block_bootstrap_sharpe(s, n_boot=1000, block_size=10):
#     """Draw moving-block bootstrap Sharpe ratios from a 1D return array.

#     Each replicate concatenates randomly selected contiguous blocks and truncates
#     to the original sample length. Return n_boot unannualized mean/std ratios
#     using population standard deviation. Uses NumPy's global random state."""
#     s = np.asarray(s)
#     if s.ndim != 1:
#         raise ValueError('s must be 1D')
#     n = s.size
#     if block_size < 1 or block_size > n:
#         raise ValueError('block_size must satisfy 1 <= block_size <= len(s)')
#     n_blocks = int(np.ceil(n / block_size))
#     max_start = n - block_size
#     # One block start for each block and bootstrap replicate.
#     starts = np.random.randint(0, max_start + 1, size=(n_blocks, n_boot))
#     offsets = np.arange(block_size)
#     idx = starts[..., None] + offsets
#     # Concatenate blocks down the time axis, retaining replicate columns.
#     idx = idx.transpose(0, 2, 1).reshape(n_blocks * block_size, n_boot)
#     idx = idx[:n, :]
#     s_boot = s[idx]
#     mu = np.mean(s_boot, axis=0)
#     sigma = np.std(s_boot, axis=0)
#     boot_samples = mu / sigma
#     return boot_samples

# def valid_strategy(s, n_boot, sr_mult, alpha=0.05, alpha_n=1000, pct_fee=0, block_size=20, view=True):
#     """Test the path with the lowest annualized Sharpe via block bootstrap.

#     Fit a normal approximation to its bootstrap Sharpes and compare the
#     probability below zero with alpha / alpha_n (Bonferroni adjustment).
#     Return the acceptance boolean. view controls printing and plotting only.
#     pct_fee is retained for compatibility; s must already include fees."""
#     paths_sr = sr_mult * np.mean(s, axis=0) / np.std(s, axis=0)
#     idx_lowest_sr = np.argmin(paths_sr)
#     b_samples = block_bootstrap_sharpe(s[:, idx_lowest_sr], n_boot=n_boot, block_size=block_size)
#     b_samples *= sr_mult
#     mean_b = np.mean(b_samples)
#     scale_b = np.std(b_samples)
#     valid = norm.cdf(0, loc=mean_b, scale=scale_b) < alpha / alpha_n
#     if valid:
#         txt = f'** ACCEPT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha / alpha_n}]**'
#         if view:
#             print(txt)
#     else:
#         txt = f'** REJECT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha / alpha_n}]**'
#         if view:
#             print(txt)
#     if view:
#         plt.title('(Worst path) SR bootstrap distribution')
#         x_min, x_max = (0.5 * np.min(b_samples), np.max(b_samples) * 1.5)
#         x_pdf = np.linspace(x_min, x_max, 250)
#         pdf = np.exp(-0.5 * np.power((x_pdf - mean_b) / scale_b, 2)) / np.sqrt(2 * np.pi * scale_b * scale_b)
#         plt.hist(b_samples, density=True, label='Histogram')
#         plt.plot(x_pdf, pdf, label='Gaussian Fit')
#         plt.axvline(0)
#         plt.grid(True)
#         plt.legend()
#         plt.show()
#     return valid

# def performance_summary(s, sr_mult, pct_fee=0):
#     """Print arithmetic annualized mean, volatility and Sharpe.

#     Statistics pool all elements of s, including multiple paths. sr_mult is
#     the square root of periods per year. pct_fee is not applied here."""
#     print()
#     txt = '** PERFORMANCE SUMMARY **'
#     print(txt)
#     print()
#     print('Return: ', np.power(sr_mult, 2) * np.mean(s))
#     print('Standard deviation: ', sr_mult * np.std(s))
#     print('Sharpe: ', sr_mult * np.mean(s) / np.std(s))
#     print()

# class Paths(list):
#     """List of backtest paths, each represented by a Dataset."""

#     def add(self, dataset: Union[Data, Dataset]):
#         """Append a Dataset, or wrap a Data instance under the key 'data'."""
#         if isinstance(dataset, Data):
#             tmp = Dataset()
#             tmp.add('data', dataset)
#             self.append(tmp)
#         else:
#             self.append(dataset)

#     def post_process(self, pct_fee=0.0, seq_fees=False, sr_mult=np.sqrt(250), n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000, key=None, start_date='', end_date='', simple_view=False):
#         """Analyze one dataset key across all paths; return a path-return DataFrame.

#     Defaults to the first key. Fees are applied before plots and bootstrap.
#     Assumes matching array lengths and uses the LAST path's timestamps.
#     For compatibility, start_date and end_date are accepted but unused.
#     simple_view suppresses only distribution and asset-weight plots."""
#         if len(self) == 0:
#             print('No paths to process!')
#             return
#         keys = list(self[0].keys())
#         key = key if key is not None else keys[0]
#         print(f'Post process for key {key}')
#         s = []
#         w = []
#         for dataset in self:
#             s.append(dataset[key].s[:, None])
#             w.append(dataset[key].w)
#         if len(s) == 0:
#             print('No results to process!')
#             return
#         ts = dataset[key].index()
#         s = np.hstack(s)
#         w = np.stack(w, axis=2)
#         s = calculate_fees(s, w, seq_fees, pct_fee)
#         equity_curve(s, ts, color='g', pct_fee=pct_fee)
#         if not simple_view:
#             returns_distribution(s, pct_fee=pct_fee, bins=50)
#             visualize_weights(w, ts)
#         valid_strategy(s, n_boot, sr_mult, alpha=alpha, alpha_n=alpha_n, pct_fee=pct_fee, block_size=block_size)
#         performance_summary(s, sr_mult, pct_fee=pct_fee)
#         out = pd.DataFrame(s, index=ts)
#         out.columns = [f'path_{i + 1}' for i in range(len(out.columns))]
#         return out

#     def portfolio_post_process(self, pct_fee=0.0, seq_fees=False, sr_mult=np.sqrt(250), n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000, view_weights=True, use_pw=True, multiplier=1, start_date='', end_date='', resample_to='B', normalize_sw=False):
#         """Combine strategy returns and allocations within each path.

#     Apply strategy fees and align calendars at the original observation frequency.
#     Missing returns/exposures become zero; allocation updates carry forward.
#     Mix returns using the allocation on each original row, then SUM the portfolio
#     returns in each output bin (additive returns/P&L, not compounded percentages).
#     Allocation, leverage and count plots use the LAST snapshot in each bin.
#     Empty return bins are zero; empty snapshot bins are zero.
#     With normalize_sw=True, divide each row's allocations by their absolute sum
#     after alignment and forward-filling, independently for each path. Signs are
#     preserved; rows with no allocation remain zero. Then apply multiplier.
#     The default False preserves the original allocation magnitudes. With
#     use_pw=False, normalization produces equal weights among nonzero allocations.

#     Return a DataFrame with columns path_1, path_2, etc. start_date filters returns
#     and leverage with a strict lower bound on output-bin labels; end_date remains
#     unused for compatibility. Allocation plots use the same selected output index.

#     Compatibility note: both leverage plots use absolute asset exposures times
#     SIGNED strategy allocations. The plot named net leverage therefore retains
#     the original calculation; this cleanup does not change that convention."""
#         if len(self) == 0:
#             print('No paths to process!')
#             return
#         keys = list(self[0].keys())
#         if not isinstance(pct_fee, dict):
#             pct_fee = {k: pct_fee for k in keys}
#         paths_s = []
#         paths_pw = []
#         paths_leverage = []
#         paths_net_leverage = []
#         paths_n_datasets = []
#         for dataset in self:
#             path_s = []
#             path_pw = []
#             path_w_abs_sum = []
#             path_w_sum = []
#             for key, data in dataset.items():
#                 s = data.s
#                 w = data.w
#                 pw = data.sw
#                 ts = data.index()
#                 if not use_pw:
#                     pw = np.ones_like(pw)
#                 s = pd.DataFrame(calculate_fees(s[:, None], w[:, :, None], seq_fees, pct_fee.get(key, 0)), columns=[key], index=ts)
#                 path_s.append(s)
#                 pw = pd.DataFrame(pw[:, None], columns=[key], index=ts)
#                 path_pw.append(pw)
#                 w = pd.DataFrame(w, columns=data.w_cols, index=ts)
#                 path_w_abs_sum.append(pd.DataFrame(w.abs().sum(axis=1), columns=[key]))
#                 path_w_sum.append(pd.DataFrame(w.abs().sum(axis=1), columns=[key]))
#             # Align original observations before applying strategy allocations.
#             path_s = pd.concat(path_s, axis=1)
#             path_pw = pd.concat(path_pw, axis=1)
#             path_w_abs_sum = pd.concat(path_w_abs_sum, axis=1)
#             path_w_sum = pd.concat(path_w_sum, axis=1)
#             path_s.columns = keys
#             path_pw.columns = keys
#             path_w_abs_sum.columns = keys
#             path_w_sum.columns = keys
#             path_s = path_s.fillna(0)
#             # Count observed nonzero allocations BEFORE forward-filling gaps.
#             path_count_non_zero = path_pw.copy(deep=True)
#             path_count_non_zero = path_count_non_zero.fillna(0)
#             # Carry allocation updates forward; explicit zero allocations remain zero.
#             path_pw = path_pw.ffill()
#             path_w_abs_sum = path_w_abs_sum.fillna(0)
#             path_w_sum = path_w_sum.fillna(0)
#             # Normalize gross strategy allocation, not underlying asset leverage.
#             if normalize_sw:
#                 total_weight = path_pw.abs().sum(axis=1).replace(0, np.nan)
#                 path_pw = path_pw.div(total_weight, axis=0).fillna(0)
#             path_pw *= multiplier
#             non_zero_counts = path_count_non_zero.ne(0).sum(axis=1)
#             path_s = pd.DataFrame(np.sum(path_s * path_pw, axis=1), columns=['s'])
#             path_net_leverage = pd.DataFrame(np.sum(path_w_sum * path_pw, axis=1), columns=['s'])
#             path_leverage = pd.DataFrame(np.sum(path_w_abs_sum * path_pw, axis=1), columns=['s'])
#             path_counts = pd.DataFrame(non_zero_counts, columns=['n'])

#             # Aggregate completed portfolio returns, so fold changes inside a
#             # bin apply only to the returns belonging to their original rows.
#             if resample_to is not None:
#                 path_s = path_s.resample(resample_to).sum()
#                 path_pw = path_pw.resample(resample_to).last().fillna(0)
#                 path_leverage = path_leverage.resample(resample_to).last().fillna(0)
#                 path_net_leverage = path_net_leverage.resample(resample_to).last().fillna(0)
#                 path_counts = path_counts.resample(resample_to).last().fillna(0)

#             paths_s.append(path_s)
#             paths_pw.append(path_pw)
#             paths_net_leverage.append(path_net_leverage)
#             paths_leverage.append(path_leverage)
#             paths_n_datasets.append(path_counts)
#         s = pd.concat(paths_s, axis=1)
#         lev = pd.concat(paths_leverage, axis=1)
#         net_lev = pd.concat(paths_net_leverage, axis=1)
#         n_datasets = pd.concat(paths_n_datasets, axis=1)
#         if start_date != '':
#             s = s[s.index > start_date]
#             lev = lev[lev.index > start_date]
#             net_lev = net_lev[net_lev.index > start_date]
#             n_datasets = n_datasets[n_datasets.index > start_date]
#         # Align allocation snapshots to the plotted return dates across paths.
#         w = np.stack([e.reindex(index=s.index, columns=keys).to_numpy() for e in paths_pw], axis=2)
#         out = s.copy(deep=True)
#         out.columns = [f'path_{i + 1}' for i in range(len(out.columns))]
#         ts = s.index
#         s = s.values
#         equity_curve(s, ts, color='g', pct_fee=pct_fee)
#         returns_distribution(s, pct_fee=pct_fee, bins=50)
#         if view_weights:
#             visualize_weights(w, ts, keys)
#         lev.plot(legend=False, title='Paths Leverage')
#         plt.grid(True)
#         plt.show()
#         net_lev.plot(legend=False, title='Paths Net Leverage')
#         plt.grid(True)
#         plt.show()
#         n_datasets.plot(legend=False, title='Number of datasets')
#         plt.grid(True)
#         plt.show()
#         valid_strategy(s, n_boot, sr_mult, alpha=alpha, alpha_n=alpha_n, pct_fee=pct_fee, block_size=block_size)
#         performance_summary(s, sr_mult, pct_fee=pct_fee)
#         return out






















###########################
###########################

# """Backtest plotting, bootstrap validation and portfolio post-processing.

# Readability-only revision of the original implementation. Public signatures,
# calculation order, random draws and return formats are preserved. Known legacy
# conventions are documented at the relevant methods rather than silently changed.
# All implementation is contained in this file; no other post_process module is used.
# """
# import numpy as np
# import pandas as pd
# from scipy.stats import norm
# import matplotlib.pyplot as plt
# from typing import List, Union, Dict
# from tm.containers import Data, Dataset
# from tm.constants import *

# def calculate_fees(s, weights, seq_fees, pct_fee):
#     """Subtract strategy-level fees from returns.

#     s has shape (time, paths); weights has shape (time, assets, paths).
#     Sequential fees use absolute changes in asset weights, with no entry fee
#     on the first row. Otherwise fees use gross exposure on every row.
#     pct_fee must broadcast against the resulting (time, paths) fee basis.
#     Returns an array with the same shape as s. Allocations are not applied here."""
#     if seq_fees:
#         # The initial observation is deliberately exempt from sequential fees.
#         dw = np.abs(weights[1:] - weights[:-1])
#         dw = np.vstack(([np.zeros_like(dw[0])], dw))
#         dw = np.sum(dw, axis=1)
#     else:
#         dw = np.sum(np.abs(weights), axis=1)
#     return s - pct_fee * dw

# def equity_curve(s, ts, color='g', pct_fee=0, title: str='Equity curve'):
#     """Plot cumulative additive returns for each path.

#     pct_fee is retained for call compatibility; fees must already be in s."""
#     s_df = pd.DataFrame(np.cumsum(s, axis=0), index=ts)
#     s_df.plot(color=color, title=title, legend=False)
#     plt.grid(True)
#     plt.show()

# def returns_distribution(s, pct_fee=0, bins=50):
#     """Plot the pooled return distribution across all paths.

#     pct_fee is retained for call compatibility and is not applied here."""
#     title = 'Strategy returns distribution'
#     plt.title(title)
#     plt.hist(s.ravel(), bins=bins, density=True)
#     plt.grid(True)
#     plt.show()

# def visualize_weights(w, ts, cols=None):
#     """Plot signed and absolute weight totals, then individual assets.

#     w has shape (time, assets, paths). ts supplies the time index; cols optionally
#     supplies asset labels. A single asset only produces the two total plots."""
#     aux = pd.DataFrame(np.sum(w, axis=1), index=ts)
#     aux.plot(title='Weights sum', legend=False)
#     plt.grid(True)
#     plt.show()
#     aux = pd.DataFrame(np.sum(np.abs(w), axis=1), index=ts)
#     aux.plot(title='Total Leverage', legend=False)
#     plt.grid(True)
#     plt.show()
#     p = w.shape[1]
#     if p > 1:
#         for i in range(p):
#             title = 'Weight for asset %s' % (i + 1) if cols is None else 'Weight for ' + cols[i]
#             aux = pd.DataFrame(w[:, i, :], index=ts)
#             aux.plot(title=title, legend=False)
#             plt.grid(True)
#             plt.show()

# def bootstrap_sharpe(s, n_boot=1000):
#     """Draw independent observations with replacement from a 1D return array.

#     Return n_boot unannualized mean/std ratios (population std, ddof=0).
#     Uses NumPy's global random state; seed it externally for reproducibility."""
#     l = s.size
#     idx = np.arange(l, dtype=int)
#     idx_ = np.random.choice(idx, (l, n_boot), replace=True)
#     s_ = s[idx_]
#     boot_samples = np.mean(s_, axis=0) / np.std(s_, axis=0)
#     return boot_samples

# def block_bootstrap_sharpe(s, n_boot=1000, block_size=10):
#     """Draw moving-block bootstrap Sharpe ratios from a 1D return array.

#     Each replicate concatenates randomly selected contiguous blocks and truncates
#     to the original sample length. Return n_boot unannualized mean/std ratios
#     using population standard deviation. Uses NumPy's global random state."""
#     s = np.asarray(s)
#     if s.ndim != 1:
#         raise ValueError('s must be 1D')
#     n = s.size
#     if block_size < 1 or block_size > n:
#         raise ValueError('block_size must satisfy 1 <= block_size <= len(s)')
#     n_blocks = int(np.ceil(n / block_size))
#     max_start = n - block_size
#     # One block start for each block and bootstrap replicate.
#     starts = np.random.randint(0, max_start + 1, size=(n_blocks, n_boot))
#     offsets = np.arange(block_size)
#     idx = starts[..., None] + offsets
#     # Concatenate blocks down the time axis, retaining replicate columns.
#     idx = idx.transpose(0, 2, 1).reshape(n_blocks * block_size, n_boot)
#     idx = idx[:n, :]
#     s_boot = s[idx]
#     mu = np.mean(s_boot, axis=0)
#     sigma = np.std(s_boot, axis=0)
#     boot_samples = mu / sigma
#     return boot_samples

# def valid_strategy(s, n_boot, sr_mult, alpha=0.05, alpha_n=1000, pct_fee=0, block_size=20, view=True):
#     """Test the path with the lowest annualized Sharpe via block bootstrap.

#     Fit a normal approximation to its bootstrap Sharpes and compare the
#     probability below zero with alpha / alpha_n (Bonferroni adjustment).
#     Return the acceptance boolean. view controls printing and plotting only.
#     pct_fee is retained for compatibility; s must already include fees."""
#     paths_sr = sr_mult * np.mean(s, axis=0) / np.std(s, axis=0)
#     idx_lowest_sr = np.argmin(paths_sr)
#     b_samples = block_bootstrap_sharpe(s[:, idx_lowest_sr], n_boot=n_boot, block_size=block_size)
#     b_samples *= sr_mult
#     mean_b = np.mean(b_samples)
#     scale_b = np.std(b_samples)
#     valid = norm.cdf(0, loc=mean_b, scale=scale_b) < alpha / alpha_n
#     if valid:
#         txt = f'** ACCEPT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha / alpha_n}]**'
#         if view:
#             print(txt)
#     else:
#         txt = f'** REJECT STRATEGY from normal approximation of bootstrap Sharpe [alpha = {alpha / alpha_n}]**'
#         if view:
#             print(txt)
#     if view:
#         plt.title('(Worst path) SR bootstrap distribution')
#         x_min, x_max = (0.5 * np.min(b_samples), np.max(b_samples) * 1.5)
#         x_pdf = np.linspace(x_min, x_max, 250)
#         pdf = np.exp(-0.5 * np.power((x_pdf - mean_b) / scale_b, 2)) / np.sqrt(2 * np.pi * scale_b * scale_b)
#         plt.hist(b_samples, density=True, label='Histogram')
#         plt.plot(x_pdf, pdf, label='Gaussian Fit')
#         plt.axvline(0)
#         plt.grid(True)
#         plt.legend()
#         plt.show()
#     return valid

# def performance_summary(s, sr_mult, pct_fee=0):
#     """Print arithmetic annualized mean, volatility and Sharpe.

#     Statistics pool all elements of s, including multiple paths. sr_mult is
#     the square root of periods per year. pct_fee is not applied here."""
#     print()
#     txt = '** PERFORMANCE SUMMARY **'
#     print(txt)
#     print()
#     print('Return: ', np.power(sr_mult, 2) * np.mean(s))
#     print('Standard deviation: ', sr_mult * np.std(s))
#     print('Sharpe: ', sr_mult * np.mean(s) / np.std(s))
#     print()

# class Paths(list):
#     """List of backtest paths, each represented by a Dataset."""

#     def add(self, dataset: Union[Data, Dataset]):
#         """Append a Dataset, or wrap a Data instance under the key 'data'."""
#         if isinstance(dataset, Data):
#             tmp = Dataset()
#             tmp.add('data', dataset)
#             self.append(tmp)
#         else:
#             self.append(dataset)

#     def post_process(self, pct_fee=0.0, seq_fees=False, sr_mult=np.sqrt(250), n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000, key=None, start_date='', end_date='', simple_view=False):
#         """Analyze one dataset key across all paths; return a path-return DataFrame.

#     Defaults to the first key. Fees are applied before plots and bootstrap.
#     Assumes matching array lengths and uses the LAST path's timestamps.
#     For compatibility, start_date and end_date are accepted but unused.
#     simple_view suppresses only distribution and asset-weight plots."""
#         if len(self) == 0:
#             print('No paths to process!')
#             return
#         keys = list(self[0].keys())
#         key = key if key is not None else keys[0]
#         print(f'Post process for key {key}')
#         s = []
#         w = []
#         for dataset in self:
#             s.append(dataset[key].s[:, None])
#             w.append(dataset[key].w)
#         if len(s) == 0:
#             print('No results to process!')
#             return
#         ts = dataset[key].index()
#         s = np.hstack(s)
#         w = np.stack(w, axis=2)
#         s = calculate_fees(s, w, seq_fees, pct_fee)
#         equity_curve(s, ts, color='g', pct_fee=pct_fee)
#         if not simple_view:
#             returns_distribution(s, pct_fee=pct_fee, bins=50)
#             visualize_weights(w, ts)
#         valid_strategy(s, n_boot, sr_mult, alpha=alpha, alpha_n=alpha_n, pct_fee=pct_fee, block_size=block_size)
#         performance_summary(s, sr_mult, pct_fee=pct_fee)
#         out = pd.DataFrame(s, index=ts)
#         out.columns = [f'path_{i + 1}' for i in range(len(out.columns))]
#         return out

#     def portfolio_post_process(self, pct_fee=0.0, seq_fees=False, sr_mult=np.sqrt(250), n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000, view_weights=True, use_pw=True, multiplier=1, start_date='', end_date='', resample_to='B', normalize_sw=False):
#         """Combine strategy returns and allocations within each path.

#     Keeps the original procedure: apply fees to each strategy, resample returns,
#     allocations and asset weights with last(), then align strategy calendars.
#     Missing returns/exposures become zero; allocation gaps introduced by alignment
#     carry forward. Empty resampling bins already contain zero and are not filled.
#     With normalize_sw=True, divide each row's allocations by their absolute sum
#     after alignment and forward-filling, independently for each path. Signs are
#     preserved; rows with no allocation remain zero. Then apply multiplier.
#     The default False preserves the original allocation magnitudes. With
#     use_pw=False, normalization produces equal weights among nonzero allocations.

#     Return a DataFrame with columns path_1, path_2, etc. start_date filters returns
#     and leverage with a strict lower bound; end_date remains unused. Allocation
#     plots retain the original unfiltered weights, including the existing shape
#     limitation when start_date removes rows and view_weights is True.

#     Compatibility note: both leverage plots use absolute asset exposures times
#     SIGNED strategy allocations. The plot named net leverage therefore retains
#     the original calculation; this cleanup does not change that convention."""
#         if len(self) == 0:
#             print('No paths to process!')
#             return
#         keys = list(self[0].keys())
#         if not isinstance(pct_fee, dict):
#             pct_fee = {k: pct_fee for k in keys}
#         paths_s = []
#         paths_pw = []
#         paths_leverage = []
#         paths_net_leverage = []
#         paths_n_datasets = []
#         for dataset in self:
#             path_s = []
#             path_pw = []
#             path_w_abs_sum = []
#             path_w_sum = []
#             for key, data in dataset.items():
#                 s = data.s
#                 w = data.w
#                 pw = data.sw
#                 ts = data.index()
#                 if not use_pw:
#                     pw = np.ones_like(pw)
#                 s = pd.DataFrame(calculate_fees(s[:, None], w[:, :, None], seq_fees, pct_fee.get(key, 0)), columns=[key], index=ts)
#                 if resample_to is not None:
#                     s = s.resample(resample_to).last().fillna(0)
#                 path_s.append(s)
#                 pw = pd.DataFrame(pw[:, None], columns=[key], index=ts)
#                 if resample_to is not None:
#                     pw = pw.resample(resample_to).last().fillna(0)
#                 path_pw.append(pw)
#                 w = pd.DataFrame(w, columns=data.w_cols, index=ts)
#                 if resample_to is not None:
#                     w = w.resample(resample_to).last().fillna(0)
#                 path_w_abs_sum.append(pd.DataFrame(w.abs().sum(axis=1), columns=[key]))
#                 path_w_sum.append(pd.DataFrame(w.abs().sum(axis=1), columns=[key]))
#             # Align the separately resampled strategy tables.
#             path_s = pd.concat(path_s, axis=1)
#             path_pw = pd.concat(path_pw, axis=1)
#             path_w_abs_sum = pd.concat(path_w_abs_sum, axis=1)
#             path_w_sum = pd.concat(path_w_sum, axis=1)
#             path_s.columns = keys
#             path_pw.columns = keys
#             path_w_abs_sum.columns = keys
#             path_w_sum.columns = keys
#             path_s = path_s.fillna(0)
#             # Count observed nonzero allocations BEFORE forward-filling gaps.
#             path_count_non_zero = path_pw.copy(deep=True)
#             path_count_non_zero = path_count_non_zero.fillna(0)
#             # Carry allocation updates forward, but keep pre-existing zero bins.
#             path_pw = path_pw.ffill()
#             path_w_abs_sum = path_w_abs_sum.fillna(0)
#             path_w_sum = path_w_sum.fillna(0)
#             # Normalize gross strategy allocation, not underlying asset leverage.
#             if normalize_sw:
#                 total_weight = path_pw.abs().sum(axis=1).replace(0, np.nan)
#                 path_pw = path_pw.div(total_weight, axis=0).fillna(0)
#             path_pw *= multiplier
#             non_zero_counts = path_count_non_zero.ne(0).sum(axis=1)
#             path_s = pd.DataFrame(np.sum(path_s * path_pw, axis=1), columns=['s'])
#             paths_s.append(path_s)
#             paths_pw.append(path_pw)
#             paths_net_leverage.append(pd.DataFrame(np.sum(path_w_sum * path_pw, axis=1), columns=['s']))
#             paths_leverage.append(pd.DataFrame(np.sum(path_w_abs_sum * path_pw, axis=1), columns=['s']))
#             paths_n_datasets.append(pd.DataFrame(non_zero_counts, columns=['n']))
#         s = pd.concat(paths_s, axis=1)
#         w = np.stack([e.values for e in paths_pw], axis=2)
#         lev = pd.concat(paths_leverage, axis=1)
#         net_lev = pd.concat(paths_net_leverage, axis=1)
#         n_datasets = pd.concat(paths_n_datasets, axis=1)
#         if start_date != '':
#             s = s[s.index > start_date]
#             lev = lev[lev.index > start_date]
#             net_lev = net_lev[net_lev.index > start_date]
#             n_datasets = n_datasets[n_datasets.index > start_date]
#         out = s.copy(deep=True)
#         out.columns = [f'path_{i + 1}' for i in range(len(out.columns))]
#         ts = s.index
#         s = s.values
#         equity_curve(s, ts, color='g', pct_fee=pct_fee)
#         returns_distribution(s, pct_fee=pct_fee, bins=50)
#         if view_weights:
#             visualize_weights(w, ts, keys)
#         lev.plot(legend=False, title='Paths Leverage')
#         plt.grid(True)
#         plt.show()
#         net_lev.plot(legend=False, title='Paths Net Leverage')
#         plt.grid(True)
#         plt.show()
#         n_datasets.plot(legend=False, title='Number of datasets')
#         plt.grid(True)
#         plt.show()
#         valid_strategy(s, n_boot, sr_mult, alpha=alpha, alpha_n=alpha_n, pct_fee=pct_fee, block_size=block_size)
#         performance_summary(s, sr_mult, pct_fee=pct_fee)
#         return out
