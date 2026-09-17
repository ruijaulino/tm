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

        
        lev = np.sum(np.abs(w), axis = 1)

        s = pd.DataFrame(s, index = ts)
        s.columns = [f'path_{i+1}' for i in range(len(s.columns))]
        lev = pd.DataFrame(lev, index = ts)
        lev.columns = [f'path_{i+1}' for i in range(len(lev.columns))]        
        

        return {'strategy':s, 'lev':lev}

    def portfolio_post_process(self, pct_fee = 0., seq_fees = False, sr_mult = np.sqrt(250), n_boot = 1000, block_size = 20, alpha = 0.05, alpha_n = 1000, view_weights = True, use_sw = True, multiplier = 1, normalize_sw = False, start_date = '', end_date = '', resample_to = 'B'):
        """
        Post-process a set of portfolio paths.

        Cache timestamp conversion, resampling positions and alignment maps across
        paths. Combine returns and leverage with NumPy, constructing pandas
        objects only for the final outputs and optional weight visualization.
        The cache is local to this call; differing path timestamps are supported.
        Sorted integer Unix-second timestamps use a NumPy fast path for 'B'
        resampling. Other frequencies and timestamp layouts use pandas.
        Contiguous row selections use array views instead of advanced-index copies.

        Notes
        -----
        The fast resampling path assumes s, sw and w contain finite values.  This is
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
        paths_sw = []
        paths_leverage = []
        paths_net_leverage = []
        paths_n_datasets = []

        # Cache the most recent timestamp layout for each dataset key. Usually
        # every path uses the same timestamps, so pandas resamples once per key.
        layout_cache = {}
        previous_layouts = None
        alignment = None

        def row_selector(positions):
            # Basic slicing returns a view; integer-array indexing makes a copy.
            if positions.size and np.all(positions[1:] == positions[:-1] + 1):
                return slice(int(positions[0]), int(positions[-1]) + 1)
            return positions

        for dataset in self:
            layouts = []
            for key in keys:
                data = dataset[key]
                source_ts = np.asarray(data.ts)
                cached = layout_cache.get(key)
                if cached is None or not np.array_equal(source_ts, cached[0]):
                    if (
                        resample_to == 'B'
                        and source_ts.size
                        and np.issubdtype(source_ts.dtype, np.integer)
                        and np.all(source_ts[1:] >= source_ts[:-1])
                    ):
                        # Default pandas BusinessDay bins are left-closed:
                        # Friday includes Saturday/Sunday. Consecutive business
                        # dates have consecutive codes, including before 1970.
                        days = (source_ts // 86400).astype(np.int64)
                        weeks, weekdays = np.divmod(days + 3, 7)
                        codes = weeks * 5 + np.minimum(weekdays, 4)
                        pos = np.r_[np.flatnonzero(codes[1:] != codes[:-1]),
                                    source_ts.size - 1].astype(np.intp)
                        valid_rows = (codes[pos] - codes[0]).astype(np.intp)
                        output_codes = np.arange(codes[0], codes[-1] + 1)
                        weeks, weekdays = np.divmod(output_codes, 5)
                        output_days = weeks * 7 + weekdays - 3
                        out_index = pd.DatetimeIndex(
                            output_days.astype('datetime64[D]').astype('datetime64[ns]'),
                            freq='B',
                        )
                    elif resample_to is None:
                        index = pd.to_datetime(source_ts, unit='s')
                        out_index = index
                        valid_rows = np.arange(data.n, dtype=np.intp)
                        pos = valid_rows
                    else:
                        index = pd.to_datetime(source_ts, unit='s')
                        positions = pd.Series(
                            np.arange(data.n, dtype=np.int64),
                            index=index, copy=False,
                        ).resample(resample_to).last()
                        out_index = positions.index
                        valid_rows = np.flatnonzero(positions.notna().to_numpy())
                        pos = positions.iloc[valid_rows].to_numpy(dtype=np.intp)
                    cached = (source_ts, out_index, valid_rows, pos)
                    layout_cache[key] = cached
                layouts.append(cached)

            # Reuse the union and integer alignment maps when timestamps match.
            if previous_layouts is None or any(
                new is not old for new, old in zip(layouts, previous_layouts)
            ):
                path_index = layouts[0][1]
                same_index = all(path_index.equals(layout[1]) for layout in layouts[1:])
                if not same_index:
                    for layout in layouts[1:]:
                        path_index = path_index.union(layout[1])
                    path_index = path_index.sort_values()
                alignment = []
                for _, out_index, valid_rows, pos in layouts:
                    all_rows = (
                        np.arange(len(path_index), dtype=np.intp)
                        if same_index else path_index.get_indexer(out_index)
                    )
                    rows = all_rows[valid_rows]
                    has_prev = pos > 0
                    previous_pos = pos[has_prev]
                    alignment.append((
                        row_selector(all_rows), row_selector(rows), pos,
                        row_selector(pos), has_prev,
                        row_selector(previous_pos), row_selector(previous_pos - 1),
                    ))
                previous_layouts = layouts

            m = len(path_index)
            sw_values = np.full((m, len(keys)), np.nan, dtype=np.float64)
            for j, key in enumerate(keys):
                all_rows, rows, pos, source_rows, _, _, _ = alignment[j]
                # Empty resample bins are zero. Alignment gaps remain NaN.
                sw_values[all_rows, j] = 0.0
                sw_values[rows, j] = dataset[key].sw[source_rows] if use_sw else 1.0

            non_zero_counts = np.sum(
                (~np.isnan(sw_values)) & (sw_values != 0), axis=1
            )

            # Forward-fill each column without an additional n x k index array.
            # Leading NaNs remain NaN, exactly as in DataFrame.ffill().
            row_numbers = np.arange(m, dtype=np.intp)
            for j in range(len(keys)):
                column = sw_values[:, j]
                missing = np.isnan(column)
                if missing.any():
                    last = np.maximum.accumulate(
                        np.where(missing, -1, row_numbers)
                    )
                    fill = missing & (last >= 0)
                    column[fill] = column[last[fill]]

            if normalize_sw:
                totals = np.nansum(np.abs(sw_values), axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    sw_values /= totals[:, None]
            sw_values *= multiplier

            # Stream each dataset into the portfolio totals. No aligned return,
            # gross-leverage or net-leverage DataFrames/matrices are required.
            values = np.zeros((m, 3), dtype=np.float64)
            for j, key in enumerate(keys):
                _, rows, pos, source_rows, has_prev, current_rows, previous_rows = alignment[j]
                if not pos.size:
                    continue
                data = dataset[key]
                w = data.w
                w_last = w[source_rows]
                gross_last = np.sum(np.abs(w_last), axis=1)
                net_last = np.sum(w_last, axis=1)

                fee = pct_fee.get(key, 0)
                if np.ndim(fee) != 0:
                    raise ValueError('pct_fee values must be scalars')
                if fee == 0:
                    s_last = data.s[source_rows]
                elif seq_fees:
                    # Turnover uses the previous SOURCE row, not the previous
                    # resampled row. This preserves fees-before-resampling.
                    turnover = np.zeros(pos.size, dtype=np.float64)
                    turnover[has_prev] = np.sum(
                        np.abs(w[current_rows] - w[previous_rows]), axis=1
                    )
                    s_last = data.s[source_rows] - fee * turnover
                else:
                    s_last = data.s[source_rows] - fee * gross_last

                contribution = np.column_stack((s_last, gross_last, net_last)).astype(
                    np.float64, copy=False
                )
                contribution *= sw_values[rows, j, None]
                contribution[np.isnan(contribution)] = 0.0
                values[rows] += contribution

            paths_s.append(pd.Series(values[:, 0], index=path_index, name='s'))
            paths_leverage.append(pd.Series(values[:, 1], index=path_index, name='s'))
            paths_net_leverage.append(pd.Series(values[:, 2], index=path_index, name='s'))
            paths_n_datasets.append(pd.Series(non_zero_counts, index=path_index, name='n'))
            if view_weights:
                paths_sw.append(pd.DataFrame(sw_values, index=path_index, columns=keys))

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



        ts = s.index
        s_values = s.to_numpy(copy=False)

        equity_curve(s_values, ts, color='g', pct_fee=pct_fee)
        returns_distribution(s_values, pct_fee=pct_fee, bins=50)

        if view_weights:
            # Align each path to the final filtered index.  This also fixes the old
            # start_date mismatch between w and ts.
            w = np.stack(
                [sw.reindex(ts).to_numpy(copy=False) for sw in paths_sw],
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


        s.columns = [f'path_{i+1}' for i in range(s.shape[1])]
        lev.columns = [f'path_{i+1}' for i in range(lev.shape[1])]


        return {'strategy':s, 'lev':lev}

def base_strategy_assembly(
    data, method='iv', n_paths=3, k_folds=4, multiplier = 1,
    seed=None
):
    """
    Parameters
    ----------
    data : dict
        {
            'strat1': {'strategy': returns_df, 'lev': leverage_df},
            ...
        }

        Returns and leverage are matched by timestamp and column label.
        Returns are assumed to already reflect strategy leverage.

    return_leverage : bool
        If True, append assembled leverage to the returned outputs.

    Returns
    -------
    assembled, weights, available_counts
    optionally followed by assembled_leverage.

    Notes
    -----
    Weights are normalized once per fold. Missing strategy returns
    receive zero allocation without redistributing their weight.

    Portfolio leverage is the weighted sum of strategy leverages,
    before any netting of underlying positions.
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("data must be a nonempty dictionary")
    if method not in ('iv', 'g'):
        raise ValueError("unknown method")
    if not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")

    names = list(data)
    frames = []
    leverage_frames = []

    for name, item in data.items():
        if not isinstance(item, dict) or not {'strategy', 'lev'} <= item.keys():
            raise ValueError(f"{name}: expected 'strategy' and 'lev'")

        s, lev = item['strategy'], item['lev']

        for label, df in [('strategy', s), ('lev', lev)]:
            if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
                raise ValueError(f"{name}/{label}: expected a dataframe with columns")
            if not df.index.is_unique or not df.columns.is_unique:
                raise ValueError(f"{name}/{label}: index and columns must be unique")

        if not s.columns.isin(lev.columns).all():
            raise ValueError(f"{name}: leverage must contain every strategy column")

        frames.append(s)
        leverage_frames.append(lev)

    # Union of return timestamps.
    index = frames[0].index
    for df in frames[1:]:
        index = index.union(df.index)
    index = index.sort_values()

    n = len(index)
    n_strategies = len(names)

    if not isinstance(k_folds, int) or not 2 <= k_folds <= n:
        raise ValueError("k_folds must be between 2 and the number of timestamps")

    arrays = [
        s.reindex(index).to_numpy(dtype=float)
        for s in frames
    ]
    leverage_arrays = [
        lev.reindex(index=index, columns=s.columns).to_numpy(dtype=float)
        for s, lev in zip(frames, leverage_frames)
    ]

    if any(np.isinf(a).any() for a in arrays + leverage_arrays):
        raise ValueError("returns and leverage may contain NaN, but not infinity")
    if any(np.any(a < 0) for a in leverage_arrays):
        raise ValueError("gross leverage must be nonnegative")

    folds = np.array_split(np.arange(n), k_folds)
    rng = np.random.default_rng(seed)

    result = np.zeros((n, n_paths))
    counts = np.zeros((n, n_paths), dtype=int)
    weight_history = np.zeros((n, n_paths, n_strategies))
    leverage_history = np.zeros((n, n_paths))

    for path in range(n_paths):
        # Match the selected return path to its leverage path.
        selected = [rng.integers(a.shape[1]) for a in arrays]

        returns = np.column_stack([
            a[:, col] for a, col in zip(arrays, selected)
        ])
        leverage = np.column_stack([
            a[:, col] for a, col in zip(leverage_arrays, selected)
        ])

        available = ~np.isnan(returns)
        counts[:, path] = available.sum(axis=1)

        for test_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False

            train = pd.DataFrame(returns[train_mask])
            variance = train.var(ddof=1).to_numpy()

            if method == 'iv':
                numerator = np.ones(n_strategies)
                denominator = np.sqrt(variance)
            else:
                numerator = np.maximum(train.mean().to_numpy(), 0.0)
                denominator = variance

            valid = (
                np.isfinite(numerator)
                & np.isfinite(denominator)
                & (denominator > 0)
            )

            fold_weights = np.divide(
                numerator,
                denominator,
                out=np.zeros(n_strategies),
                where=valid,
            )

            total = fold_weights.sum()
            if total > 0:
                fold_weights /= total

            # No renormalization when a strategy is unavailable.
            row_weights = available[test_idx] * fold_weights
            row_weights *= multiplier
            weight_history[test_idx, path, :] = row_weights

            test_returns = np.nan_to_num(returns[test_idx], nan=0.0)
            result[test_idx, path] = np.sum(
                row_weights * test_returns, axis=1
            )

            # Missing leverage matters only for nonzero allocations.
            test_leverage = leverage[test_idx]
            contributions = np.zeros_like(test_leverage)
            np.multiply(
                row_weights,
                test_leverage,
                out=contributions,
                where=row_weights > 0,
            )
            leverage_history[test_idx, path] = contributions.sum(axis=1)

    columns = [f'path{i + 1}' for i in range(n_paths)]

    assembled = pd.DataFrame(result, index=index, columns=columns)

    weights = pd.DataFrame(
        weight_history.reshape(n, n_paths * n_strategies),
        index=index,
        columns=pd.MultiIndex.from_product(
            [columns, names],
            names=['path', 'strategy'],
        ),
    )

    available_counts = pd.DataFrame(counts, index=index, columns=columns)

    assembled_leverage = pd.DataFrame(
        leverage_history, index=index, columns=columns
    )
    return assembled, weights, available_counts, assembled_leverage


def plot_strategy_assembly(
    assembled, weights, counts, leverage, compound=False,
):
    # 1. Cumulative returns
    cumulative = (
        (1 + assembled).cumprod() - 1
        if compound else assembled.cumsum()
    )

    fig_returns, ax = plt.subplots(
        figsize=(12, 5), constrained_layout=True
    )
    cumulative.plot(ax=ax)
    ax.set_title('Cumulative returns')
    ax.set_ylabel(
        'Compounded return' if compound else 'Cumulative return (sum)'
    )
    ax.axhline(0, color='black', linewidth=0.7)
    ax.grid(alpha=0.25)

    # 2. Weights: all strategies and paths on one axis
    paths = assembled.columns
    strategies = weights.columns.get_level_values('strategy').unique()
    cmap = plt.get_cmap('tab20', len(strategies))
    line_styles = ['-', '--', ':', '-.']

    fig_weights, ax = plt.subplots(
        figsize=(12, 5), constrained_layout=True
    )

    for j, path in enumerate(paths):
        for i, strategy in enumerate(strategies):
            ax.plot(
                weights.index,
                weights[(path, strategy)],
                color=cmap(i),
                linestyle=line_styles[j % len(line_styles)],
                linewidth=1.3,
                label=f'{strategy} — {path}',
            )

    ax.set_title('Strategy weights')
    ax.set_ylabel('Allocation')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(
        loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False
    )

    # 3. Number of available strategies
    fig_counts, ax = plt.subplots(
        figsize=(12, 4), constrained_layout=True
    )
    counts.plot(ax=ax, drawstyle='steps-post')
    ax.set_title('Available strategies')
    ax.set_ylabel('Count')
    ax.set_yticks(np.arange(len(strategies) + 1))
    ax.set_ylim(-0.1, len(strategies) + 0.1)
    ax.grid(alpha=0.25)

    # 4. Portfolio leverage: already multiplied by strategy weights
    fig_leverage, ax = plt.subplots(
        figsize=(12, 5), constrained_layout=True
    )

    # Matplotlib preserves gaps where leverage is unknown (NaN).
    for path in paths:
        ax.plot(leverage.index, leverage[path], label=path)

    ax.set_title('Portfolio leverage')
    ax.set_ylabel('Leverage')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    
    plt.show()    
    



def strategy_assembly(
    data, method='iv', n_paths=3, k_folds=4, sr_mul = np.sqrt(260), multiplier = 1,
    seed=None):

    """
    Parameters
    ----------
    data : dict
        {
            'strat1': {'strategy': returns_df, 'lev': leverage_df},
            ...
        }

        Returns and leverage are matched by timestamp and column label.
        Returns are assumed to already reflect strategy leverage.

    return_leverage : bool
        If True, append assembled leverage to the returned outputs.

    Returns
    -------
    assembled, weights, available_counts
    optionally followed by assembled_leverage.

    Notes
    -----
    Weights are normalized once per fold. Missing strategy returns
    receive zero allocation without redistributing their weight.

    Portfolio leverage is the weighted sum of strategy leverages,
    before any netting of underlying positions.
    """
    assembled, weights, counts, leverage = base_strategy_assembly(data = data, method=method, n_paths=n_paths, k_folds=k_folds, multiplier = multiplier, seed = seed)

    print('** STATISTICS **')
    v = assembled.values
    print('-> Annual return: ', np.mean(np.mean(v, axis = 0))*sr_mul*sr_mul )
    print('-> Annual scale: ', np.mean(np.std(v, axis = 0))*sr_mul )
    print('-> Annual sharpe: ', np.mean(np.mean(v, axis = 0)/np.std(v, axis = 0))*sr_mul )
    print('-> Weights')
    print(weights.mean(axis=0).unstack('strategy').mean(axis=0))




    plot_strategy_assembly(
        assembled, weights, counts, leverage
    )
    return {'assembled':assembled, 'weights':weights, 'counts':counts, 'leverage':leverage}

