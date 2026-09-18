"""Portfolio preparation, analysis, Paths utilities and strategy assembly.

All post-processing implementation is included here; portfolio_pandas.py is
not required. Existing tm.containers Data/Dataset types are still used by Paths.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from tm.containers import Data, Dataset



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


def performance_summary(s, sr_mult, pct_fee = 0):
    print()
    txt='** PERFORMANCE SUMMARY **' 
    print(txt)
    print()
    print('Return: ', np.power(sr_mult, 2) * np.mean(s))
    print('Standard deviation: ', sr_mult * np.std(s))
    print('Sharpe: ', sr_mult * np.mean(s) / np.std(s))
    print()


def prepare_portfolio(paths, resample_to='B', seq_fees=False, weight_attr='pw'):
    """Prepare tables once. Each path contains time-by-strategy DataFrames.

    Inputs are the existing Paths objects. s is pre-fee additive returns;
    w is the original asset-weight matrix; pw/sw is the portfolio allocation.
    All source values must be finite and timestamps sorted. Integer data.ts is
    interpreted as Unix seconds; otherwise data.index() supplies timestamps.

    Process one path at a time to bound intermediate memory. Strategies share
    one pandas grouping per path, and identical time grids reuse the grouping.
    Frequencies are delegated to pandas; there is no custom calendar arithmetic.
    Widely differing source grids can still require large intermediate tables.
    """
    paths = list(paths)
    if not paths or not paths[0]:
        raise ValueError('paths must contain at least one nonempty dataset')
    keys = list(paths[0])
    names = [f'path_{i+1}' for i in range(len(paths))]
    frames = []
    cache = {}
    recent = None
    calendar = None
    metrics = ['returns', 'fee_basis', 'pw', 'gross', 'net', 'available']

    for dataset in paths:
        if set(dataset) != set(keys):
            raise ValueError('all paths must contain the same strategy keys')
        parts = {}
        for key in keys:
            d = dataset[key]
            s, w = np.asarray(d.s), np.asarray(d.w)
            pw = np.asarray(getattr(d, weight_attr))
            raw = np.asarray(getattr(d, 'ts', None))
            seconds = raw.ndim == 1 and np.issubdtype(raw.dtype, np.integer)
            if not seconds:
                raw = np.asarray(d.index(), dtype='datetime64[ns]')
            if raw.ndim != 1 or s.shape != (len(raw),) or pw.shape != s.shape:
                raise ValueError(f'{key}: timestamps, returns and allocations must be vectors of equal length')
            if w.ndim != 2 or w.shape[0] != len(raw):
                raise ValueError(f'{key}: w must have shape (n, assets)')
            if not all(np.isfinite(a).all() for a in (s, w, pw)):
                raise ValueError(f'{key}: source arrays must be finite')

            entry = None
            for candidate in (recent, cache.get(key)):
                if candidate is not None and candidate[1] == seconds and np.array_equal(raw, candidate[0]):
                    entry = candidate
                    break
            if entry is None:
                index = pd.to_datetime(raw, unit='s') if seconds else pd.DatetimeIndex(raw)
                if index.hasnans or not index.is_monotonic_increasing:
                    raise ValueError('timestamps must be sorted and contain no NaT')
                if resample_to is None and not index.is_unique:
                    raise ValueError('timestamps must be unique without resampling')
                entry = (raw, seconds, index)
            recent = cache[key] = entry
            index = entry[2]

            gross = np.abs(w).sum(axis=1)
            if seq_fees:
                basis = np.zeros(len(s))
                basis[1:] = np.abs(w[1:] - w[:-1]).sum(axis=1)
            else:
                basis = gross
            part = pd.DataFrame({
                'returns': s, 'fee_basis': basis, 'pw': pw,
                'gross': gross, 'net': w.sum(axis=1),
                'available': np.ones(len(s)),
            }, index=index)
            if not index.is_unique:
                # Collapse identical timestamps before cross-strategy alignment.
                # Fees already include every original source-row transition.
                group = part.groupby(level=0, sort=False)
                duplicate_sums = group[['returns', 'fee_basis']].sum()
                part = group.last()
                part[['returns', 'fee_basis']] = duplicate_sums
            parts[key] = part

        # Align all strategies before resampling. NaNs introduced by alignment
        # are skipped by last(), so each strategy keeps its own last observation.
        frame = pd.concat(parts, axis=1)
        if not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()
        if resample_to is not None and len(frame):
            # Ask pandas for bin boundaries once per distinct path time grid.
            # size() counts rows even if a particular strategy is missing there.
            if calendar is None or not frame.index.equals(calendar[0]):
                sizes = frame.iloc[:, 0].resample(resample_to).size()
                codes = np.repeat(np.arange(len(sizes)), sizes.to_numpy())
                calendar = (frame.index, sizes.index, codes)
            _, output_index, codes = calendar
            groups = frame.groupby(codes, sort=False)
            # Whole-table reductions avoid per-column Python aggregation calls.
            summed = groups.sum()
            frame = groups.last()
            sum_columns = frame.columns.get_level_values(1).isin(['returns', 'fee_basis'])
            frame.loc[:, sum_columns] = summed.loc[:, sum_columns]
            frame = frame.reindex(pd.RangeIndex(len(output_index)))
            frame.index = output_index
        frames.append(frame)

    # Align paths once too, so every output uses the same time axis.
    same_grid = all(frame.index.equals(frames[0].index) for frame in frames[1:])
    combined = pd.concat(frames, axis=1, keys=names).fillna(0)
    if not combined.index.is_monotonic_increasing:
        combined = combined.sort_index()
    if combined.empty:
        raise ValueError('no observations to prepare')
    if resample_to is not None and not same_grid:
        # Include calendar gaps between paths with disjoint time ranges.
        full_index = pd.date_range(combined.index[0], combined.index[-1], freq=resample_to)
        # Preserve labels from independently anchored multi-day path grids too.
        combined = combined.reindex(full_index.union(combined.index), fill_value=0)

    data = {}
    for name in names:
        path = combined[name]
        data[name] = {
            metric: path.xs(metric, level=1, axis=1).reindex(columns=keys)
            for metric in metrics
        }
        data[name]['available'] = data[name]['available'].astype(bool)
    return dict(data=data, timestamps=combined.index, strategies=keys,
                paths=names, seq_fees=seq_fees)


def analyze_portfolio(
    prepared, pct_fee=0., use_pw=True, normalize_pw=False, multiplier=1.,
    start_date='', end_date='', sr_mult=np.sqrt(250),
    view=True, view_weights=True, n_boot=0, block_size=20,
    alpha=0.05, alpha_n=1000,
):
    """Reuse prepared pandas tables; return the same array outputs as NumPy.

    Normalization is independent at each timestamp and path. The final allocation
    in each bin multiplies its full aggregated strategy return. Gross leverage is
    before cross-strategy position netting. Fees apply to strategy returns using
    signed allocation, preserving the existing convention.
    """
    index = prepared['timestamps']
    mask = np.ones(len(index), dtype=bool)
    if start_date != '':
        mask &= index > pd.Timestamp(start_date)
    if end_date != '':
        mask &= index <= pd.Timestamp(end_date)
    rows = np.flatnonzero(mask)
    if not rows.size:
        raise ValueError('no observations within the requested date range')
    selection = slice(int(rows[0]), int(rows[-1])+1)
    keys = prepared['strategies']
    fees = pd.Series(
        [pct_fee.get(k, 0.) for k in keys] if isinstance(pct_fee, dict) else [pct_fee]*len(keys),
        index=keys, dtype=float,
    )
    if not np.isfinite(fees.to_numpy()).all():
        raise ValueError('fees must be finite scalars')
    collected = {k: [] for k in ['strategy', 'lev', 'net_lev', 'weights', 'counts', 'available_counts']}
    for name in prepared['paths']:
        tables = prepared['data'][name]
        available = tables['available'].iloc[selection]
        pw = tables['pw'].iloc[selection]
        weights = pw.where(available, 0.) if use_pw else available.astype(float)
        collected['counts'].append(weights.ne(0).sum(axis=1).to_numpy())
        collected['available_counts'].append(available.sum(axis=1).to_numpy())
        if normalize_pw:
            total = weights.abs().sum(axis=1)
            weights = weights.div(total.replace(0, np.nan), axis=0).fillna(0)
        weights *= multiplier
        net_returns = tables['returns'].iloc[selection] - tables['fee_basis'].iloc[selection].mul(fees, axis='columns')
        collected['strategy'].append((net_returns * weights).sum(axis=1).to_numpy())
        collected['lev'].append((tables['gross'].iloc[selection] * weights.abs()).sum(axis=1).to_numpy())
        collected['net_lev'].append((tables['net'].iloc[selection] * weights).sum(axis=1).to_numpy())
        collected['weights'].append(weights.to_numpy())

    result = {k: np.stack(v, axis=-1) for k, v in collected.items()}
    returns = result['strategy']
    mean, std = returns.mean(axis=0), returns.std(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe = sr_mult * mean / std
    result.update(timestamps=index[selection].to_numpy(), strategies=keys.copy(),
                  paths=prepared['paths'].copy(), annual_return=sr_mult**2*mean,
                  annual_volatility=sr_mult*std, sharpe=sharpe)
    if view:
        _plot_portfolio(result, view_weights)
        for i, name in enumerate(result['paths']):
            print(f"{name}: return={result['annual_return'][i]:.4g}, "
                  f"vol={result['annual_volatility'][i]:.4g}, Sharpe={sharpe[i]:.4g}")
    if n_boot:
        if not 1 <= block_size <= len(returns):
            raise ValueError('block_size must be between 1 and the number of rows')
        if not np.isfinite(sharpe).all():
            raise ValueError('bootstrap validation requires finite path Sharpes')
        result['valid'] = valid_strategy(returns, n_boot, sr_mult, alpha=alpha,
                                       alpha_n=alpha_n, block_size=block_size, view=view)
    return result


def _plot_portfolio(result, view_weights):
    import matplotlib.pyplot as plt

    ts = result['timestamps']
    for field, title, cumulative in [
        ('strategy', 'Cumulative returns', True),
        ('lev', 'Portfolio gross leverage', False),
        ('net_lev', 'Portfolio net leverage', False),
        ('counts', 'Strategies with nonzero allocation', False),
    ]:
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        values = result[field]
        if cumulative:
            values = values.cumsum(axis=0)
        for p, label in enumerate(result['paths']):
            ax.plot(ts, values[:, p], label=label)
        ax.set_title(title)
        ax.grid(alpha=.25)
        ax.legend()

    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.hist(result['strategy'].ravel(), bins=50, density=True)
    ax.set_title('Return distribution')
    ax.grid(alpha=.25)

    if view_weights:
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        colors = plt.get_cmap('tab20', len(result['strategies']))
        styles = ['-', '--', ':', '-.']
        for j, key in enumerate(result['strategies']):
            for p, path in enumerate(result['paths']):
                ax.plot(ts, result['weights'][:, j, p], color=colors(j),
                        linestyle=styles[p % len(styles)], label=f'{key} — {path}')
        ax.set_title('Portfolio allocation weights')
        ax.grid(alpha=.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.show()


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
    from scipy.stats import norm
    import matplotlib.pyplot as plt
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
    
    if view:
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)

        ax.hist(
            b_samples, bins=50, density=True,
            color='tab:blue', alpha=0.55, label='Bootstrap samples',
        )

        # Add symmetric padding, including when all samples are negative.
        padding = max(0.1 * np.ptp(b_samples), 0.1 * scale_b, 0.01)
        x_pdf = np.linspace(
            b_samples.min() - padding, b_samples.max() + padding, 400
        )
        if scale_b > 0:
            ax.plot(
                x_pdf, norm.pdf(x_pdf, loc=mean_b, scale=scale_b),
                color='tab:orange', linewidth=2, label='Gaussian fit',
            )

        ax.axvline(
            0, color='tab:red', linestyle='--', linewidth=1.2,
            label='Zero Sharpe',
        )
        ax.set_title(f'Bootstrap Sharpe distribution — worst path (path_{idx_lowest_sr + 1})')
        ax.set_xlabel('Annualized Sharpe ratio')
        ax.set_ylabel('Density')
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.25)
        ax.legend()
        plt.show()

    return valid


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

    def portfolio_post_process(
        self, pct_fee=0., seq_fees=False, sr_mult=np.sqrt(250),
        n_boot=1000, block_size=20, alpha=0.05, alpha_n=1000,
        view_weights=True, use_sw=True, multiplier=1, normalize_sw=False,
        start_date='', end_date='', resample_to='B', *, view=True,
    ):
        """Prepare and analyze the portfolio using the pandas implementation.

        Existing parameter names and the {'strategy', 'lev'} DataFrame return
        format are preserved. The pandas helper calls strategy allocations 'pw';
        weight_attr='sw' maps preparation to this package's Data.sw attribute.

        Returns and original-observation fee bases are SUMMED per resampling bin.
        Allocation weights and asset leverage use the LAST observation in the bin.
        Missing strategies receive zero allocation (no forward filling). Gross
        leverage uses absolute strategy allocations and is before asset netting.

        view=False disables plots and printed summaries. n_boot=0 independently
        disables bootstrap validation. The default bootstrap settings are retained.
        Set sr_mult for the output frequency, e.g. np.sqrt(52) for weekly data.

        To reuse preparation for several analyses, call prepare_portfolio and
        analyze_portfolio directly from this module with weight_attr='sw'.
        """
        if not self:
            print('No paths to process!')
            return

        # Preparation is independent of the fee rate and allocation settings.
        prepared = prepare_portfolio(
            self,
            resample_to=resample_to,
            seq_fees=seq_fees,
            weight_attr='sw',
        )

        # Map the existing sw option names to the helper's pw option names.
        result = analyze_portfolio(
            prepared,
            pct_fee=pct_fee,
            use_pw=use_sw,
            normalize_pw=normalize_sw,
            multiplier=multiplier,
            start_date=start_date,
            end_date=end_date,
            sr_mult=sr_mult,
            view=view,
            view_weights=view_weights,
            n_boot=n_boot,
            block_size=block_size,
            alpha=alpha,
            alpha_n=alpha_n,
        )

        # Preserve the public return format used by strategy_assembly and callers.
        index = pd.DatetimeIndex(result['timestamps'])
        columns = result['paths']
        return {
            'strategy': pd.DataFrame(result['strategy'], index=index, columns=columns),
            'lev': pd.DataFrame(result['lev'], index=index, columns=columns),
        }


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
