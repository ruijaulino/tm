# Package review — 18 September 2026

The architecture has a useful separation: Data/Dataset → transforms → predictive models → allocation → cross-validation → portfolio analysis. The highest-value next step is to make that pipeline reliable and reproducible before adding estimators or optimizing it.

Scope: package-wide source review covering containers, orchestration, transformations, allocation, model families, research helpers, post-processing, packaging, and notebook inventory. All 30 Python modules parse. Nine notebooks were inventoried, not executed. This is not a full mathematical validation of every estimator.

The working tree already contained changes to allocation.py and post_process.py; findings refer to that current working tree. No implementation files were changed.

## Highest-priority corrections

| Priority | Finding and evidence | Proposed change |
|---|---|---|
| P1 | Feature values can be assigned incorrect labels. `Data._get_columns` uses `np.isin`, preserving incoming order while setting requested labels (`tm/containers.py:168–186`). Reproduced: request x1,x2 from x2,x1 produces labels x1,x2 with values still 20,10. | Resolve each requested name to its exact source position; reject missing/duplicate names. Align target weights and returns using the same schema. |
| P1 | `Model.evaluate/live` checks that supplied columns are in required columns, the reverse of the advertised check (`tm/model.py:116,145`). It rejects extra columns while allowing required columns to be absent. | Validate required-column coverage, then reorder consistently. Include reordered, extra, missing and multi-target columns in contract tests. |
| P1 | Portfolio resampling sums returns but keeps the last strategy allocation, then multiplies them (`tm/post_process.py:173,247–248`). Reproduced: returns [0.1,0.2], allocations [1,0] become 0 instead of 0.1 in a daily bin. | Apply allocations and calculate net P&L at original observation frequency before aggregation. For reusable preparation, retain source-level information needed by normalization/multiplier options. |
| P1 | Sequential fee basis uses changes in underlying weights only (`tm/post_process.py:140`), ignoring allocation changes. Signed allocations also reverse the sign of fees (`:247–248`). Reproduced: zero returns with allocation -1 and fee 0.01 yield positive 0.01; allocation 0→1 with unchanged underlying weights incurs zero sequential fees. | Define execution semantics explicitly. For portfolio turnover, compute changes in executed positions allocation×asset weight; charge nonnegative costs. Decide whether strategies are executed separately or netted by asset. Include initial positions and sequence boundaries. |
| P1 | Gaussian HMM sampling passes posterior variance `vn` as the normal scale (`tm/base/hmm.py:382`, with repeated patterns at 534,1256,1583 and fallback branches). | Pass sqrt(variance), including cached posterior/prior parameters; audit regression-intercept samplers too. Test conditional sample mean and variance against their analytic values. |
| P1 | `uHMMEmissions.posterior_predictive` sums component variances but omits between-state dispersion (`tm/base/hmm.py:804–811`). | Use sum(p×(variance+mean²))−mixture_mean². Example: equally likely states with means -1,+1 and variance 1 need mixture variance 2, not 1. Audit all emission implementations against a common moment contract. |
| P1 | Repeated adapter fits append models without clearing them (`tm/base/model_converters.py:32,92`); predictions still index the original models. Reproduced: fit means 1 then 9 leaves two models and first model mean 1. State models also retain distributions for states absent from a subsequent fit. | Make estimate replace learned state by default; introduce explicit warm-start behavior only where supported. Reset allocation calibration too: it currently recalibrates through old quantiles and centering. |
| P1 | `Optimal(max_w=...)` ignores the parameter and clips to ±1 (`tm/allocation/allocation.py:141`). Reproduced max_w=0.1 returning 1. `Model.set_external_multiplier` calls an absent allocator method. | Make limits functional and documented; validate finite bounds and define per-asset versus total leverage limits. Implement or remove unsupported multiplier controls. |
| P1 | Concrete runtime failures: small/unfitted LinRegr references undefined `m` (`tm/base/lr.py:56`); live inference with t references undefined `t` (`tm/model.py:156`); MLR empty-cluster fallback references undefined `X` (`tm/base/mlr.py:131`). | Fix branches and add narrow regression tests. Decide whether insufficient training data produces an explicit error or documented neutral prediction. |

## Backtest and data integrity

1. **Distinguish cross-validation from chronological simulation.** `cvbt` defaults to training on both sides of each held-out block; strategy assembly similarly trains on every other fold (`tm/workflows.py:75`, `tm/post_process.py:676`). This can be an intentional research CV scheme, but does not simulate historical deployment. Provide explicit walk-forward and blocked-CV modes, make inner model selection follow the same temporal rules, and configure purge/embargo from the label horizon rather than random trimming alone.
2. **Track evaluated observations explicitly.** Sequential CV leaves the initial fold's initialized zero returns in the output. Post-processing treats every source row as available. Add an evaluated mask so untested periods do not become apparent zero-return observations in metrics. Clear previous results on reruns; `ModelSet.evaluate` currently multiplies existing `sw` again (`tm/model.py:443`).
3. **Enforce input contracts at boundaries.** Require compatible lengths, finite floating arrays, unique column names and an explicit timestamp policy. `before/after` rely on sorted timestamps, while `from_df` does not enforce ordering. Validate folds against available observations and minimum training size. Preserve deliberate separate-sequence semantics when stacking.
4. **Respect sequence boundaries in every sequential model.** HMM explicitly uses `msidx`, but rolling filters operate on the entire concatenated input; `FastTFHMM` drops kwargs when forwarding predictions. Reset filter/history state at each sequence boundary. Test that adding an unrelated preceding sequence cannot change a sequence's predictions.
5. **Normalize datetime units explicitly.** `datetime_to_int` divides the raw integer representation by 1e9, assuming nanoseconds. Normalize resolution and define timezone/subsecond behavior rather than depending on pandas storage units.
6. **Define return and execution conventions.** Document whether y is a contemporaneous or forward return, when x becomes available, and whether results are additive P&L or compounded returns. Keep the terminal placeholder out of economic observations; eventually replace sentinel -1 with an explicit forecast interface.

## Model and numerical consistency

- Standardize predictive mean `(n,p)` and covariance `(n,p,p)` at an adapter boundary. `RollMean` returns a two-dimensional covariance; `AsSingle` feeds a one-dimensional feature to models trained with a matrix; `AsUnivariate` assumes vector outputs although uGaussian returns matrix/tensor outputs. `StateGaussian` requires vector z while Data supplies a matrix, and accesses z.shape before its None fallback. Build a compatibility matrix of exported models and wrappers.
- Protect constant features in ScaleTransform from division by zero. Convert numeric inputs to float before in-place transforms so integer datasets cannot truncate scaled values.
- Check covariance symmetry, finiteness and conditioning before allocation. Add configurable shrinkage/variance floors and rank-aware least squares where appropriate. LinRegr, Laplace WLS and ConditionalGaussian currently have fragile rank/singularity paths.
- Separate an actual covariance from a matrix used to calculate a desired allocation. `RollInvMultiVol` returns scale×correlation, which is generally nonsymmetric. Such an operator should not be passed through covariance transformations under a covariance contract.
- Use the newer BayesianLinearRegression's input validation, fitted checks and convergence diagnostics as a pattern. Choose a supported Bayesian API and document/deprecate alternatives rather than retaining multiple overlapping implementations indefinitely.
- Expose convergence diagnostics consistently for mixtures and Gibbs samplers. Add deterministic simulation-based moment checks, likelihood/convergence checks, and posterior predictive checks before relying on model comparisons.
- Thread a local random generator/seed through splits, models and bootstrap. Strategy assembly already uses default_rng; most other stochastic paths use global state. Record fold boundaries and seed configuration with outputs.
- Treat `valid_strategy` as a specific approximate diagnostic: it fits a Gaussian to bootstrap Sharpe samples and evaluates a tail probability. Return the underlying estimates and configuration, guard degenerate variance, test calibration on synthetic null data, and document how alpha_n relates to the actual search history.

## Packaging, API and maintainability

- Declare runtime dependencies: numpy, pandas, scipy, matplotlib, scikit-learn, numba and tqdm are imported but pyproject.toml declares none. Specify supported Python versions and test an installed wheel in a fresh environment.
- Move plotting imports behind plotting functions or a visualization extra. The eager top-level exports make even Data depend on model/plotting dependencies. Prefer explicit exports and a smaller import graph.
- Add a dedicated tests directory and CI. Existing test/dev functions are embedded demonstrations, several using outdated APIs; there is no conventional automated suite or CI configuration in this checkout.
- Replace public-input assertions with descriptive ValueError/TypeError exceptions. Avoid mutable defaults, broad bare exceptions, print-based diagnostics, and undocumented in-place mutations.
- Separate pure calculation, plotting and serialization. Consolidate post_process_prev.py and duplicated rolling helpers; move experiments, long commented alternatives and demo blocks out of shipped modules.
- Make save/load record package version, model configuration, feature schema, training interval and dependencies. Document pickle as trusted-input-only. Avoid embedding the entire estimation dataset by default if it is not needed for deployment.
- Expand README with one synthetic, executable fit→CV→analysis→live example; array/schema contracts; timing conventions; model compatibility; installation; and reproducibility. Promote selected notebooks to maintained examples.

## Performance work after correctness

Measure representative single/multi-asset workloads first: fit, predict, nested CV, memory, and portfolio preparation separately. Likely targets are repeated deep copies (especially stored training datasets), repeated transform/filter work, Python HMM prediction loops, and dense covariance tensors for diagonal models. Preserve the useful existing separation between portfolio preparation and analysis, and the prefix-sum bootstrap optimization, while adding reference-result equivalence checks. Parallel execution should come after random-state isolation and deterministic fold definitions.

## Suggested implementation order

1. **Correctness patch:** schema ordering/coverage, failing branches, refit behavior, allocator limits, HMM moments, and source-frequency portfolio accounting. Add focused regression tests alongside fixes.
2. **Reliable evaluation:** explicit split modes, evaluated masks, sequence-boundary tests, deterministic RNG, and a shared prediction contract.
3. **Usable release:** dependency metadata, clean-install CI, documented examples, public API cleanup and versioned persistence metadata.
4. **Measured optimization:** benchmarks, memory/copy reductions and targeted acceleration.

## Validation performed

- Parsed all 30 Python modules successfully.
- Ran seven isolated probes against actual AST-loaded definitions using available NumPy and pandas. Outputs reproduced small-sample prediction failure, feature mislabelling, stale adapter refit, ignored max_w, resampling allocation error, signed fee reversal, and missing allocation-turnover costs.
- Full package import failed with `ModuleNotFoundError: matplotlib`; this runtime also lacks scipy, scikit-learn, numba, tqdm and pytest. No dependencies were installed, no complete integration suite ran, and no estimator performance claims are made.
- `audit_checks.py` preserves the probes and explicitly bypasses optional imports; it is evidence for individual functions, not a substitute for package integration tests.
