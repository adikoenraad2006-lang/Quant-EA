# Quant-EA — a four-layer strategy tester

A retail-strategy testing system in Python: download the data, implement the
popular strategy library, sweep every setting across every asset, filter the
results honestly, then stress what is left.

```
layer1_data_strategies.py   data + the strategy library (47 families, 348 configs)
layer2_sweep.py             every config x asset, six filters, the funnel report
layer3_robustness.py        parameter sensitivity + bootstrap stress test
layer4_cross_sectional.py   cross-sectional momentum, scored the same way
```

## Run it

```bash
pip install -r requirements.txt
python run_all.py            # or run the four scripts in order
```

Layer 1 caches raw OHLCV under `data/`, so re-runs are instant. Everything else
writes to `results/`:

| file | what it holds |
| --- | --- |
| `sweep_results.csv` | one row per backtest, with the six filter flags |
| `survivors.csv` | the rows that passed all six |
| `param_sensitivity.csv` | per-family spread of OOS Sharpe across the grid |
| `bootstrap_results.csv` | percentile Sharpe + worst-case drawdown per survivor |
| `cross_sectional_momentum.csv` | one row per lookback |
| `cross_sectional_windows.csv` | per walk-forward window |

`python tests/test_pipeline.py` checks the machinery: position values, the
one-bar lag, truncation invariance (no look-ahead) for all 348 configs, the cost
arithmetic, the walk-forward split, the filters, and that the cross-sectional
book scores flat on random walks.

Every script takes `--synthetic`, which swaps in random-walk prices so the
pipeline can be exercised without network access. **Anything measured on
synthetic data is a smoke test, not a result** — the scripts say so loudly in
their output.

## Running it on Tickstory data

Point every layer at a directory of exports instead of yfinance:

```bash
python run_all.py --source tickstory --data-dir ./tickstory
```

Each script takes the same flags, so you can also run them one at a time. The
reader sniffs each file rather than assuming a template, and handles the shapes
Tickstory actually writes:

| export | looks like |
| --- | --- |
| tick csv | `2015.01.02 00:00:00.123,1.20998,1.21024,0.75,1.50` |
| M1 csv (MT4) | `2015.01.02,00:00,1.21000,1.21010,1.20990,1.21005,42` |
| MT5 csv | tab-separated with a `<DATE>	<TIME>	<OPEN>...` header |
| generic | `2015-01-02 00:00:00,1.21000,...` or `20150102 000000;...` |
| MT4 `.hst` | binary, version 400 or 401 |

`.gz` and `.zip` are read in place. Symbols come from filenames, with the usual
noise stripped — `EURUSD_M1_2015-2025.csv` and `GBPJPY-M15.txt.gz` become
`EURUSD` and `GBPJPY`. Daily bars are cached under `data/tickstory/`, so the
slow parse happens once.

**Export M1 bars, not raw ticks.** This is a daily-bar system, so ticks buy you
nothing and a decade of one pair is hundreds of millions of rows. Tick files do
work — they are streamed in chunks and aggregated as they go, so memory stays
flat — but M1 gets you the same daily bars in seconds.

### Three things to get right

**1. Where the day is cut.** Tickstory stamps rows in whatever offset you chose
at export (Dukascopy source data is UTC), and a daily bar has to be cut
somewhere. The default is `--session-close-utc 22` — 17:00 New York, the FX
convention. Cutting at UTC midnight instead (`--session-close-utc 0`) splits the
Sunday-evening open into a stub bar of its own and shifts every daily close by
two hours, which moves every indicator that reads Close. If you exported with a
broker offset already applied (say GMT+2), pass the hour in *that* clock.

**2. Costs.** The built-in table is equity-shaped; FX symbols fall through to a
spread estimate inferred from the symbol (1bp for majors, 2bp for crosses,
2.5bp for metals, one way). Those are placeholders. Use your own:

```bash
echo "symbol,bps
EURUSD,0.6
GBPJPY,2.4
XAUUSD,3.0" > costs.csv
python run_all.py --source tickstory --data-dir ./tickstory --costs costs.csv
```

**3. Volume is tick volume.** Tickstory reports tick counts, not traded size.
The six volume strategies will run on it and it is a reasonable activity proxy
in FX, but it is not the quantity the equity side of the universe reports, so
volume-family results are not comparable across the two sources. If your export
has no volume column at all, the reader counts ticks per day and says so.

### What this does not model for FX

**Swap / rollover.** Holding an FX position overnight earns or pays the interest
differential, every night. Several of these families hold for weeks, so on a
carry pair that is not a rounding error — it can be the whole result, in either
direction. The cost model charges spread on turnover and nothing else, so an FX
run here measures the price signal only.

**Overlapping legs.** Layer 4 ranks assets against each other, which assumes the
cross-section is made of separable things. EURUSD, EURJPY and GBPJPY share
currency legs, so a long-top-third / short-bottom-third book over FX pairs can
end up concentrated in one currency rather than diversified. It still runs; read
the result as currency momentum with that caveat, not as the equity-style
cross-section the layer was written for.

**Session count.** Sharpe is annualised at 252 (`ANNUALIZATION` in
`qea/config.py`) and FX runs ~260 sessions a year, so FX Sharpes here are
understated by about 1.5%. `--train-bars` and `--test-bars` are counts of bars,
not calendar time, for the same reason.

`python tests/test_tickstory.py` writes a file in every supported format from
one known set of bars and checks they all roll up to the same daily OHLCV,
covers both session cuts, both `.hst` versions, compression, symbol naming, and
finishes by running all 348 configs over the result.

## Layer 1 — data and the strategy library

Daily OHLCV from yfinance, `auto_adjust=True`, 2010-01-01 to 2025-01-01, for 29
liquid assets: index ETFs (SPY, QQQ, IWM, DIA), sector ETFs (XLK, XLF, XLE, XLV,
XLI, XLU, XLY, XLP), commodities/rates/international (GLD, USO, TLT, HYG, EFA,
EEM, EWZ), crypto (BTC-USD, ETH-USD) and large caps (AAPL, MSFT, NVDA, TSLA,
AMZN, GOOGL, META, JPM). Anything with under 500 bars is skipped. Open, High,
Low, Close and Volume are all kept, so indicators that need them work.

Each strategy is `f(df, **params) -> pd.Series` of daily positions in
`{-1, 0, 1}` — long, flat, short. The `@strategy` decorator applies the one-bar
execution lag centrally, so today's position is built only from data up to
yesterday's close. Every family is tagged with a category:

| category | families |
| --- | --- |
| trend (19) | MA crossover, time-series momentum, ROC momentum, MACD, Donchian breakout, Bollinger breakout, Supertrend, Parabolic SAR, ADX trend, Ichimoku, linreg slope, Aroon, Vortex, TRIX, Hull MA, KAMA, Turtle, dual momentum, Elder Ray |
| meanrev (12) | RSI revert, Bollinger revert, z-score revert, stochastic, CCI, Williams %R, Keltner revert, VWAP revert, percent-B, Connors RSI, Ultimate Oscillator, gap fade |
| volume (6) | OBV trend, Chaikin money flow, money flow index, volume surge, Force Index, Chaikin oscillator |
| volatility (3) | ATR breakout, volatility breakout, squeeze breakout |
| pattern (4) | engulfing, three-bar reversal, higher-highs/lows, pivot bounce |
| composite (3) | MACD+RSI confirmation, triple-screen, chandelier |

`build_configs()` expands each family over a small grid and returns
`(name, function, params, category, family)` tuples — **348 configs**, which is
`348 x 29 = 10,092` backtests.

## Layer 2 — the sweep and the funnel

For every (config x asset): in-sample Sharpe, out-of-sample Sharpe from the
stitched walk-forward series, out-of-sample max drawdown, and trade count. All
of it lands in `results/sweep_results.csv`.

A strategy survives only if it passes **all six**:

1. OOS max drawdown better than -35%
2. OOS Sharpe above 0.5
3. OOS Sharpe below 2.5 (above that the asset did the work, not the strategy)
4. OOS Sharpe no more than ~30% above in-sample (a big gap is the overfit signature)
5. at least 30 trades
6. in-sample Sharpe positive

The funnel report prints the attrition — total backtests, how many had a
positive OOS Sharpe, how many cleared 0.5, how many survived all six — then
survival rate by category and by family with mean OOS Sharpe, and a
top-survivors table.

### Assumptions filled in here

The screenshot for the first half of Layer 2 wasn't in the set, so the
walk-forward and cost mechanics are inferred from what the later prompts
reference. Both are configurable in `qea/config.py`:

**Walk-forward split.** The first 1,260 bars (~5 years) of each asset are the
in-sample block. From there the tester steps forward in 252-bar (~1 year)
windows; each window's returns are appended to one stitched out-of-sample
series, and every reported OOS number comes from that series. None of these
strategies fits parameters on the training data, so the walk-forward is a pure
time split — the point is that the OOS period is untouched by config selection,
not that anything is re-estimated.

**Transaction costs.** A position of ±1 is fully long/short the asset. Every
change pays `|pos_t - pos_t-1| * cost_bps / 10,000` of notional, so a long→short
flip pays twice the one-way cost. Costs are per asset: 2bp for index ETFs, 3-4bp
for sector/large-cap, 5-7bp for the thinner ETFs (HYG, USO, EWZ), 20bp for
crypto.

## Layer 3 — robustness

**Parameter sensitivity.** Each config is first collapsed to its mean OOS Sharpe
across assets, then per family the report gives the mean, the standard deviation
across settings, and the fraction of settings with a positive OOS Sharpe. A
family that only works on one exact setting and falls apart on the others is a
red flag. A tight spread with a high positive fraction means the edge does not
rest on one magic number.

**Bootstrap stress test.** For each top survivor, its OOS daily returns are
resampled a few hundred times (default 200) to get a distribution of outcomes
instead of the single one that happened: 5th/50th/95th percentile Sharpe, median
and worst-case drawdown. Survivors are flagged `solid` or `fragile` on their
worst-case drawdown.

Note on method: `--method resample` (the default) draws with replacement, so
both Sharpe and drawdown vary. `--method shuffle` permutes the actual returns —
that leaves the Sharpe numerically unchanged by construction, since mean and
standard deviation don't care about order, and only the equity path and its
drawdown move. The script says so in its output.

## Layer 4 — cross-sectional momentum

Same universe, same minimum-history filter. Every 21 trading days all assets are
ranked by trailing return at three lookbacks — 3 months, 6 months, and the
standard 12-months-minus-the-most-recent-month — long the top third, short the
bottom third, equal weight, held until the next rebalance. Weights drift with
the assets in between, so per-asset costs are charged only on real turnover.
Long leg sums to +1 and short leg to -1, i.e. dollar-neutral at 2x gross.

It is scored exactly like the main tester — the same in-sample block, the same
stitched walk-forward OOS Sharpe and drawdown — so the number is directly
comparable to the single-asset momentum backtests from the sweep, which the
report prints side by side along with the drawdowns as they come out and a
regime check (how much of the result rests on the single best year).

## What this does not claim

- The six filters are applied to out-of-sample results, so the survivors are
  selected on the same data used to judge them. Survivor statistics are
  therefore optimistic; the funnel's attrition rate is the honest number, not
  any individual survivor's Sharpe.
- Costs are a flat per-asset estimate, not a market-impact model, and there is
  no borrow cost or financing on the short side.
- No slippage on gaps, no position sizing, no risk overlay, no walk-forward
  re-fitting.

## Network note

The yfinance path needs `query1.finance.yahoo.com` and `fc.yahoo.com`. Both are
blocked by the egress policy of the sandbox this repo was authored in, so no
real-data run was performed here — the pipeline was validated with `--synthetic`
and against generated files in each Tickstory format. Run the scripts on a
machine with normal network access for the yfinance universe. The Tickstory path
reads local files and needs no network at all.
