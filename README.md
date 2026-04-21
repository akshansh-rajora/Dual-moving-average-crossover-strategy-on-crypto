# Crypto Dual SMA Crossover Backtester

A rigorous quantitative backtesting framework for a dual moving average crossover strategy on cryptocurrency markets. Built from scratch in Python — no black-box backtest libraries.

## Strategy Overview

**Signal logic:**
- **BUY** when the fast SMA (20-day) crosses *above* the slow SMA (50-day) — "golden cross"
- **SELL** when the fast SMA crosses *below* the slow SMA — "death cross"
- Hold 100% in asset while long; hold cash (flat) otherwise

This is a **trend-following** strategy — it profits in sustained directional moves and loses in choppy sideways markets. Crypto's high-trending nature makes it a strong fit.

## Project Structure

```
crypto_backtest/
├── backtest.py          # Core engine — signals, backtest, metrics, walk-forward
├── optimise.py          # Grid-search over SMA parameter space
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python backtest.py        # Run the main backtest → prints report + saves chart
python optimise.py        # Find best SMA parameters (optional)
```

## Performance Metrics Explained

Every metric here is something you will be asked about in quant interviews.

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **Sharpe Ratio** | (Ann. Return − Risk-Free) / Ann. Vol | Return per unit of total risk. >1 is acceptable, >2 is strong for a systematic strategy |
| **Max Drawdown** | Max peak-to-trough decline in equity | Worst-case loss from any high point. Critical for position sizing |
| **Calmar Ratio** | Ann. Return / \|Max Drawdown\| | Return per unit of drawdown risk. >0.5 is respectable |
| **Win Rate** | % of trades that are profitable | Alone means nothing — must be paired with profit factor |
| **Profit Factor** | Gross profit / Gross loss | >1.2 means the strategy makes more than it loses |
| **Annualised Vol** | Daily std dev × √365 | Normalised measure of risk; feeds into Sharpe |

## Walk-Forward Validation

The backtest is validated using **walk-forward analysis** — the gold standard for avoiding overfitting:

1. The full dataset is split into 4 sequential folds
2. For each fold, the *first 60%* is in-sample (train), the *last 40%* is out-of-sample (test)
3. Strategy is evaluated only on the out-of-sample portion
4. Consistent OOS Sharpe across folds → strategy is robust, not curve-fitted

A strategy that looks great in-sample but collapses out-of-sample is worthless. This test distinguishes the two.

## Key Design Decisions (For Interviews)

**Why vectorised, not loop-based?**
Vectorised backtests (using numpy/pandas operations on arrays) are orders of magnitude faster than looping over each bar. In production, you need to run thousands of parameter combinations — a loop-based backtest is a bottleneck.

**Why use log returns?**
Log returns are time-additive: `log(P_t/P_0) = Σ log(P_i/P_{i-1})`. This makes cumulative return calculations exact and avoids compounding errors over long periods.

**Why shift position by 1 day?**
`df["strategy_return"] = df["position"].shift(1) * df["asset_return"]`
The position is determined at the *close* of day t, and earns the return of day t+1. Without this shift, you'd be using tomorrow's price to decide today's trade — a look-ahead bias that inflates performance.

**Why no transaction costs in the base run?**
The base run shows the theoretical maximum. You can then add a cost assumption (e.g. 0.1% per trade) and observe the performance decay. For crypto on major exchanges, taker fees are typically 0.05–0.1%.

## Extensions (Good for Interview Discussion)

- **Add transaction costs**: multiply each trade return by `(1 - fee)²`
- **Add slippage model**: fill at `close * (1 + slippage)` for buys
- **Add position sizing**: e.g. Kelly criterion or fixed fractional
- **Try EMA instead of SMA**: exponential weighting reacts faster to price changes
- **Add a second asset**: BTC + ETH pair, run strategy on both and combine into a portfolio

## References

- Sharpe, W. (1994). *The Sharpe Ratio*. Journal of Portfolio Management.
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*. Wiley.
- Chan, E. (2009). *Quantitative Trading*. Wiley.
