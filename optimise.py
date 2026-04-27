"""
SMA Parameter Optimisation
===========================
Grid-searches fast/slow SMA combinations and ranks by Sharpe ratio.
Run this AFTER backtest.py to find the best parameters — then re-run
the main backtest with those values.

WARNING: This is IN-SAMPLE optimisation. Always validate winners with
the walk-forward module in backtest.py before trusting the results.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from backtest import fetch_data, compute_signals, run_backtest, compute_metrics

TICKER       = "BTC-USD"
START        = "2018-01-01"
END          = "2024-12-31"
INITIAL_CAP  = 10_000
RISK_FREE    = 0.04
TRADING_DAYS = 365

# Grid to search
FAST_RANGE = range(5,  51, 5)   # 5, 10, 15, ..., 50
SLOW_RANGE = range(20, 201, 10) # 20, 30, ..., 200


def optimise():
    df_raw = fetch_data(TICKER, START, END)
    results = []

    total = len(FAST_RANGE) * len(SLOW_RANGE)
    done  = 0

    for fast in FAST_RANGE:
        for slow in SLOW_RANGE:
            if fast >= slow:
                continue
            try:
                df_sig = compute_signals(df_raw, fast, slow)
                df_bt  = run_backtest(df_sig, INITIAL_CAP)
                m      = compute_metrics(df_bt, INITIAL_CAP, RISK_FREE, TRADING_DAYS)
                results.append({
                    "fast":         fast,
                    "slow":         slow,
                    "sharpe":       round(m["strat_sharpe"], 3),
                    "ann_return":   round(m["strat_ann_return"] * 100, 2),
                    "max_dd":       round(m["strat_max_dd"] * 100, 2),
                    "calmar":       round(m["strat_calmar"], 3),
                    "n_trades":     m["n_trades"],
                })
            except Exception:
                pass
            done += 1

    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print("\n── Top 15 SMA combinations by Sharpe ────────────────")
    print(df_res.head(15).to_string(index=False))
    print()
    print(f"  Best: SMA {df_res.iloc[0]['fast']:.0f}/{df_res.iloc[0]['slow']:.0f}"
          f"  Sharpe={df_res.iloc[0]['sharpe']:.2f}"
          f"  Ann.Return={df_res.iloc[0]['ann_return']:.1f}%"
          f"  MaxDD={df_res.iloc[0]['max_dd']:.1f}%")

    df_res.to_csv("optimisation_results.csv", index=False)
    print("  Full results saved → optimisation_results.csv\n")

    return df_res


if __name__ == "__main__":
    optimise()
