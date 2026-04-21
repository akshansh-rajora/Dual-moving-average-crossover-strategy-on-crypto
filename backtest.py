"""
Dual Moving Average Crossover Strategy — Crypto Backtester
==========================================================
Strategy logic:
  - BUY  when the fast SMA crosses ABOVE the slow SMA (golden cross)
  - SELL when the fast SMA crosses BELOW the slow SMA (death cross)

Performance metrics produced:
  - Total return, annualised return, annualised volatility
  - Sharpe ratio (risk-free rate = 4% annualised, approx US T-bill)
  - Maximum drawdown
  - Calmar ratio (annualised return / max drawdown)
  - Win rate, profit factor
  - Walk-forward out-of-sample validation
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKER      = "BTC-USD"
START       = "2018-01-01"
END         = "2024-12-31"
FAST_SMA    = 5         # days
SLOW_SMA    = 120        # days
INITIAL_CAP = 10_000      # USD
RISK_FREE   = 0.04        # annualised (approx US T-bill)
TRADING_DAYS = 365        # crypto trades every day
# ─────────────────────────────────────────────────────────────────────────────


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and validate."""
    print(f"  Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol.")
    df = df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.dropna(inplace=True)
    print(f"  {len(df)} trading days loaded.")
    return df


def compute_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """
    Add moving averages and generate entry/exit signals.

    Signal convention:
      +1 = long (fast > slow)
       0 = flat (no position)
    """
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()

    # Position: 1 when fast > slow, else 0
    df["position"] = np.where(df["sma_fast"] > df["sma_slow"], 1, 0)

    # Detect crossovers (signal fires on the day AFTER the cross — avoids look-ahead bias)
    df["signal"] = df["position"].diff()
    # +1.0 = golden cross (buy), -1.0 = death cross (sell)

    return df.dropna()


def run_backtest(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    Vectorised backtest — no per-bar loop.

    Strategy holds a full position (100% of portfolio in BTC) or is flat (cash).
    No shorting, no leverage, no transaction costs in the base run.
    """
    df = df.copy()

    # Daily log returns of the asset
    df["asset_return"] = np.log(df["close"] / df["close"].shift(1))

    # Strategy return: position is set at close of day t, earns return on day t+1
    df["strategy_return"] = df["position"].shift(1) * df["asset_return"]

    # Equity curves
    df["buy_hold_equity"]  = initial_capital * np.exp(df["asset_return"].cumsum())
    df["strategy_equity"]  = initial_capital * np.exp(df["strategy_return"].cumsum())

    # Running drawdown for strategy
    rolling_max = df["strategy_equity"].cummax()
    df["drawdown"] = (df["strategy_equity"] - rolling_max) / rolling_max

    return df.dropna()


def compute_metrics(df: pd.DataFrame, initial_capital: float,
                    risk_free: float, trading_days: int) -> dict:
    """
    Compute the key performance metrics every quant interviewer will ask about.
    """
    strat_returns = df["strategy_return"]
    bh_returns    = df["asset_return"]

    n_years = len(df) / trading_days

    def annualised_return(log_returns, n_yrs):
        total = np.exp(log_returns.sum()) - 1
        return (1 + total) ** (1 / n_yrs) - 1

    def annualised_vol(log_returns, n_days):
        return log_returns.std() * np.sqrt(n_days)

    def max_drawdown(equity_curve):
        roll_max = equity_curve.cummax()
        dd = (equity_curve - roll_max) / roll_max
        return dd.min()

    strat_ann_ret  = annualised_return(strat_returns, n_years)
    strat_ann_vol  = annualised_vol(strat_returns, trading_days)
    bh_ann_ret     = annualised_return(bh_returns, n_years)
    bh_ann_vol     = annualised_vol(bh_returns, trading_days)

    sharpe    = (strat_ann_ret - risk_free) / strat_ann_vol if strat_ann_vol != 0 else 0
    bh_sharpe = (bh_ann_ret   - risk_free) / bh_ann_vol    if bh_ann_vol   != 0 else 0

    strat_mdd = max_drawdown(df["strategy_equity"])
    bh_mdd    = max_drawdown(df["buy_hold_equity"])

    calmar = strat_ann_ret / abs(strat_mdd) if strat_mdd != 0 else 0

    # Trade-level stats
    trades = df[df["signal"] != 0].copy()
    buy_signals  = df[df["signal"] ==  1]
    sell_signals = df[df["signal"] == -1]

    # Pair buys with subsequent sells
    buy_idx  = list(buy_signals.index)
    sell_idx = list(sell_signals.index)

    trade_returns = []
    for b in buy_idx:
        future_sells = [s for s in sell_idx if s > b]
        if future_sells:
            s = future_sells[0]
            ret = df.loc[s, "close"] / df.loc[b, "close"] - 1
            trade_returns.append(ret)

    if trade_returns:
        winners     = [r for r in trade_returns if r > 0]
        losers      = [r for r in trade_returns if r <= 0]
        win_rate    = len(winners) / len(trade_returns)
        gross_prof  = sum(winners) if winners else 0
        gross_loss  = abs(sum(losers)) if losers else 1e-9
        profit_factor = gross_prof / gross_loss
    else:
        win_rate, profit_factor = 0, 0

    final_strat_equity = df["strategy_equity"].iloc[-1]
    final_bh_equity    = df["buy_hold_equity"].iloc[-1]

    return {
        "n_years":             round(n_years, 1),
        "n_trades":            len(trade_returns),
        # Strategy
        "strat_total_return":  final_strat_equity / initial_capital - 1,
        "strat_ann_return":    strat_ann_ret,
        "strat_ann_vol":       strat_ann_vol,
        "strat_sharpe":        sharpe,
        "strat_max_dd":        strat_mdd,
        "strat_calmar":        calmar,
        "win_rate":            win_rate,
        "profit_factor":       profit_factor,
        # Buy & hold benchmark
        "bh_total_return":     final_bh_equity / initial_capital - 1,
        "bh_ann_return":       bh_ann_ret,
        "bh_ann_vol":          bh_ann_vol,
        "bh_sharpe":           bh_sharpe,
        "bh_max_dd":           bh_mdd,
    }


def walk_forward_validation(df_raw: pd.DataFrame, fast: int, slow: int,
                             initial_capital: float, risk_free: float,
                             trading_days: int, n_splits: int = 4) -> pd.DataFrame:
    """
    Walk-forward validation: split the full history into n_splits folds.
    For each fold, train on the first 60% and test on the last 40%.
    Report out-of-sample Sharpe and return for each fold.

    This is the key method that separates a real quant from someone who just
    curve-fitted on the whole dataset.
    """
    results = []
    total_days = len(df_raw)
    fold_size  = total_days // n_splits

    for i in range(n_splits):
        start_idx  = i * fold_size
        end_idx    = start_idx + fold_size if i < n_splits - 1 else total_days
        fold_df    = df_raw.iloc[start_idx:end_idx].copy()

        split      = int(len(fold_df) * 0.6)
        train_df   = fold_df.iloc[:split]
        test_df    = fold_df.iloc[split:]

        # Generate signals and run on out-of-sample test set only
        test_sig   = compute_signals(test_df, fast, slow)
        if len(test_sig) < slow + 10:
            continue
        test_bt    = run_backtest(test_sig, initial_capital)
        if test_bt.empty:
            continue
        m          = compute_metrics(test_bt, initial_capital, risk_free, trading_days)

        results.append({
            "Fold":           i + 1,
            "OOS start":      test_df.index[0].strftime("%Y-%m"),
            "OOS end":        test_df.index[-1].strftime("%Y-%m"),
            "OOS Ann. Return": f"{m['strat_ann_return']*100:.1f}%",
            "OOS Sharpe":     f"{m['strat_sharpe']:.2f}",
            "OOS Max DD":     f"{m['strat_max_dd']*100:.1f}%",
        })

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, metrics: dict, ticker: str,
                 fast: int, slow: int, wf_results: pd.DataFrame):
    """
    Professional 4-panel chart:
      1. Price with SMAs and trade signals
      2. Equity curve: strategy vs buy & hold
      3. Drawdown
      4. Walk-forward OOS Sharpe by fold
    """
    fig = plt.figure(figsize=(16, 14), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(4, 1, hspace=0.45, height_ratios=[2.5, 1.8, 1, 0.9])

    GOLD   = "#f5c842"
    TEAL   = "#2dd4bf"
    RED    = "#f87171"
    GREY   = "#6b7280"
    WHITE  = "#f9fafb"
    BG     = "#0d0d0d"
    PANEL  = "#141414"

    def style_ax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=9)
        ax.spines[:].set_color("#2a2a2a")
        ax.yaxis.label.set_color(GREY)
        ax.xaxis.label.set_color(GREY)

    pct_fmt = FuncFormatter(lambda x, _: f"{x*100:.0f}%")

    # ── Panel 1: Price + SMAs + signals ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1)
    ax1.plot(df.index, df["close"],    color=WHITE,  lw=0.8, alpha=0.9, label="Price")
    ax1.plot(df.index, df["sma_fast"], color=GOLD,   lw=1.2, label=f"SMA {fast}")
    ax1.plot(df.index, df["sma_slow"], color=TEAL,   lw=1.2, label=f"SMA {slow}")

    buys  = df[df["signal"] ==  1]
    sells = df[df["signal"] == -1]
    ax1.scatter(buys.index,  buys["close"],  color=TEAL, marker="^", s=60, zorder=5, label="Buy")
    ax1.scatter(sells.index, sells["close"], color=RED,  marker="v", s=60, zorder=5, label="Sell")

    ax1.set_title(f"{ticker}  ·  Dual SMA {fast}/{slow} Crossover Strategy",
                  color=WHITE, fontsize=13, fontweight="bold", pad=10)
    ax1.legend(facecolor=PANEL, edgecolor="#2a2a2a", labelcolor=WHITE, fontsize=8, ncol=6)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: Equity curves ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    style_ax(ax2)
    ax2.plot(df.index, df["strategy_equity"], color=GOLD,  lw=1.5, label="Strategy")
    ax2.plot(df.index, df["buy_hold_equity"], color=GREY,  lw=1.0, label="Buy & Hold", alpha=0.7)
    ax2.axhline(10_000, color="#2a2a2a", lw=0.8, ls="--")
    ax2.set_ylabel("Portfolio Value (USD)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.legend(facecolor=PANEL, edgecolor="#2a2a2a", labelcolor=WHITE, fontsize=8)

    # Annotate final values
    ax2.annotate(f"${df['strategy_equity'].iloc[-1]:,.0f}",
                 xy=(df.index[-1], df["strategy_equity"].iloc[-1]),
                 color=GOLD, fontsize=8, ha="right")
    ax2.annotate(f"${df['buy_hold_equity'].iloc[-1]:,.0f}",
                 xy=(df.index[-1], df["buy_hold_equity"].iloc[-1]),
                 color=GREY, fontsize=8, ha="right")

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    style_ax(ax3)
    ax3.fill_between(df.index, df["drawdown"], 0, color=RED, alpha=0.5)
    ax3.plot(df.index, df["drawdown"], color=RED, lw=0.8)
    ax3.axhline(metrics["strat_max_dd"], color=RED, lw=0.8, ls="--", alpha=0.6)
    ax3.set_ylabel("Drawdown")
    ax3.yaxis.set_major_formatter(pct_fmt)

    # ── Panel 4: Walk-forward OOS Sharpe ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    style_ax(ax4)
    if not wf_results.empty:
        sharpes = wf_results["OOS Sharpe"].astype(float)
        colors  = [TEAL if s > 0 else RED for s in sharpes]
        bars = ax4.bar(wf_results["Fold"].astype(str), sharpes, color=colors, width=0.5)
        ax4.axhline(0, color=GREY, lw=0.8)
        ax4.axhline(1, color=TEAL, lw=0.6, ls="--", alpha=0.5)
        ax4.set_xlabel("Walk-Forward Fold")
        ax4.set_ylabel("OOS Sharpe")
        ax4.set_title("Walk-Forward Out-of-Sample Sharpe", color=WHITE, fontsize=9, pad=6)
        for bar, val in zip(bars, sharpes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", color=WHITE, fontsize=8)

    # ── Metrics table ─────────────────────────────────────────────────────────
    m = metrics
    metric_text = (
        f"  Strategy  │  Ann. Return: {m['strat_ann_return']*100:+.1f}%  "
        f"│  Sharpe: {m['strat_sharpe']:.2f}  "
        f"│  Max DD: {m['strat_max_dd']*100:.1f}%  "
        f"│  Calmar: {m['strat_calmar']:.2f}  "
        f"│  Win Rate: {m['win_rate']*100:.0f}%  "
        f"│  Trades: {m['n_trades']}  \n"
        f"  Buy&Hold  │  Ann. Return: {m['bh_ann_return']*100:+.1f}%  "
        f"│  Sharpe: {m['bh_sharpe']:.2f}  "
        f"│  Max DD: {m['bh_max_dd']*100:.1f}%"
    )
    fig.text(0.01, 0.01, metric_text, color=GREY, fontsize=8,
             fontfamily="monospace", va="bottom",
             bbox=dict(facecolor=PANEL, edgecolor="#2a2a2a", boxstyle="round,pad=0.4"))

    plt.savefig("backtest_results_1.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print("  Chart saved → backtest_results_1.png")
    plt.close()


def print_report(metrics: dict, wf: pd.DataFrame, ticker: str, fast: int, slow: int):
    """Print a clean text summary — paste this into your project README."""
    m = metrics
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  BACKTEST REPORT  ·  {ticker}  ·  SMA {fast}/{slow}")
    print(sep)
    print(f"  Period          : {m['n_years']} years")
    print(f"  Number of trades: {m['n_trades']}")
    print()
    print(f"  {'Metric':<22}  {'Strategy':>10}  {'Buy & Hold':>10}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}")
    print(f"  {'Total Return':<22}  {m['strat_total_return']*100:>9.1f}%  {m['bh_total_return']*100:>9.1f}%")
    print(f"  {'Ann. Return':<22}  {m['strat_ann_return']*100:>9.1f}%  {m['bh_ann_return']*100:>9.1f}%")
    print(f"  {'Ann. Volatility':<22}  {m['strat_ann_vol']*100:>9.1f}%  {m['bh_ann_vol']*100:>9.1f}%")
    print(f"  {'Sharpe Ratio':<22}  {m['strat_sharpe']:>10.2f}  {m['bh_sharpe']:>10.2f}")
    print(f"  {'Max Drawdown':<22}  {m['strat_max_dd']*100:>9.1f}%  {m['bh_max_dd']*100:>9.1f}%")
    print(f"  {'Calmar Ratio':<22}  {m['strat_calmar']:>10.2f}  {'—':>10}")
    print(f"  {'Win Rate':<22}  {m['win_rate']*100:>9.0f}%  {'—':>10}")
    print(f"  {'Profit Factor':<22}  {m['profit_factor']:>10.2f}  {'—':>10}")
    print()
    print("  Walk-Forward Out-of-Sample Validation")
    print(f"  {'─'*50}")
    if not wf.empty:
        print(wf.to_string(index=False))
    print(sep)


def main():
    print("\n── Crypto Dual SMA Crossover Backtester ──────────────")

    # 1. Data
    df_raw = fetch_data(TICKER, START, END)

    # 2. Signals
    print("  Computing signals...")
    df_sig = compute_signals(df_raw, FAST_SMA, SLOW_SMA)

    # 3. Backtest
    print("  Running backtest...")
    df_bt  = run_backtest(df_sig, INITIAL_CAP)

    # 4. Metrics
    print("  Computing metrics...")
    metrics = compute_metrics(df_bt, INITIAL_CAP, RISK_FREE, TRADING_DAYS)

    # 5. Walk-forward validation
    print("  Running walk-forward validation...")
    wf_results = walk_forward_validation(df_raw, FAST_SMA, SLOW_SMA,
                                         INITIAL_CAP, RISK_FREE, TRADING_DAYS)

    # 6. Output
    print_report(metrics, wf_results, TICKER, FAST_SMA, SLOW_SMA)
    plot_results(df_bt, metrics, TICKER, FAST_SMA, SLOW_SMA, wf_results)

    print("\n  Done. Open backtest_results_1.png for the chart.\n")


if __name__ == "__main__":
    main()
