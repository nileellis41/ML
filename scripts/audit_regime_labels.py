"""Audit HMM regime labels for economic plausibility.

Checks that mean market returns satisfy:
    crisis_mean < risk_off_mean < neutral_mean < risk_on_mean

Also checks for one-period shift bugs by printing representative dates per regime.

Run from the project root:
    python scripts/audit_regime_labels.py

Exit codes:
    0  — all assertions pass
    1  — ordering violated or missing data
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

REGIMES_PATH = ROOT / "data" / "regimes" / "hmm_probs.parquet"
ALPACA_DIR = ROOT / "data" / "raw" / "alpaca"
FRED_DIR = ROOT / "data" / "raw" / "fred"

EXPECTED_ORDER = ["crisis", "risk_off", "neutral", "risk_on"]


def _load_market_returns() -> pd.Series:
    """Load daily log returns for the broadest available equity series."""
    # Prefer NASDAQCOM (FRED, 2014–present) — same series used for HMM inputs
    nasdaq_path = FRED_DIR / "NASDAQCOM.parquet"
    if nasdaq_path.exists():
        df = pd.read_parquet(nasdaq_path)
        prices = df.squeeze()
        ret = np.log(prices / prices.shift(1)).dropna()
        ret.name = "NASDAQCOM"
        return ret

    # Fallback: SPY from Alpaca
    spy_path = ALPACA_DIR / "SPY.parquet"
    if spy_path.exists():
        df = pd.read_parquet(spy_path)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        ret.name = "SPY"
        return ret

    raise RuntimeError("Neither NASDAQCOM nor SPY available for market returns.")


def main() -> int:
    if not REGIMES_PATH.exists():
        print(f"ERROR: HMM regime file not found at {REGIMES_PATH}")
        print("Run:  python -m src.experiments.run_all --run-hmm --skip-validation")
        return 1

    regime_df = pd.read_parquet(REGIMES_PATH)
    if "predicted_label" not in regime_df.columns:
        print("ERROR: 'predicted_label' column missing from regime parquet.")
        return 1

    market_ret = _load_market_returns()
    print(f"Market returns: {market_ret.name}  {market_ret.index[0].date()} to {market_ret.index[-1].date()}")
    print(f"HMM OOS window: {regime_df.index[0].date()} to {regime_df.index[-1].date()}")
    print(f"HMM OOS rows:   {len(regime_df)}")
    print()

    labels = regime_df["predicted_label"]
    common = labels.index.intersection(market_ret.index)
    if len(common) == 0:
        print("ERROR: No overlapping dates between HMM output and market returns.")
        return 1

    labels_aligned = labels.reindex(common)
    ret_aligned = market_ret.reindex(common)

    print("=" * 60)
    print("REGIME MEAN RETURNS (should be crisis < risk_off < neutral < risk_on)")
    print("=" * 60)

    stats: dict[str, dict] = {}
    for label in EXPECTED_ORDER:
        mask = labels_aligned == label
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:12s}  n=0  (no dates in this regime)")
            stats[label] = {"mean": np.nan, "n": 0}
            continue
        r = ret_aligned[mask]
        mu = float(r.mean())
        sigma = float(r.std())
        ann_mu = float(np.exp(mu * 252) - 1)
        stats[label] = {"mean": mu, "ann_mean": ann_mu, "n": n}
        print(f"  {label:12s}  n={n:4d}  daily_mean={mu:+.5f}  ann_return={ann_mu:+.2%}  daily_std={sigma:.5f}")

    print()

    # Print 5 representative dates per regime
    print("=" * 60)
    print("REPRESENTATIVE DATES PER REGIME  (check for economic plausibility)")
    print("=" * 60)
    for label in EXPECTED_ORDER:
        mask = labels_aligned == label
        dates = labels_aligned[mask].index
        if len(dates) == 0:
            continue
        # Pick 5 spread across the date range
        idxs = np.linspace(0, len(dates) - 1, min(5, len(dates)), dtype=int)
        sample = dates[idxs]
        print(f"\n  {label}:")
        for d in sample:
            r = ret_aligned.get(d, np.nan)
            print(f"    {d.date()}  market_ret={r:+.4f}")

    print()

    # -----------------------------------------------------------------------
    # Ordering assertion
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("ORDERING CHECK")
    print("=" * 60)

    means = {lbl: stats[lbl]["mean"] for lbl in EXPECTED_ORDER if stats[lbl]["n"] > 0}
    ordered_labels = [l for l in EXPECTED_ORDER if l in means]

    if len(ordered_labels) < 2:
        print("WARN: Fewer than 2 regimes with data — cannot check ordering.")
        return 0

    failed = False
    for i in range(len(ordered_labels) - 1):
        lo_lbl = ordered_labels[i]
        hi_lbl = ordered_labels[i + 1]
        lo_mean = means[lo_lbl]
        hi_mean = means[hi_lbl]
        ok = lo_mean < hi_mean
        status = "OK" if ok else "FAIL"
        print(f"  [{status}]  {lo_lbl} mean ({lo_mean:+.5f}) < {hi_lbl} mean ({hi_mean:+.5f})")
        if not ok:
            failed = True

    print()
    if failed:
        print("REGIME LABELS APPEAR MISALIGNED — check for shift bugs in hmm.py")
        print("Possible causes:")
        print("  - Label assignment uses period-t features to label period t+1")
        print("  - HMM state numbering not sorted by return before labeling")
        return 1
    else:
        print("PASS — regime ordering holds: crisis < risk_off < neutral < risk_on")

    # -----------------------------------------------------------------------
    # Check for known bad dates (if within OOS window)
    # -----------------------------------------------------------------------
    known_crisis_dates = {
        "2020-03-16": "COVID crash",
        "2020-03-18": "COVID crash",
        "2022-01-24": "Fed tightening selloff",
        "2022-09-13": "CPI surprise selloff",
    }
    print()
    print("=" * 60)
    print("SPOT CHECK — KNOWN MARKET STRESS DATES")
    print("=" * 60)
    print(f"{'Date':<14} {'Event':<28} {'Regime':<14} {'Market Ret':>12}")
    for date_str, event in known_crisis_dates.items():
        ts = pd.Timestamp(date_str)
        if ts not in labels_aligned.index:
            print(f"  {date_str:<14} {event:<28} (not in HMM OOS window)")
            continue
        lbl = labels_aligned[ts]
        r = ret_aligned.get(ts, np.nan)
        flag = "  <<< check" if lbl in ("risk_on", "neutral") else ""
        print(f"  {date_str:<14} {event:<28} {str(lbl):<14} {r:>+10.4f}{flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
