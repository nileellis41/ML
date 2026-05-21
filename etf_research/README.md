# ETF Research Pipeline

Comparative research pipeline evaluating statistical, ML, RL, and agentic systems for ETF return prediction and portfolio allocation.

---

## Prerequisites

All packages are installed. API keys are in `.env` and load automatically — no manual export needed.

Run everything from the project root:

```
cd c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research
```

---

## Part 1 — Candlestick Sample Images (visual stop point)

Render 5 sample candlestick images and inspect them before training the CNN.

```
cd c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research

python -c "
from src.utils.seeds import set_all_seeds; set_all_seeds()
import yaml
from src.data.alpaca_loader import load_universe
from src.data.candlestick_render import produce_sample_images
with open('config/assets.yaml') as f:
    a = yaml.safe_load(f)
ohlcv = load_universe(a['sector_etfs'][:5], start='2018-01-01')
paths = produce_sample_images(list(ohlcv.keys()), ohlcv, n_samples=5)
for p in paths:
    print(p)
"
```

**Check output here before continuing:**
```
c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\figures\candle_samples\
```

Each file should look like a candlestick chart — white wicks and bodies on a black background, one column per trading day, with a volume bar row at the bottom.

---

## Part 2 — Full Pipeline

Runs in order: FRED validation → pull all OHLCV → build feature panel → walk-forward folds → leakage check → all prediction models (linear, logistic, PCA, LSTM) in base and regime variants.

```
cd c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research

python -m src.experiments.run_all --refit
```

Predictions are saved here as they complete:
```
c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\predictions\
```

One file per model-variant, e.g. `linear_base.parquet`, `lstm_regime.parquet`.

> LSTM folds take the longest. Linear and logistic finish in seconds. The full run across all models and all folds is 30–90 minutes depending on CPU/GPU.

To run only a subset of models:
```
python -m src.experiments.run_all --refit --models linear logistic pca
```

---

## Part 3 — Comparison Table and Plots

Reads all cached predictions and generates the master comparison table plus all figures.

```
cd c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research

python -m src.evaluation.reporting --run-all
```

**Output files:**

| File | Description |
|------|-------------|
| `c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\tables\master_comparison.csv` | Full metrics table (Sharpe, return, drawdown, turnover, regime-conditional) |
| `c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\tables\master_comparison.md` | Markdown version for copy-paste |
| `c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\figures\equity_curves_base.png` | Equity curve overlay — base models |
| `c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\figures\equity_curves_regime.png` | Equity curve overlay — regime models |
| `c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\figures\drawdowns.png` | Drawdown comparison |
| `c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\results\figures\regime_shaded_spy.png` | SPY with regime-coloured background |

---

## Tests

```
cd c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research

python -m pytest tests/test_walk_forward_no_leakage.py tests/test_candlestick_no_lookahead.py -v
```

Expected: **28 passed**. Run these before training any model.

FRED tests require the API key (already in `.env`):
```
python -m pytest tests/test_fred_series_validity.py -v -m fred
```

---

## Notebooks

Open in order. Each has a stop point marked with instructions.

```
cd c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research
jupyter notebook
```

| Notebook | Purpose | Stop point |
|----------|---------|------------|
| `notebooks\01_data_exploration.ipynb` | Alpaca smoke test, FRED validation, coverage heatmap | None |
| `notebooks\02_candlestick_inspection.ipynb` | Render and display 5 sample images | **Yes — inspect images before CNN** |
| `notebooks\03_regime_detection.ipynb` | HMM + LSTM regime classifier, regime-shaded SPY | **Yes — inspect regime labels** |
| `notebooks\04_prediction_models.ipynb` | Walk-forward for all prediction models | **Yes — check linear output first** |
| `notebooks\05_portfolio_systems.ipynb` | Model-based RL, PPO, Agentic RAG | None |
| `notebooks\06_final_comparison.ipynb` | Master table, equity curves, regime plots | None |

---

## Asset Universe

| Category | Tickers |
|----------|---------|
| Sector ETFs (11) | XLK XLF XLV XLE XLY XLP XLI XLB XLU XLRE XLC |
| Indices (3) | SPY QQQ IWM |
| Commodities (3) | USO GLD DBA |
| Bond ETFs (4) | TLT IEF LQD HYG |

All price/volume data from **Alpaca Market Data API**. No yfinance.

## Model Variants

Every prediction model has two variants:
- `_base` — price + macro features only
- `_regime` — same features + 4-column HMM regime probability vector appended

## Key Design Decisions

- **No yfinance** — all price data flows through Alpaca (`alpaca-py`)
- **No look-ahead** — walk-forward engine enforced by leakage tests; candlestick images use only trailing data
- **FRED failures are loud** — `validate_fred.py` hard-fails if any series returns empty; no silent substitutions
- **Candlestick images** — 64×60px, 2-channel (OHLC + volume), min-max scaled per window, cached as `.npy`
- **Regime augmentation** — 4-state Gaussian HMM labels supervise the LSTM regime classifier; both emit probability vectors
- **Auto .env loading** — `src/utils/seeds.py` loads `.env` on import; no shell exports needed

## Repository Layout

```
c:\Users\nilee\OneDrive\Desktop\NYU\ML\etf_research\
├── .env                         # API keys (auto-loaded)
├── config\
│   ├── assets.yaml              # ETF universe (23 tickers)
│   ├── features.yaml            # 40 validated FRED series + sector blocks
│   └── experiments.yaml         # Walk-forward windows, model hyperparameters
├── data\
│   ├── raw\alpaca\              # Cached OHLCV parquets ({ticker}.parquet)
│   ├── raw\fred\                # Cached FRED series ({series_id}.parquet)
│   ├── processed\panel.parquet  # Aligned multi-asset feature panel
│   ├── processed\candles\       # Rendered candlestick .npy arrays
│   └── regimes\                 # hmm_probs.parquet, lstm_probs.parquet
├── src\
│   ├── data\                    # alpaca_loader, fred_loader, validate_fred, features, panel, candlestick_render
│   ├── regimes\                 # hmm.py, lstm_regime.py
│   ├── models\                  # base, linear, pca_factor, var, lstm, cnn, hybrids
│   ├── portfolio\               # model_based_rl, policy_gradient, agentic_rag
│   ├── evaluation\              # walk_forward, metrics, reporting
│   ├── experiments\             # run_all.py
│   └── utils\                   # seeds.py, io.py
├── notebooks\                   # 01 through 06
├── tests\                       # 28 tests (leakage, FRED, candlestick)
└── results\
    ├── tables\                  # master_comparison.csv / .md
    ├── figures\                 # equity curves, drawdowns, regime plots, candle samples
    ├── predictions\             # {model}_{variant}.parquet
    └── agent_traces\            # Agentic RAG reasoning logs (JSON)
```
