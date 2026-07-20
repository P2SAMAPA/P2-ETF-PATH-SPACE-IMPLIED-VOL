# Path-Space Implied Volatility for ETFs

Models implied volatility as a functional on path space (via signature methods), rather than as a function of strike only. Captures path-dependent features of the implied volatility surface that static approaches miss. The per‑ETF score is the path-space implied volatility.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Truncated path signature (iterated integrals)
- Signature of returns + macro factor
- Ridge regression for IV prediction
- Score = path-space IV (higher = higher expected vol)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-path-space-implied-vol-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High path-space IV → expected volatility is high → options are expensive.
- Low path-space IV → expected volatility is low.

## Requirements

See `requirements.txt`.
