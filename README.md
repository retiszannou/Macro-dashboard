# OECD Macro-Financial Surveillance Dashboard

A professional Bloomberg-style dark dashboard for monitoring macroeconomic and financial conditions across OECD countries. Built with Dash, Plotly, and live data from yfinance, the World Bank, and FRED.

---

## What it covers

Four analytical modules, each with interactive controls and live data:

**Equities** - Price performance, daily returns, rolling volatility, and correlation matrix for major OECD equity indices (S&P 500, DAX, CAC 40, Nikkei, FTSE, and more).

**Macro** - GDP growth, inflation, unemployment, debt-to-GDP, and current account balances for up to 15 OECD countries. Country comparison bar charts, time-series trends, and a two-indicator scatter plot.

**FX** - Exchange rate dynamics for eight major currency pairs versus the US dollar. Normalised performance, daily returns, volatility, and cross-currency correlations.

**Monetary Policy** - Fed Funds Rate, ECB rate, US CPI, unemployment, and credit spreads from FRED. Gauge indicators for the latest readings and historical time-series charts.

---

## Data sources

| Source | Data |
|---|---|
| yfinance | Equity indices, FX rates (real-time) |
| World Bank API | GDP, inflation, unemployment, debt (annual) |
| FRED (St. Louis Fed) | Policy rates, CPI, M2, credit spreads |

The dashboard falls back to synthetic demonstration data automatically if any source is unavailable.

---

## Installation

```bash
git clone https://github.com/your-username/macro-dashboard.git
cd macro-dashboard
pip install -r requirements.txt
python app.py
```

Then open your browser at `http://localhost:8050`.

---

## Project structure

```
macro-dashboard/
├── app.py          # Dash app layout and callbacks
├── src/
│   ├── data.py     # Data fetching and fallbacks
│   └── charts.py   # Plotly chart factory
└── requirements.txt
```

---

## Design

Dark theme inspired by Bloomberg Terminal and institutional trading dashboards. All charts share a consistent color system and typography. The layout is fully responsive.

---

> Built for academic and research purposes. Not financial advice.
