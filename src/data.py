"""
src/data.py
Data fetching layer for the OECD Macro-Financial Dashboard.
Sources: yfinance, World Bank (wbdata), FRED via pandas-datareader.
All functions return pandas DataFrames ready for charting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import pandas_datareader.data as web
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False

try:
    import wbdata
    WB_AVAILABLE = True
except ImportError:
    WB_AVAILABLE = False


# ── Country universe ──────────────────────────────────────────────────────────

OECD_COUNTRIES = {
    "United States": "US",
    "Germany":       "DE",
    "France":        "FR",
    "United Kingdom":"GB",
    "Japan":         "JP",
    "Canada":        "CA",
    "Italy":         "IT",
    "Spain":         "ES",
    "Netherlands":   "NL",
    "Sweden":        "SE",
    "Switzerland":   "CH",
    "Australia":     "AU",
    "South Korea":   "KR",
    "Poland":        "PL",
    "Mexico":        "MX",
}

EQUITY_INDICES = {
    "S&P 500":    "^GSPC",
    "DAX":        "^GDAXI",
    "CAC 40":     "^FCHI",
    "FTSE 100":   "^FTSE",
    "Nikkei 225": "^N225",
    "TSX":        "^GSPTSE",
    "ASX 200":    "^AXJO",
    "KOSPI":      "^KS11",
    "SMI":        "^SSMI",
}

FX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "JPY/USD": "JPY=X",
    "CAD/USD": "CAD=X",
    "AUD/USD": "AUDUSD=X",
    "CHF/USD": "CHF=X",
    "KRW/USD": "KRW=X",
    "MXN/USD": "MXN=X",
}

BONDS = {
    "US 10Y":  "^TNX",
    "DE 10Y":  "^IRDE10Y.DE",
    "GB 10Y":  "^IRGB10YD.GB",
    "JP 10Y":  "^IRJP10YD.JP",
}

FRED_SERIES = {
    "US Fed Funds Rate":   "FEDFUNDS",
    "US CPI YoY":          "CPIAUCSL",
    "US Unemployment":     "UNRATE",
    "EA Refinancing Rate": "ECBMLFR",
    "US M2":               "M2SL",
    "US Credit Spread":    "BAMLH0A0HYM2",
}

WB_INDICATORS = {
    "GDP Growth (%)":       "NY.GDP.MKTP.KD.ZG",
    "Inflation (%)":        "FP.CPI.TOTL.ZG",
    "Unemployment (%)":     "SL.UEM.TOTL.ZS",
    "Debt/GDP (%)":         "GC.DOD.TOTL.GD.ZS",
    "Current Account/GDP":  "BN.CAB.XOKA.GD.ZS",
}


# ── Equity data ───────────────────────────────────────────────────────────────

def fetch_equity_indices(period: str = "1y") -> pd.DataFrame:
    """Download closing prices for major OECD equity indices."""
    if not YF_AVAILABLE:
        return _synthetic_equity()

    frames = {}
    for name, ticker in EQUITY_INDICES.items():
        try:
            data = yf.download(ticker, period=period, auto_adjust=True,
                               progress=False, timeout=10)
            if not data.empty:
                close = data["Close"]
                if hasattr(close, "squeeze"):
                    close = close.squeeze()
                frames[name] = close
        except Exception:
            pass

    if not frames:
        return _synthetic_equity()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.ffill().dropna(how="all")
    return df


def compute_returns(prices: pd.DataFrame, period: str = "1D") -> pd.DataFrame:
    """Compute percentage returns from price DataFrame."""
    if period == "1D":
        return prices.pct_change(1) * 100
    elif period == "1W":
        return prices.pct_change(5) * 100
    elif period == "1M":
        return prices.pct_change(21) * 100
    return prices.pct_change(1) * 100


def compute_volatility(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised rolling volatility (%)."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252) * 100


def compute_correlation(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling correlation matrix on the last `window` observations."""
    subset = prices.dropna().tail(window)
    return subset.pct_change().dropna().corr()


# ── FX data ───────────────────────────────────────────────────────────────────

def fetch_fx(period: str = "1y") -> pd.DataFrame:
    """Download FX rates vs USD."""
    if not YF_AVAILABLE:
        return _synthetic_fx()

    frames = {}
    for name, ticker in FX_PAIRS.items():
        try:
            data = yf.download(ticker, period=period, auto_adjust=True,
                               progress=False, timeout=10)
            if not data.empty:
                close = data["Close"]
                if hasattr(close, "squeeze"):
                    close = close.squeeze()
                frames[name] = close
        except Exception:
            pass

    if not frames:
        return _synthetic_fx()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.ffill().dropna(how="all")


# ── Macro data (World Bank) ───────────────────────────────────────────────────

def fetch_world_bank(countries: list = None,
                     start_year: int = 2010) -> pd.DataFrame:
    """Fetch macro indicators from World Bank for OECD countries."""
    if countries is None:
        countries = list(OECD_COUNTRIES.values())[:10]

    if not WB_AVAILABLE:
        return _synthetic_macro(countries)

    start = datetime(start_year, 1, 1)
    end   = datetime(datetime.now().year, 12, 31)

    records = []
    for ind_name, ind_code in WB_INDICATORS.items():
        try:
            raw = wbdata.get_dataframe(
                {ind_code: ind_name},
                country=countries,
                date=(start, end),
            )
            if raw is not None and not raw.empty:
                raw = raw.reset_index()
                raw.columns = ["country", "date", "value"]
                raw["indicator"] = ind_name
                records.append(raw)
        except Exception:
            pass

    if not records:
        return _synthetic_macro(countries)

    df = pd.concat(records, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


# ── FRED data ─────────────────────────────────────────────────────────────────

def fetch_fred(series_keys: list = None, start: str = "2015-01-01") -> pd.DataFrame:
    """Download FRED series via pandas-datareader."""
    if series_keys is None:
        series_keys = ["US Fed Funds Rate", "US CPI YoY",
                       "US Unemployment", "US Credit Spread"]

    if not PDR_AVAILABLE:
        return _synthetic_fred(series_keys)

    frames = {}
    for key in series_keys:
        code = FRED_SERIES.get(key)
        if not code:
            continue
        try:
            s = web.DataReader(code, "fred",
                               start=start,
                               end=datetime.today().strftime("%Y-%m-%d"))
            frames[key] = s.squeeze()
        except Exception:
            pass

    if not frames:
        return _synthetic_fred(series_keys)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.ffill()


# ── Synthetic fallbacks (offline / demo mode) ─────────────────────────────────

def _synthetic_equity(n_days: int = 252) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.today(), periods=n_days)
    data  = {}
    bases = [4500, 15000, 7200, 7500, 32000, 20000, 7300, 2400, 11000]
    for i, (name, _) in enumerate(EQUITY_INDICES.items()):
        drift = 0.0003 + np.random.uniform(-0.0001, 0.0002)
        vol   = 0.012 + np.random.uniform(0, 0.005)
        rets  = np.random.normal(drift, vol, n_days)
        price = bases[i] * np.exp(np.cumsum(rets))
        data[name] = price
    return pd.DataFrame(data, index=dates)


def _synthetic_fx(n_days: int = 252) -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.bdate_range(end=datetime.today(), periods=n_days)
    bases = [1.08, 1.27, 0.0068, 0.74, 0.65, 1.10, 0.00075, 0.058]
    data  = {}
    for i, name in enumerate(FX_PAIRS):
        vol  = 0.004
        rets = np.random.normal(0, vol, n_days)
        data[name] = bases[i] * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


def _synthetic_macro(countries: list) -> pd.DataFrame:
    np.random.seed(3)
    records = []
    years   = range(2012, datetime.now().year + 1)
    for country in countries:
        for year in years:
            for ind_name in WB_INDICATORS:
                base_val = {
                    "GDP Growth (%)":      2.0,
                    "Inflation (%)":       2.5,
                    "Unemployment (%)":    6.0,
                    "Debt/GDP (%)":       70.0,
                    "Current Account/GDP": 0.5,
                }.get(ind_name, 0)
                val = base_val + np.random.normal(0, base_val * 0.2)
                records.append({
                    "country":   country,
                    "date":      datetime(year, 12, 31),
                    "year":      year,
                    "value":     round(val, 2),
                    "indicator": ind_name,
                })
    return pd.DataFrame(records)


def _synthetic_fred(series_keys: list) -> pd.DataFrame:
    np.random.seed(11)
    dates  = pd.date_range(start="2015-01-01",
                           end=datetime.today(), freq="MS")
    data   = {}
    params = {
        "US Fed Funds Rate":   (1.5, 2.0),
        "US CPI YoY":          (2.5, 1.5),
        "US Unemployment":     (5.0, 1.5),
        "EA Refinancing Rate": (1.0, 1.5),
        "US M2":               (18000, 2000),
        "US Credit Spread":    (4.0, 1.5),
    }
    for key in series_keys:
        base, std = params.get(key, (2.0, 1.0))
        vals = base + np.cumsum(np.random.normal(0, std * 0.05, len(dates)))
        data[key] = np.clip(vals, 0, None)
    return pd.DataFrame(data, index=dates)
