"""
app.py
OECD Macro-Financial Surveillance Dashboard
A professional Bloomberg-style dark dashboard built with Dash and Plotly.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from data import (
    fetch_equity_indices, fetch_fx, fetch_world_bank,
    fetch_fred, compute_returns, compute_volatility,
    compute_correlation, OECD_COUNTRIES, WB_INDICATORS,
    EQUITY_INDICES, FX_PAIRS,
)
from charts import (
    line_prices, bar_returns, heatmap_correlation,
    area_volatility, bar_macro_comparison, line_macro_trend,
    scatter_macro, line_fred, gauge_rate, line_fx, kpi_indicator,
    BACKGROUND, SURFACE, SURFACE_2, BORDER,
    TEXT_PRIMARY, TEXT_MUTED, ACCENT_BLUE, ACCENT_GREEN,
    ACCENT_RED, ACCENT_AMBER,
)


# ── App init ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="OECD Macro-Financial Dashboard",
)
server = app.server


# ── Global styles ─────────────────────────────────────────────────────────────

GLOBAL_STYLE = {
    "backgroundColor": BACKGROUND,
    "color": TEXT_PRIMARY,
    "fontFamily": "Inter, system-ui, sans-serif",
    "minHeight": "100vh",
}

CARD_STYLE = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
}

CARD_TITLE = {
    "fontSize": "11px",
    "fontWeight": "600",
    "color": TEXT_MUTED,
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
    "marginBottom": "4px",
}

KPI_VALUE = {
    "fontSize": "22px",
    "fontWeight": "600",
    "color": TEXT_PRIMARY,
    "lineHeight": "1.2",
}

NAV_STYLE = {
    "backgroundColor": SURFACE,
    "borderBottom": f"1px solid {BORDER}",
    "padding": "0 24px",
    "position": "sticky",
    "top": "0",
    "zIndex": "100",
}

TAB_STYLE = {
    "backgroundColor": "transparent",
    "color": TEXT_MUTED,
    "border": "none",
    "borderBottom": f"2px solid transparent",
    "padding": "14px 20px",
    "fontSize": "13px",
    "fontWeight": "500",
    "cursor": "pointer",
}

TAB_SELECTED = {
    **TAB_STYLE,
    "color": TEXT_PRIMARY,
    "borderBottom": f"2px solid {ACCENT_BLUE}",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def card(children, style=None):
    s = {**CARD_STYLE, **(style or {})}
    return html.Div(children, style=s)


def section_title(text):
    return html.P(text, style=CARD_TITLE)


def delta_badge(value: float) -> html.Span:
    color = ACCENT_GREEN if value >= 0 else ACCENT_RED
    arrow = "+" if value >= 0 else ""
    return html.Span(
        f"{arrow}{value:.2f}%",
        style={"color": color, "fontSize": "12px", "fontWeight": "500"},
    )


def kpi_card(label, value, delta=None, suffix="", prefix=""):
    children = [
        section_title(label),
        html.Div([
            html.Span(f"{prefix}{value}{suffix}", style=KPI_VALUE),
            html.Span(" ", style={"marginLeft": "8px"}),
            delta_badge(delta) if delta is not None else html.Span(),
        ], style={"display": "flex", "alignItems": "baseline", "gap": "6px"}),
    ]
    return card(children)


# ── Layout ────────────────────────────────────────────────────────────────────

def build_navbar():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("MACRO", style={
                        "color": ACCENT_BLUE, "fontWeight": "600",
                        "fontSize": "15px", "letterSpacing": "0.1em",
                    }),
                    html.Span("WATCH", style={
                        "color": TEXT_PRIMARY, "fontWeight": "400",
                        "fontSize": "15px", "letterSpacing": "0.1em",
                    }),
                    html.Span(" | OECD", style={
                        "color": TEXT_MUTED, "fontSize": "12px",
                        "marginLeft": "8px",
                    }),
                ], style={"display": "flex", "alignItems": "center",
                          "padding": "14px 0"}),
            ], width="auto"),

            dbc.Col([
                dcc.Tabs(
                    id="main-tabs",
                    value="equities",
                    children=[
                        dcc.Tab(label="Equities",        value="equities",
                                style=TAB_STYLE, selected_style=TAB_SELECTED),
                        dcc.Tab(label="Macro",           value="macro",
                                style=TAB_STYLE, selected_style=TAB_SELECTED),
                        dcc.Tab(label="FX",              value="fx",
                                style=TAB_STYLE, selected_style=TAB_SELECTED),
                        dcc.Tab(label="Monetary Policy", value="monetary",
                                style=TAB_STYLE, selected_style=TAB_SELECTED),
                    ],
                    style={"borderBottom": "none", "backgroundColor": "transparent"},
                ),
            ]),

            dbc.Col([
                html.Div([
                    html.Span(
                        id="live-clock",
                        style={"color": TEXT_MUTED, "fontSize": "12px",
                               "fontFamily": "monospace"},
                    ),
                    dcc.Interval(id="clock-interval", interval=60000, n_intervals=0),
                ], style={"display": "flex", "alignItems": "center",
                          "justifyContent": "flex-end", "padding": "14px 0"}),
            ], width="auto"),
        ], align="center"),
    ], style=NAV_STYLE)


def build_equities_tab():
    return html.Div([
        # Controls
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Period", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="eq-period",
                        options=[
                            {"label": "1 Month",  "value": "1mo"},
                            {"label": "3 Months", "value": "3mo"},
                            {"label": "6 Months", "value": "6mo"},
                            {"label": "1 Year",   "value": "1y"},
                            {"label": "2 Years",  "value": "2y"},
                        ],
                        value="1y",
                        clearable=False,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Indices", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="eq-indices",
                        options=[{"label": k, "value": k}
                                 for k in EQUITY_INDICES],
                        value=list(EQUITY_INDICES.keys())[:6],
                        multi=True,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Normalise to 100", style={**CARD_TITLE, "marginBottom": "8px"}),
                    dcc.Checklist(
                        id="eq-normalise",
                        options=[{"label": " Yes", "value": "yes"}],
                        value=["yes"],
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"color": TEXT_PRIMARY, "fontSize": "13px"},
                    ),
                ], width=2),
            ], className="g-3"),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        # KPI row
        dbc.Row(id="eq-kpi-row", className="g-3 mb-3"),

        # Charts
        dbc.Row([
            dbc.Col([
                card([dcc.Graph(id="eq-price-chart", config={"displayModeBar": False})]),
            ], width=8),
            dbc.Col([
                card([dcc.Graph(id="eq-returns-chart", config={"displayModeBar": False})]),
            ], width=4),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col([
                card([dcc.Graph(id="eq-vol-chart", config={"displayModeBar": False})]),
            ], width=6),
            dbc.Col([
                card([dcc.Graph(id="eq-corr-chart", config={"displayModeBar": False})]),
            ], width=6),
        ], className="g-3"),

        dcc.Store(id="eq-prices-store"),
        dcc.Loading(id="eq-loading", type="circle",
                    color=ACCENT_BLUE, fullscreen=False,
                    children=html.Div(id="eq-trigger")),
    ], style={"padding": "20px"})


def build_macro_tab():
    country_opts = [{"label": c, "value": c}
                    for c in OECD_COUNTRIES.values()]
    indicator_opts = [{"label": k, "value": k} for k in WB_INDICATORS]

    return html.Div([
        # Controls
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Indicator", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="macro-indicator",
                        options=indicator_opts,
                        value="GDP Growth (%)",
                        clearable=False,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Countries", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="macro-countries",
                        options=country_opts,
                        value=list(OECD_COUNTRIES.values())[:8],
                        multi=True,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=6),
            ], className="g-3"),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        # Bar + trend
        dbc.Row([
            dbc.Col([
                card([dcc.Graph(id="macro-bar-chart",
                                config={"displayModeBar": False})]),
            ], width=6),
            dbc.Col([
                card([dcc.Graph(id="macro-trend-chart",
                                config={"displayModeBar": False})]),
            ], width=6),
        ], className="g-3 mb-3"),

        # Scatter
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("X axis", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="scatter-x",
                        options=indicator_opts,
                        value="GDP Growth (%)",
                        clearable=False,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Label("Y axis", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="scatter-y",
                        options=indicator_opts,
                        value="Inflation (%)",
                        clearable=False,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ]),
            ], width=2),
            dbc.Col([
                card([dcc.Graph(id="macro-scatter",
                                config={"displayModeBar": False})]),
            ], width=10),
        ], className="g-3"),

        dcc.Store(id="macro-store"),
    ], style={"padding": "20px"})


def build_fx_tab():
    fx_opts = [{"label": k, "value": k} for k in FX_PAIRS]
    return html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Currency Pairs", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="fx-pairs",
                        options=fx_opts,
                        value=list(FX_PAIRS.keys())[:5],
                        multi=True,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Period", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="fx-period",
                        options=[
                            {"label": "3 Months", "value": "3mo"},
                            {"label": "6 Months", "value": "6mo"},
                            {"label": "1 Year",   "value": "1y"},
                            {"label": "2 Years",  "value": "2y"},
                        ],
                        value="1y",
                        clearable=False,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=2),
            ], className="g-3"),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                card([dcc.Graph(id="fx-line-chart", config={"displayModeBar": False})]),
            ], width=8),
            dbc.Col([
                card([dcc.Graph(id="fx-returns-chart", config={"displayModeBar": False})]),
            ], width=4),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col([
                card([dcc.Graph(id="fx-vol-chart", config={"displayModeBar": False})]),
            ], width=6),
            dbc.Col([
                card([dcc.Graph(id="fx-corr-chart", config={"displayModeBar": False})]),
            ], width=6),
        ], className="g-3"),

        dcc.Store(id="fx-store"),
    ], style={"padding": "20px"})


def build_monetary_tab():
    fred_opts = [
        {"label": "Fed Funds Rate",    "value": "US Fed Funds Rate"},
        {"label": "US Unemployment",   "value": "US Unemployment"},
        {"label": "US CPI YoY",        "value": "US CPI YoY"},
        {"label": "ECB Rate",          "value": "EA Refinancing Rate"},
        {"label": "US Credit Spread",  "value": "US Credit Spread"},
        {"label": "US M2",             "value": "US M2"},
    ]
    return html.Div([
        dbc.Row([
            dbc.Col([
                card([dcc.Graph(id="gauge-fed", config={"displayModeBar": False})]),
            ], width=3),
            dbc.Col([
                card([dcc.Graph(id="gauge-cpi", config={"displayModeBar": False})]),
            ], width=3),
            dbc.Col([
                card([dcc.Graph(id="gauge-unemp", config={"displayModeBar": False})]),
            ], width=3),
            dbc.Col([
                card([dcc.Graph(id="gauge-spread", config={"displayModeBar": False})]),
            ], width=3),
        ], className="g-3 mb-3"),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Series", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="fred-series",
                        options=fred_opts,
                        value=["US Fed Funds Rate", "US CPI YoY",
                               "US Unemployment"],
                        multi=True,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=8),
                dbc.Col([
                    html.Label("Start date", style={**CARD_TITLE, "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="fred-start",
                        options=[
                            {"label": "2010", "value": "2010-01-01"},
                            {"label": "2015", "value": "2015-01-01"},
                            {"label": "2018", "value": "2018-01-01"},
                            {"label": "2020", "value": "2020-01-01"},
                        ],
                        value="2015-01-01",
                        clearable=False,
                        style={"backgroundColor": SURFACE_2,
                               "color": TEXT_PRIMARY, "border": f"1px solid {BORDER}"},
                    ),
                ], width=2),
            ], className="g-3"),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        card([dcc.Graph(id="fred-chart", config={"displayModeBar": False})]),
        dcc.Store(id="fred-store"),
    ], style={"padding": "20px"})


app.layout = html.Div([
    build_navbar(),
    html.Div(id="tab-content"),
    dcc.Store(id="active-tab-store", data="equities"),
    dcc.Interval(id="data-refresh", interval=300_000, n_intervals=0),
], style=GLOBAL_STYLE)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"),
              Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "equities":
        return build_equities_tab()
    elif tab == "macro":
        return build_macro_tab()
    elif tab == "fx":
        return build_fx_tab()
    elif tab == "monetary":
        return build_monetary_tab()
    return html.Div()


@app.callback(Output("live-clock", "children"),
              Input("clock-interval", "n_intervals"))
def update_clock(_):
    return datetime.utcnow().strftime("UTC %H:%M  |  %d %b %Y")


# ── Equity callbacks ──────────────────────────────────────────────────────────

@app.callback(
    Output("eq-prices-store", "data"),
    Input("eq-period", "value"),
    Input("data-refresh", "n_intervals"),
)
def load_equity_data(period, _):
    prices = fetch_equity_indices(period=period or "1y")
    return prices.to_json(date_format="iso")


@app.callback(
    Output("eq-price-chart", "figure"),
    Output("eq-returns-chart", "figure"),
    Output("eq-vol-chart", "figure"),
    Output("eq-corr-chart", "figure"),
    Output("eq-kpi-row", "children"),
    Input("eq-prices-store", "data"),
    Input("eq-indices", "value"),
    Input("eq-normalise", "value"),
)
def update_equity_charts(prices_json, selected, normalise_val):
    if not prices_json or not selected:
        empty = go.Figure()
        empty.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=SURFACE, height=380)
        return empty, empty, empty, empty, []

    prices = pd.read_json(prices_json, convert_dates=True)
    prices.index = pd.to_datetime(prices.index)
    cols = [c for c in selected if c in prices.columns]
    if not cols:
        empty = go.Figure()
        empty.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=SURFACE, height=380)
        return empty, empty, empty, empty, []

    prices = prices[cols].dropna(how="all")
    normalise = bool(normalise_val)

    fig_price = line_prices(prices, title="Index Performance", normalise=normalise)
    returns   = compute_returns(prices, "1D")
    fig_ret   = bar_returns(returns, title="Daily Returns (%)")
    vol       = compute_volatility(prices)
    fig_vol   = area_volatility(vol, title="Rolling Volatility (21d, annualised)")
    corr      = compute_correlation(prices)
    fig_corr  = heatmap_correlation(corr, title="Correlation Matrix (60d)")

    # KPI cards
    last_ret = returns.dropna().iloc[-1] if len(returns.dropna()) > 0 else pd.Series()
    kpi_cols = []
    for col in cols[:6]:
        ret = float(last_ret[col]) if col in last_ret.index else 0.0
        last_price = float(prices[col].dropna().iloc[-1])
        kpi_cols.append(
            dbc.Col([kpi_card(col, f"{last_price:,.1f}", delta=ret)], width=2)
        )

    return fig_price, fig_ret, fig_vol, fig_corr, kpi_cols


# ── Macro callbacks ───────────────────────────────────────────────────────────

@app.callback(
    Output("macro-store", "data"),
    Input("data-refresh", "n_intervals"),
)
def load_macro_data(_):
    df = fetch_world_bank()
    return df.to_json(date_format="iso")


@app.callback(
    Output("macro-bar-chart", "figure"),
    Output("macro-trend-chart", "figure"),
    Output("macro-scatter", "figure"),
    Input("macro-store", "data"),
    Input("macro-indicator", "value"),
    Input("macro-countries", "value"),
    Input("scatter-x", "value"),
    Input("scatter-y", "value"),
)
def update_macro_charts(data_json, indicator, countries, sx, sy):
    empty = go.Figure()
    empty.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=SURFACE, height=380)

    if not data_json:
        return empty, empty, empty

    df = pd.read_json(data_json, convert_dates=True)
    if df.empty:
        return empty, empty, empty

    ind  = indicator or "GDP Growth (%)"
    ctry = countries[:8] if countries else list(OECD_COUNTRIES.values())[:8]

    fig_bar   = bar_macro_comparison(df, ind, title=f"{ind} - Latest")
    fig_trend = line_macro_trend(df, ind, ctry, title=f"{ind} Over Time")
    fig_scat  = scatter_macro(df, sx or "GDP Growth (%)",
                              sy or "Inflation (%)")

    return fig_bar, fig_trend, fig_scat


# ── FX callbacks ──────────────────────────────────────────────────────────────

@app.callback(
    Output("fx-store", "data"),
    Input("fx-period", "value"),
    Input("data-refresh", "n_intervals"),
)
def load_fx_data(period, _):
    fx = fetch_fx(period=period or "1y")
    return fx.to_json(date_format="iso")


@app.callback(
    Output("fx-line-chart", "figure"),
    Output("fx-returns-chart", "figure"),
    Output("fx-vol-chart", "figure"),
    Output("fx-corr-chart", "figure"),
    Input("fx-store", "data"),
    Input("fx-pairs", "value"),
)
def update_fx_charts(data_json, pairs):
    empty = go.Figure()
    empty.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=SURFACE, height=380)

    if not data_json or not pairs:
        return empty, empty, empty, empty

    fx   = pd.read_json(data_json, convert_dates=True)
    fx.index = pd.to_datetime(fx.index)
    cols = [c for c in pairs if c in fx.columns]
    if not cols:
        return empty, empty, empty, empty

    fx = fx[cols].dropna(how="all")

    fig_line = line_fx(fx, title="FX Rates (normalised to 100)")
    returns  = compute_returns(fx, "1D")
    fig_ret  = bar_returns(returns, title="Daily FX Returns (%)")
    vol      = compute_volatility(fx)
    fig_vol  = area_volatility(vol, title="FX Volatility (21d, annualised)")
    corr     = compute_correlation(fx)
    fig_corr = heatmap_correlation(corr, title="FX Correlation Matrix")

    return fig_line, fig_ret, fig_vol, fig_corr


# ── Monetary callbacks ────────────────────────────────────────────────────────

@app.callback(
    Output("fred-store", "data"),
    Input("fred-start", "value"),
    Input("data-refresh", "n_intervals"),
)
def load_fred_data(start, _):
    all_series = ["US Fed Funds Rate", "US CPI YoY",
                  "US Unemployment", "EA Refinancing Rate",
                  "US Credit Spread", "US M2"]
    df = fetch_fred(series_keys=all_series, start=start or "2015-01-01")
    return df.to_json(date_format="iso")


@app.callback(
    Output("fred-chart", "figure"),
    Output("gauge-fed", "figure"),
    Output("gauge-cpi", "figure"),
    Output("gauge-unemp", "figure"),
    Output("gauge-spread", "figure"),
    Input("fred-store", "data"),
    Input("fred-series", "value"),
)
def update_monetary_charts(data_json, series):
    empty = go.Figure()
    empty.update_layout(paper_bgcolor=BACKGROUND, height=220)

    if not data_json:
        return empty, empty, empty, empty, empty

    df = pd.read_json(data_json, convert_dates=True)
    df.index = pd.to_datetime(df.index)

    selected = series or ["US Fed Funds Rate", "US CPI YoY", "US Unemployment"]
    fig_line = line_fred(df, series=selected, title="Monetary Policy Indicators")

    def last_val(col):
        if col in df.columns:
            s = df[col].dropna()
            return float(s.iloc[-1]) if len(s) > 0 else 0.0
        return 0.0

    fed_val    = last_val("US Fed Funds Rate")
    cpi_val    = last_val("US CPI YoY")
    unemp_val  = last_val("US Unemployment")
    spread_val = last_val("US Credit Spread")

    return (
        fig_line,
        gauge_rate(fed_val,    label="Fed Funds Rate",   max_val=8),
        gauge_rate(cpi_val,    label="US CPI YoY (%)",   max_val=12),
        gauge_rate(unemp_val,  label="Unemployment (%)", max_val=15),
        gauge_rate(spread_val, label="HY Credit Spread", max_val=12),
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
