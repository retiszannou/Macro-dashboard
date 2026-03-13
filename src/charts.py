"""
src/charts.py
Reusable Plotly chart factory for the OECD Macro-Financial Dashboard.
All figures use a consistent dark Bloomberg-style theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Design system ─────────────────────────────────────────────────────────────

BACKGROUND   = "#0a0e1a"
SURFACE      = "#111827"
SURFACE_2    = "#1a2235"
BORDER       = "#1f2d45"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED   = "#64748b"
ACCENT_BLUE  = "#3b82f6"
ACCENT_GREEN = "#10b981"
ACCENT_RED   = "#ef4444"
ACCENT_AMBER = "#f59e0b"
ACCENT_PURPLE= "#8b5cf6"

PALETTE = [
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER,
    ACCENT_RED, ACCENT_PURPLE, "#06b6d4",
    "#f97316", "#84cc16", "#ec4899",
]

BASE_LAYOUT = dict(
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, system-ui, sans-serif",
              color=TEXT_PRIMARY, size=12),
    margin=dict(l=48, r=24, t=36, b=40),
    xaxis=dict(
        gridcolor=BORDER, linecolor=BORDER,
        tickcolor=BORDER, showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=BORDER, linecolor=BORDER,
        tickcolor=BORDER, showgrid=True,
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor=SURFACE_2,
        bordercolor=BORDER,
        font=dict(color=TEXT_PRIMARY, size=12),
    ),
)


def _apply_base(fig: go.Figure, title: str = "",
                height: int = 380) -> go.Figure:
    layout = dict(BASE_LAYOUT)
    layout["title"] = dict(text=title, font=dict(size=14, color=TEXT_PRIMARY),
                           x=0.01, xanchor="left")
    layout["height"] = height
    fig.update_layout(**layout)
    return fig


# ── Equity charts ─────────────────────────────────────────────────────────────

def line_prices(prices: pd.DataFrame, title: str = "Equity Indices",
                normalise: bool = True) -> go.Figure:
    """Multi-line chart of equity prices, optionally normalised to 100."""
    fig = go.Figure()
    df  = prices.copy()

    if normalise:
        df = df.div(df.iloc[0]) * 100

    for i, col in enumerate(df.columns):
        series = df[col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=col,
            mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"<b>{col}</b><br>%{{x|%d %b %Y}}<br>"
                          f"{'Index' if normalise else 'Price'}: %{{y:.1f}}<extra></extra>",
        ))

    if normalise:
        fig.add_hline(y=100, line_dash="dot",
                      line_color=TEXT_MUTED, line_width=1)

    return _apply_base(fig, title)


def bar_returns(returns: pd.DataFrame, label: str = "Latest",
                title: str = "Performance") -> go.Figure:
    """Horizontal bar chart of returns for the latest period."""
    last = returns.dropna().iloc[-1].sort_values()
    colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in last.values]

    fig = go.Figure(go.Bar(
        x=last.values,
        y=last.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in last.values],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        hovertemplate="<b>%{y}</b><br>Return: %{x:.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    fig = _apply_base(fig, title, height=340)
    fig.update_layout(yaxis=dict(tickfont=dict(size=11)))
    return fig


def heatmap_correlation(corr: pd.DataFrame,
                        title: str = "Correlation Matrix") -> go.Figure:
    """Correlation heatmap with diverging colorscale."""
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0,  ACCENT_RED],
            [0.5,  SURFACE_2],
            [1.0,  ACCENT_GREEN],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y} / %{x}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=TEXT_PRIMARY),
            outlinecolor=BORDER,
            thickness=12,
        ),
    ))
    return _apply_base(fig, title, height=400)


def area_volatility(vol: pd.DataFrame,
                    title: str = "Annualised Volatility (%)") -> go.Figure:
    """Stacked area chart of rolling volatility."""
    fig = go.Figure()
    for i, col in enumerate(vol.columns):
        s = vol[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=col,
            mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            fill="tozeroy",
            fillcolor=PALETTE[i % len(PALETTE)].replace(")", ", 0.08)").replace("rgb", "rgba"),
            hovertemplate=f"<b>{col}</b><br>Vol: %{{y:.1f}}%<extra></extra>",
        ))
    return _apply_base(fig, title)


# ── Macro charts ──────────────────────────────────────────────────────────────

def bar_macro_comparison(df: pd.DataFrame, indicator: str,
                         year: int = None,
                         title: str = "") -> go.Figure:
    """Horizontal bar chart comparing one indicator across countries."""
    sub = df[df["indicator"] == indicator].copy()
    if year:
        sub = sub[sub["year"] == year]
    else:
        sub = sub.sort_values("year").groupby("country").last().reset_index()

    sub = sub.dropna(subset=["value"]).sort_values("value", ascending=True)

    colors = []
    for v in sub["value"]:
        if indicator in ("GDP Growth (%)", "Current Account/GDP"):
            colors.append(ACCENT_GREEN if v >= 0 else ACCENT_RED)
        else:
            colors.append(ACCENT_BLUE)

    fig = go.Figure(go.Bar(
        x=sub["value"],
        y=sub["country"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in sub["value"]],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=BORDER, line_width=1)
    return _apply_base(fig, title or indicator, height=400)


def line_macro_trend(df: pd.DataFrame, indicator: str,
                     countries: list,
                     title: str = "") -> go.Figure:
    """Line chart of a macro indicator over time for selected countries."""
    sub = df[(df["indicator"] == indicator) &
             (df["country"].isin(countries))].copy()
    sub = sub.sort_values("year")

    fig = go.Figure()
    for i, country in enumerate(countries):
        s = sub[sub["country"] == country]
        fig.add_trace(go.Scatter(
            x=s["year"], y=s["value"],
            name=country,
            mode="lines+markers",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
        ))
    return _apply_base(fig, title or indicator)


def scatter_macro(df: pd.DataFrame, x_ind: str, y_ind: str,
                  year: int = None, title: str = "") -> go.Figure:
    """Scatter plot comparing two macro indicators across countries."""
    def get_latest(ind):
        sub = df[df["indicator"] == ind].copy()
        if year:
            return sub[sub["year"] == year].set_index("country")["value"]
        return sub.sort_values("year").groupby("country")["value"].last()

    x = get_latest(x_ind)
    y = get_latest(y_ind)
    common = x.index.intersection(y.index)

    fig = go.Figure(go.Scatter(
        x=x[common], y=y[common],
        mode="markers+text",
        text=common,
        textposition="top center",
        textfont=dict(size=10, color=TEXT_MUTED),
        marker=dict(size=10, color=ACCENT_BLUE,
                    line=dict(color=BORDER, width=1)),
        hovertemplate="<b>%{text}</b><br>"
                      f"{x_ind}: %{{x:.2f}}<br>"
                      f"{y_ind}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=x_ind, yaxis_title=y_ind)
    return _apply_base(fig, title or f"{x_ind} vs {y_ind}", height=420)


# ── Monetary policy charts ────────────────────────────────────────────────────

def line_fred(df: pd.DataFrame, series: list = None,
              title: str = "Central Bank Rates") -> go.Figure:
    """Line chart of FRED monetary policy series."""
    fig  = go.Figure()
    cols = series if series else df.columns.tolist()
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=col,
            mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}}%<extra></extra>",
        ))
    return _apply_base(fig, title)


def gauge_rate(value: float, label: str = "Rate",
               min_val: float = 0, max_val: float = 10) -> go.Figure:
    """Gauge indicator for a single rate."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title=dict(text=label, font=dict(color=TEXT_PRIMARY, size=13)),
        number=dict(suffix="%", font=dict(color=TEXT_PRIMARY, size=28)),
        gauge=dict(
            axis=dict(range=[min_val, max_val],
                      tickcolor=TEXT_MUTED,
                      tickfont=dict(color=TEXT_MUTED, size=10)),
            bar=dict(color=ACCENT_BLUE),
            bgcolor=SURFACE_2,
            borderwidth=1,
            bordercolor=BORDER,
            steps=[
                dict(range=[min_val, max_val * 0.33], color=SURFACE),
                dict(range=[max_val * 0.33, max_val * 0.66], color=SURFACE_2),
                dict(range=[max_val * 0.66, max_val], color=BORDER),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_PRIMARY),
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ── FX charts ─────────────────────────────────────────────────────────────────

def line_fx(fx: pd.DataFrame, pairs: list = None,
            normalise: bool = True,
            title: str = "FX Rates vs USD") -> go.Figure:
    """Line chart of FX pairs."""
    fig  = go.Figure()
    cols = pairs if pairs else fx.columns.tolist()
    df   = fx[cols].copy()
    if normalise:
        df = df.div(df.iloc[0]) * 100

    for i, col in enumerate(df.columns):
        s = df[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=col,
            mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"<b>{col}</b><br>%{{x|%d %b %Y}}<br>%{{y:.2f}}<extra></extra>",
        ))
    if normalise:
        fig.add_hline(y=100, line_dash="dot",
                      line_color=TEXT_MUTED, line_width=1)
    return _apply_base(fig, title)


# ── KPI card helper ───────────────────────────────────────────────────────────

def kpi_indicator(value: float, delta: float, label: str) -> go.Figure:
    """Small indicator figure for KPI cards."""
    color = ACCENT_GREEN if delta >= 0 else ACCENT_RED
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=value,
        delta=dict(reference=value - delta,
                   valueformat=".2f",
                   increasing=dict(color=ACCENT_GREEN),
                   decreasing=dict(color=ACCENT_RED)),
        title=dict(text=label,
                   font=dict(color=TEXT_MUTED, size=12)),
        number=dict(font=dict(color=TEXT_PRIMARY, size=26)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=110,
        margin=dict(l=12, r=12, t=24, b=8),
    )
    return fig
