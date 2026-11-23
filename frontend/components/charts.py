import plotly.express as px
import plotly.graph_objects as go


DARK_TEMPLATE = "plotly_dark"
ACCENT_BLUE = "#3b82f6"
ACCENT_TEAL = "#14b8a6"
GREEN = "#22c55e"


def line_chart(df, x, y, title, color=ACCENT_BLUE, x_title=None, y_title=None):
    fig = px.line(df, x=x, y=y, title=title, template=DARK_TEMPLATE, color_discrete_sequence=[color])
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=x_title or x,
        yaxis_title=y_title or y,
    )
    return fig


def bar_chart(df, x, y, title, color=ACCENT_BLUE, x_title=None, y_title=None):
    fig = px.bar(df, x=x, y=y, title=title, template=DARK_TEMPLATE, color_discrete_sequence=[color])
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=x_title or x,
        yaxis_title=y_title or y,
    )
    return fig


def scatter_chart(df, x, y, color_col, title, x_title=None, y_title=None):
    fig = px.scatter(
        df, x=x, y=y, color=color_col, title=title,
        template=DARK_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=x_title or x,
        yaxis_title=y_title or y,
    )
    return fig


def probability_bar(labels, probs, title="Prediction Probabilities"):
    colors = [ACCENT_BLUE if p == max(probs) else "#334155" for p in probs]
    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color=colors, text=[f"{p:.1%}" for p in probs], textposition="outside"))
    fig.update_layout(
        title=title,
        template=DARK_TEMPLATE,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        yaxis=dict(range=[0, 1.1], tickformat=".0%"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
