import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import ast

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------

# Load both FinBERT outputs
file1 = "fin_news_clean-2-finBERT-1.xlsx"
file2 = "finbert_output_20250927-1.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# ---- Process File 1 ----
# Extract sentiment scores from finbert_probs dictionary
if 'finbert_probs' in df1.columns:
    df1['finbert_probs'] = df1['finbert_probs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df1['positive'] = df1['finbert_probs'].apply(lambda x: x.get('positive', 0) if isinstance(x, dict) else 0)
    df1['negative'] = df1['finbert_probs'].apply(lambda x: x.get('negative', 0) if isinstance(x, dict) else 0)
    df1['neutral'] = df1['finbert_probs'].apply(lambda x: x.get('neutral', 0) if isinstance(x, dict) else 0)

# Calculate compound score for file1 (positive - negative)
df1['compound'] = df1['positive'] - df1['negative']

# Rename sentiment_label column
if 'finbert_label' in df1.columns:
    df1['sentiment_label'] = df1['finbert_label']

# ---- Process File 2 ----
# Rename columns to match
df2 = df2.rename(columns={
    'finbert_positive': 'positive',
    'finbert_negative': 'negative',
    'finbert_neutral': 'neutral',
    'finbert_score': 'compound',
    'finbert_label': 'sentiment_label',
    'created_utc': 'date'
})

# Combine both data sources
df = pd.concat([df1, df2], ignore_index=True)

# FIX TIMEZONE ISSUE: Convert to UTC first, then remove timezone info
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df['date'] = df['date'].dt.tz_localize(None)  # Remove timezone info
df = df.dropna(subset=['date'])

# Extract just the date (remove time component for daily aggregation)
df['date'] = df['date'].dt.date
df['date'] = pd.to_datetime(df['date'])

# Ensure expected FinBERT sentiment columns exist
cols = ['positive', 'neutral', 'negative', 'compound']
for c in cols:
    if c not in df.columns:
        raise ValueError(f"Column {c} not found in dataset")

# Create sentiment label if not given
if 'sentiment_label' not in df.columns:
    df['sentiment_label'] = df['compound'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    )

# Aggregate to market level by day
daily_sentiment = df.groupby('date')[cols].mean().reset_index()

# Add sentiment percentages
sentiment_daily_counts = df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
sentiment_pct = sentiment_daily_counts.pivot(index='date', columns='sentiment_label', values='count').fillna(0)
sentiment_pct = sentiment_pct.div(sentiment_pct.sum(axis=1), axis=0).reset_index()

# Merge with compound mean
broad_df = pd.merge(daily_sentiment, sentiment_pct, on='date', how='outer').fillna(0)
broad_df = broad_df.sort_values('date')

# -----------------------------
# Plotly Figures
# -----------------------------

# 1. Sentiment Gauge (latest compound)
latest_val = broad_df['compound'].iloc[-1]
prev_val = broad_df['compound'].iloc[-2] if len(broad_df) > 1 else 0

gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=latest_val,
    delta={"reference": prev_val, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
    title={"text": "Market Sentiment Index"},
    gauge={"axis": {"range": [-1, 1]},
           "bar": {"color": "gold"},
           "steps": [
               {"range": [-1, -0.2], "color": "red"},
               {"range": [-0.2, 0.2], "color": "gray"},
               {"range": [0.2, 1], "color": "green"}]}
))

# 2. Sentiment Trend Line
trend_fig = px.line(broad_df, x="date", y="compound", title="Average Compound Sentiment Over Time",
                    labels={"compound": "Compound Sentiment Score", "date": "Date"})
trend_fig.add_hline(y=0, line_dash="dot", line_color="gray")

# 3. Sentiment Distribution (stacked area)
stacked_fig = go.Figure()
for label in ['positive', 'neutral', 'negative']:
    if label in broad_df.columns:
        stacked_fig.add_trace(go.Scatter(
            x=broad_df['date'],
            y=broad_df[label],
            mode='lines',
            stackgroup='one',
            name=label.capitalize()
        ))
stacked_fig.update_layout(title="Sentiment Composition Over Time", yaxis_title="Proportion", xaxis_title="Date")

# 4. Sector/Category Sentiment (if available)
if 'category' in df.columns:
    sector_df = df.groupby('category')['compound'].mean().reset_index()
    sector_fig = px.bar(sector_df, x='category', y='compound', title='Average Sentiment by Category',
                        color='compound', color_continuous_scale='RdYlGn')
else:
    sector_fig = go.Figure()
    sector_fig.update_layout(title="Category Sentiment (category column not detected)")

# -----------------------------
# DASH Layout
# -----------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

app.layout = dbc.Container([
    html.H2("Broad Market Sentiment Dashboard", className="text-center mt-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=gauge_fig), width=4),
        dbc.Col(dcc.Graph(figure=trend_fig), width=8),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=stacked_fig), width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=sector_fig), width=12)
    ], className="mb-4"),

    html.Footer("Data derived from FinBERT NLP sentiment model on financial news & discussions.",
                style={"textAlign": "center", "color": "gray"})
], fluid=True)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
