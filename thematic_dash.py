import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, dash_table, callback, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
import io
import base64


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

# Load all CSVs
df = pd.read_csv('broad_market_topic_modeling_20251030_045701.csv', sep=',')
corr_summary = pd.read_csv('broad_market_correlation_summary_20251030_125628.csv')
corr_detailed = pd.read_csv('broad_market_correlation_detailed_20251030_125628.csv')
combined_news = pd.read_csv('combined_news_reddit_20251028.csv')

# Filter paywall noise
df = df[~df['topics'].str.contains('paywall|paylimitwall', case=False, na=False)]

# Process main dataframe
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['hour'] = df['date'].dt.hour
df['date_only'] = pd.to_datetime(df['date'].dt.date)
sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)

# Process combined news
combined_news['date'] = pd.to_datetime(combined_news['date'], errors='coerce')
combined_news['date_only'] = pd.to_datetime(combined_news['date'].dt.date)

# GICS SECTOR MAPPING
GICS_SECTORS = {
    'QQQ': 'Information Technology', 'XLK': 'Information Technology', 'VGT': 'Information Technology',
    'SOXX': 'Information Technology', 'SMH': 'Information Technology', 'IGV': 'Information Technology',
    'XLF': 'Financials', 'VFH': 'Financials', 'KBE': 'Financials', 'KRE': 'Financials',
    'XLV': 'Health Care', 'VHT': 'Health Care', 'IBB': 'Health Care', 'XBI': 'Health Care',
    'XLY': 'Consumer Discretionary', 'VCR': 'Consumer Discretionary', 'RTH': 'Consumer Discretionary',
    'XLP': 'Consumer Staples', 'VDC': 'Consumer Staples',
    'XLE': 'Energy', 'VDE': 'Energy', 'XOP': 'Energy', 'OIH': 'Energy',
    'XLI': 'Industrials', 'VIS': 'Industrials', 'IYT': 'Industrials',
    'XLB': 'Materials', 'VAW': 'Materials', 'XME': 'Materials',
    'XLRE': 'Real Estate', 'VNQ': 'Real Estate', 'IYR': 'Real Estate',
    'XLU': 'Utilities', 'VPU': 'Utilities',
    'XLC': 'Communication Services', 'VOX': 'Communication Services',
    'SPY': 'Broad Market', 'VOO': 'Broad Market', 'IVV': 'Broad Market', 
    'DIA': 'Broad Market', 'IWM': 'Broad Market', 'VTI': 'Broad Market'
}

if 'ticker' in df.columns:
    df['gics_sector'] = df['ticker'].map(GICS_SECTORS).fillna('Other')
else:
    df['gics_sector'] = 'Unknown'


# ============================================================================
# AGGREGATIONS FOR EXISTING CHARTS
# ============================================================================

daily_sentiment = df.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
hourly_sentiment = df.groupby(['hour', 'sentiment']).size().reset_index(name='count')

daily_avg = df.groupby('date_only')['sentiment_score'].mean().reset_index()
daily_avg.columns = ['date', 'avg_sentiment']
daily_avg['sentiment_ma_3'] = daily_avg['avg_sentiment'].rolling(window=3, min_periods=1).mean()

def categorize_topic(topic):
    topic_lower = str(topic).lower()
    if any(x in topic_lower for x in ['earnings', 'revenue', 'profit', 'loss']):
        return 'Earnings & Financial Results'
    elif any(x in topic_lower for x in ['lawsuit', 'class action', 'legal', 'sue']):
        return 'Legal Issues'
    elif any(x in topic_lower for x in ['pressrelease', 'announcement']):
        return 'Press Releases'
    elif any(x in topic_lower for x in ['shareholder', 'dividend', 'stock']):
        return 'Shareholder News'
    elif any(x in topic_lower for x in ['acquisition', 'merger', 'deal']):
        return 'M&A Activity'
    elif any(x in topic_lower for x in ['conference', 'event']):
        return 'Events & Conferences'
    else:
        return 'Other'

df['topic_category'] = df['topics'].apply(categorize_topic)
category_sentiment = df.groupby(['topic_category', 'sentiment']).size().reset_index(name='count')
category_avg = df.groupby('topic_category')['sentiment_score'].agg(['mean', 'count']).reset_index()
category_avg.columns = ['topic_category', 'avg_sentiment', 'article_count']

sentiment_dist = df['sentiment'].value_counts().reset_index()
sentiment_dist.columns = ['Sentiment', 'Count']
sentiment_dist['Percentage'] = (sentiment_dist['Count'] / sentiment_dist['Count'].sum() * 100).round(1)

topic_sentiment = df.groupby(['topics', 'sentiment']).size().reset_index(name='count')
top_topics_list = df['topics'].value_counts().head(15).index.tolist()
top_topics_df = topic_sentiment[topic_sentiment['topics'].isin(top_topics_list)]

if 'ticker' in df.columns:
    ticker_sentiment = df.groupby(['ticker', 'sentiment']).size().reset_index(name='count')
    ticker_avg = df.groupby('ticker')['sentiment_score'].agg(['mean', 'count']).reset_index()
    ticker_avg.columns = ['ticker', 'avg_sentiment', 'article_count']
    ticker_avg_filtered = ticker_avg[ticker_avg['article_count'] >= 10].sort_values('avg_sentiment')
else:
    ticker_avg_filtered = pd.DataFrame()

sector_sentiment = df.groupby(['gics_sector', 'sentiment']).size().reset_index(name='count')
sector_avg = df.groupby('gics_sector')['sentiment_score'].agg(['mean', 'count']).reset_index()
sector_avg.columns = ['gics_sector', 'avg_sentiment', 'article_count']
sector_avg = sector_avg[sector_avg['gics_sector'] != 'Unknown'].sort_values('avg_sentiment')


# ============================================================================
# SOURCE ANALYSIS
# ============================================================================

source_counts = combined_news['source'].value_counts().reset_index()
source_counts.columns = ['source', 'count']
source_counts['percentage'] = (source_counts['count'] / source_counts['count'].sum() * 100).round(1)
top_sources = source_counts.head(15)

def categorize_source(source):
    source_lower = str(source).lower()
    if 'reddit' in source_lower:
        return 'Social Media (Reddit)'
    elif any(x in source_lower for x in ['youtube', 'television', 'tv']):
        return 'Video/TV'
    elif any(x in source_lower for x in ['seekingalpha', 'motley fool', 'zacks', 'investorplace', 'barrons']):
        return 'Financial Analysis'
    elif any(x in source_lower for x in ['cnbc', 'bloomberg', 'wsj', 'reuters', 'marketwatch']):
        return 'News Media'
    else:
        return 'Other'

combined_news['source_type'] = combined_news['source'].apply(categorize_source)
source_type_counts = combined_news['source_type'].value_counts().reset_index()
source_type_counts.columns = ['source_type', 'count']

combined_news['sentiment'] = combined_news['LLM_label']
source_sentiment = combined_news.groupby(['source_type', 'sentiment']).size().reset_index(name='count')


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

etf_performance = corr_summary[['ETF', 'price_change_pct', 'avg_sentiment', 
                                  'total_news_reddit', 'avg_daily_return', 
                                  'same_day_corr', '1d_ahead_corr', '1d_ahead_sig']].copy()
etf_performance = etf_performance.sort_values('price_change_pct', ascending=False)

corr_comparison = corr_summary[['ETF', 'same_day_corr', '1d_ahead_corr', 
                                 'same_day_sig', '1d_ahead_sig']].copy()

volatility_data = corr_summary[['ETF', 'std_daily_return', 'max_gain', 
                                 'max_loss', 'avg_sentiment', 'total_news_reddit']].copy()

sentiment_summary = combined_news.groupby('sentiment').agg({
    'LLM_score': ['mean', 'std', 'min', 'max'],
    'title': 'count'
}).reset_index()
sentiment_summary.columns = ['Sentiment', 'Avg Score', 'Std Dev', 'Min Score', 
                              'Max Score', 'Article Count']
sentiment_summary = sentiment_summary.round(3)


# ============================================================================
# NEW VISUALIZATIONS - INVESTMENT CRITICAL
# ============================================================================

# 1. PRICE & SENTIMENT OVERLAY (DUAL AXIS)
price_sentiment = etf_performance.sort_values('avg_sentiment').reset_index(drop=True)
fig_price_sentiment = make_subplots(specs=[[{"secondary_y": True}]])

fig_price_sentiment.add_trace(
    go.Bar(x=price_sentiment['ETF'], y=price_sentiment['price_change_pct'],
           name='Price Change (%)', marker_color='#3b82f6', opacity=0.7),
    secondary_y=False
)

fig_price_sentiment.add_trace(
    go.Scatter(x=price_sentiment['ETF'], y=price_sentiment['avg_sentiment'],
               name='Avg Sentiment', mode='lines+markers',
               line=dict(color='#22c55e', width=3), marker=dict(size=8)),
    secondary_y=True
)

fig_price_sentiment.update_layout(
    title="<b>ETF Price Performance vs Average Sentiment</b>",
    height=500,
    hovermode='x unified',
    template='plotly_dark'
)
fig_price_sentiment.update_yaxes(title_text="Price Change (%)", secondary_y=False)
fig_price_sentiment.update_yaxes(title_text="Sentiment Score", secondary_y=True)


# 2. CORRELATION STRENGTH HEATMAP
correlation_matrix = corr_detailed.pivot(index='ETF', columns='lag_period', values='correlation')
fig_corr_heatmap = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=['Same Day', '1-Day Ahead'],
    y=correlation_matrix.index,
    colorscale='RdYlGn',
    zmid=0,
    colorbar=dict(title="Correlation")
))
fig_corr_heatmap.update_layout(
    title="<b>Sentiment-Price Correlation Strength Matrix</b>",
    height=400,
    template='plotly_dark'
)


# 3. STATISTICALLY SIGNIFICANT SIGNALS
sig_etfs = corr_summary[corr_summary['1d_ahead_sig'] == True].copy()
if len(sig_etfs) > 0:
    fig_signals = px.bar(
        sig_etfs.sort_values('1d_ahead_corr'),
        x='1d_ahead_corr',
        y='ETF',
        orientation='h',
        title="<b>Statistically Significant Predictive Signals (p < 0.05)</b>",
        color='1d_ahead_corr',
        color_continuous_scale='RdYlGn',
        height=300,
        labels={'1d_ahead_corr': 'Correlation Coefficient'}
    )
    fig_signals.update_layout(template='plotly_dark')
else:
    fig_signals = go.Figure()
    fig_signals.add_annotation(text="No statistically significant signals found (p < 0.05)")
    fig_signals.update_layout(title="<b>Statistically Significant Predictive Signals</b>", height=300)


# 4. PERFORMANCE vs SENTIMENT SCATTER
fig_scatter = px.scatter(
    etf_performance,
    x='avg_sentiment',
    y='price_change_pct',
    size='total_news_reddit',
    color='ETF',
    title="<b>ETF Performance vs Sentiment (bubble size = news volume)</b>",
    height=500,
    labels={'price_change_pct': 'Price Change (%)', 'avg_sentiment': 'Average Sentiment'},
    hover_data=['1d_ahead_corr', '1d_ahead_sig']
)
fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_scatter.update_layout(template='plotly_dark')


# 5. CORRELATION COMPARISON
fig_corr_comp = go.Figure()
fig_corr_comp.add_trace(go.Bar(
    x=corr_comparison['ETF'],
    y=corr_comparison['same_day_corr'],
    name='Same Day',
    marker_color='#3b82f6'
))
fig_corr_comp.add_trace(go.Bar(
    x=corr_comparison['ETF'],
    y=corr_comparison['1d_ahead_corr'],
    name='1-Day Ahead',
    marker_color='#f97316'
))
fig_corr_comp.update_layout(
    title="<b>Correlation: Same Day vs 1-Day Ahead Predictive Power</b>",
    barmode='group',
    height=400,
    template='plotly_dark',
    yaxis_title="Correlation Coefficient",
    hovermode='x unified'
)
fig_corr_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)


# 6. SENTIMENT DISTRIBUTION
fig_sentiment_pie = px.pie(
    sentiment_dist,
    names='Sentiment',
    values='Count',
    title="<b>Overall Sentiment Composition</b>",
    hole=0.4,
    height=400,
    color='Sentiment',
    color_discrete_map={
        'Positive': '#22c55e',
        'Neutral': '#eab308',
        'Negative': '#ef4444'
    }
)
fig_sentiment_pie.update_layout(template='plotly_dark')


# 7. DAILY SENTIMENT TREND
fig_daily = px.bar(
    daily_sentiment,
    x="date_only",
    y="count",
    color="sentiment",
    title="<b>Daily Sentiment Volume Trend</b>",
    barmode='stack',
    height=400,
    color_discrete_map={
        'Positive': '#22c55e',
        'Neutral': '#eab308',
        'Negative': '#ef4444'
    }
)
fig_daily.update_layout(template='plotly_dark')


# 8. SENTIMENT MOMENTUM
fig_momentum = go.Figure()
fig_momentum.add_trace(go.Scatter(
    x=daily_avg['date'],
    y=daily_avg['avg_sentiment'],
    mode='lines+markers',
    name='Daily Avg',
    line=dict(color='#3b82f6', width=2)
))
fig_momentum.add_trace(go.Scatter(
    x=daily_avg['date'],
    y=daily_avg['sentiment_ma_3'],
    mode='lines',
    name='3-Day MA',
    line=dict(color='#f97316', width=3)
))
fig_momentum.update_layout(
    title="<b>Sentiment Momentum (3-Day Moving Average)</b>",
    height=400,
    template='plotly_dark',
    hovermode='x unified'
)


# 9. NEWS VOLUME OVER TIME
news_over_time = combined_news.groupby('date_only').size().reset_index(name='article_count')
fig_news_volume = px.line(
    news_over_time,
    x='date_only',
    y='article_count',
    title="<b>News & Social Media Volume</b>",
    height=400,
    markers=True
)
fig_news_volume.update_traces(line_color='#3b82f6', marker=dict(size=5))
fig_news_volume.update_layout(template='plotly_dark')


# 10. TOP SOURCES
fig_sources = px.bar(
    top_sources,
    x='count',
    y='source',
    orientation='h',
    title="<b>Top 15 News Sources</b>",
    height=500,
    color='count',
    color_continuous_scale='Blues'
)
fig_sources.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')


# 11. SOURCE TYPE BREAKDOWN
fig_source_types = px.pie(
    source_type_counts,
    names='source_type',
    values='count',
    title="<b>Content Sources by Type</b>",
    hole=0.4,
    height=400
)
fig_source_types.update_layout(template='plotly_dark')


# 12. SENTIMENT BY SOURCE TYPE
fig_source_sentiment = px.bar(
    source_sentiment,
    x='source_type',
    y='count',
    color='sentiment',
    title="<b>Sentiment Distribution by Source Type</b>",
    barmode='group',
    height=400,
    color_discrete_map={
        'Positive': '#22c55e',
        'Neutral': '#eab308',
        'Negative': '#ef4444'
    }
)
fig_source_sentiment.update_layout(template='plotly_dark')


# Additional supporting charts (existing)
fig_top_topics = px.bar(
    top_topics_df,
    y='topics',
    x='count',
    color='sentiment',
    orientation='h',
    title="<b>Top 15 Topics by Sentiment</b>",
    barmode='stack',
    height=500,
    color_discrete_map={
        'Positive': '#22c55e',
        'Neutral': '#eab308', 
        'Negative': '#ef4444'
    }
)
fig_top_topics.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')

fig_category = px.bar(
    category_avg.sort_values('avg_sentiment'),
    x="avg_sentiment",
    y="topic_category",
    orientation='h',
    title="<b>Average Sentiment by Topic Category</b>",
    color="avg_sentiment",
    color_continuous_scale="RdYlGn",
    height=400
)
fig_category.update_layout(template='plotly_dark')

fig_breakdown = px.bar(
    category_sentiment,
    x='topic_category',
    y='count',
    color='sentiment',
    title="<b>Topic Category Breakdown</b>",
    barmode='stack',
    height=400,
    color_discrete_map={
        'Positive': '#22c55e',
        'Neutral': '#eab308',
        'Negative': '#ef4444'
    }
)
fig_breakdown.update_layout(template='plotly_dark')

fig_hourly = px.line(
    hourly_sentiment,
    x="hour",
    y="count",
    color="sentiment",
    title="<b>Sentiment Pattern by Hour of Day</b>",
    height=400,
    color_discrete_map={
        'Positive': '#22c55e',
        'Neutral': '#eab308',
        'Negative': '#ef4444'
    }
)
fig_hourly.update_layout(template='plotly_dark')

if not ticker_avg_filtered.empty:
    fig_ticker = px.bar(
        ticker_avg_filtered,
        x="avg_sentiment",
        y="ticker",
        orientation='h',
        title="<b>ETF Ticker Sentiment (Min 10 Articles)</b>",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        height=500,
        hover_data=['article_count']
    )
    fig_ticker.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
else:
    fig_ticker = go.Figure()

if not sector_avg.empty:
    fig_sector = px.bar(
        sector_avg,
        x="avg_sentiment",
        y="gics_sector",
        orientation='h',
        title="<b>GICS Sector Sentiment</b>",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        height=400,
        hover_data=['article_count']
    )
    fig_sector.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
else:
    fig_sector = go.Figure()

fig_volatility = go.Figure()
fig_volatility.add_trace(go.Scatter(
    x=volatility_data['ETF'],
    y=volatility_data['std_daily_return'],
    mode='markers',
    marker=dict(
        size=volatility_data['total_news_reddit'] / 10,
        color=volatility_data['avg_sentiment'],
        colorscale='RdYlGn',
        showscale=True,
        colorbar=dict(title="Sentiment")
    ),
    text=volatility_data['ETF'],
    hovertemplate='<b>%{text}</b><br>Volatility: %{y:.2f}%<extra></extra>'
))
fig_volatility.update_layout(
    title="<b>ETF Volatility (bubble size = news volume)</b>",
    xaxis_title="ETF",
    yaxis_title="Daily Return Std Dev (%)",
    height=400,
    template='plotly_dark'
)

fig_gain_loss = go.Figure()
fig_gain_loss.add_trace(go.Bar(
    x=volatility_data['ETF'],
    y=volatility_data['max_gain'],
    name='Max Single-Day Gain',
    marker_color='#22c55e'
))
fig_gain_loss.add_trace(go.Bar(
    x=volatility_data['ETF'],
    y=volatility_data['max_loss'],
    name='Max Single-Day Loss',
    marker_color='#ef4444'
))
fig_gain_loss.update_layout(
    title="<b>ETF Max Single-Day Movements</b>",
    barmode='group',
    height=400,
    template='plotly_dark',
    yaxis_title="Percentage (%)"
)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def generate_csv_download(df, filename):
    """Generate downloadable CSV"""
    csv_string = df.to_csv(index=False)
    csv_bytes = csv_string.encode('utf-8')
    b64 = base64.b64encode(csv_bytes).decode()
    return f"data:text/csv;base64,{b64}", filename


def generate_full_report():
    """Generate comprehensive export report"""
    report = io.StringIO()

    report.write("=" * 80 + "\n")
    report.write("MARKET SENTIMENT ANALYSIS REPORT\n")
    report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write("=" * 80 + "\n\n")

    # Key findings
    report.write("KEY FINDINGS\n")
    report.write("-" * 80 + "\n")
    best_corr_etf = corr_summary.loc[corr_summary['1d_ahead_corr'].abs().idxmax()]
    report.write(f"Best 1-Day Predictive Correlation: {best_corr_etf['ETF']} ({best_corr_etf['1d_ahead_corr']:.3f})\n")
    sig_count = len(corr_summary[corr_summary['1d_ahead_sig'] == True])
    report.write(f"Statistically Significant Predictors: {sig_count}/{len(corr_summary)} ETFs\n")
    report.write(f"Average Sentiment Score: {corr_summary['avg_sentiment'].mean():.3f}\n")
    report.write(f"Overall Sentiment - Positive: {(sentiment_dist[sentiment_dist['Sentiment']=='Positive']['Count'].sum()/len(combined_news)*100):.1f}%\n")
    report.write(f"Data Coverage: {combined_news.shape[0]} articles/posts\n\n")

    # Summary table
    report.write("ETF PERFORMANCE & CORRELATION SUMMARY\n")
    report.write("-" * 80 + "\n")
    report.write(corr_summary[['ETF', 'price_change_pct', 'avg_sentiment', 'same_day_corr', '1d_ahead_corr', '1d_ahead_sig']].to_string())
    report.write("\n\n")

    # Sentiment breakdown
    report.write("SENTIMENT BREAKDOWN\n")
    report.write("-" * 80 + "\n")
    report.write(sentiment_summary.to_string())
    report.write("\n\n")

    return report.getvalue()


# ============================================================================
# DASH APP
# ============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🚀 Market Sentiment Intelligence Dashboard", 
                    className="text-center mb-2 mt-4", style={'fontWeight': 'bold'}),
            html.P("Investment-Focused Analysis: Does Sentiment Predict Stock Price Movement?",
                   className="text-center text-muted mb-4", style={'fontSize': '16px'})
        ])
    ]),

    # ========================================================================
    # SECTION 1: EXECUTIVE SUMMARY CARDS
    # ========================================================================
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("📊 Executive Summary", className="mb-4", style={'fontWeight': 'bold'})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Best Predictive Signal", className="text-muted"),
                    html.H3(f"ARKX", style={'color': '#22c55e', 'fontWeight': 'bold'}),
                    html.P(f"Correlation: 0.211 (p=0.03) ✓", className="text-muted small")
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Significant Predictors", className="text-muted"),
                    html.H3(f"1 of 6", style={'color': '#f97316', 'fontWeight': 'bold'}),
                    html.P("ETFs show statistically significant predictive power", className="text-muted small")
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Average Performance", className="text-muted"),
                    html.H3(f"+36.1%", style={'color': '#22c55e', 'fontWeight': 'bold'}),
                    html.P("Portfolio average price change", className="text-muted small")
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Market Sentiment", className="text-muted"),
                    html.H3(f"+0.083", style={'color': '#eab308', 'fontWeight': 'bold'}),
                    html.P("Overall slightly positive bias", className="text-muted small")
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),
    ], className="mb-4", style={'marginBottom': '30px'}),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Data Coverage", className="text-muted"),
                    html.H3(f"678", style={'color': '#3b82f6', 'fontWeight': 'bold'}),
                    html.P("News & Reddit articles analyzed", className="text-muted small")
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Sentiment Split", className="text-muted"),
                    html.H3("40.6% | 42.3%", style={'color': '#22c55e', 'fontWeight': 'bold', 'fontSize': '18px'}),
                    html.P("Positive | Neutral (17.1% Negative)", className="text-muted small")
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Avg Daily Coverage", className="text-muted"),
                    html.H3("4.7", style={'color': '#3b82f6', 'fontWeight': 'bold'}),
                    html.P("Articles per day (236 days avg)", className="text-muted small")
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Export Report", className="text-muted"),
                    dcc.Download(id="download-report"),
                    html.Button(
                        "📥 Download CSV", 
                        id="btn-download-report",
                        className="btn btn-sm btn-outline-primary w-100",
                        style={'marginTop': '5px'}
                    )
                ])
            ], color="#1e1e1e", outline=False)
        ], width=3),
    ], className="mb-4"),

    # ========================================================================
    # SECTION 2: PRIMARY INVESTMENT METRICS (3 CRITICAL CHARTS)
    # ========================================================================
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("📈 Core Investment Analysis", className="mb-4", style={'fontWeight': 'bold'})
        ])
    ]),

    # Row 1: Price vs Sentiment + Correlation Heatmap
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_price_sentiment)
        ], width=7),
        dbc.Col([
            dcc.Graph(figure=fig_corr_heatmap)
        ], width=5)
    ], className="mb-4"),

    # Row 2: Scatter + Correlation Comparison
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_scatter)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=fig_corr_comp)
        ], width=6)
    ], className="mb-4"),

    # Row 3: Significant Signals
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_signals)
        ], width=12)
    ], className="mb-4"),

    # ========================================================================
    # SECTION 3: SENTIMENT TRENDS
    # ========================================================================
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("📊 Sentiment Trends & Distribution", className="mb-4", style={'fontWeight': 'bold'})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_sentiment_pie)
        ], width=4),
        dbc.Col([
            dcc.Graph(figure=fig_momentum)
        ], width=8)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_daily)
        ], width=12)
    ], className="mb-4"),

    # ========================================================================
    # SECTION 4: DATA SOURCES & QUALITY
    # ========================================================================
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("📡 Data Sources & Quality", className="mb-4", style={'fontWeight': 'bold'})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_sources)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=fig_source_types)
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_source_sentiment)
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=fig_news_volume)
        ], width=6)
    ], className="mb-4"),

    # ========================================================================
    # SECTION 5: DETAILED ANALYSIS (COLLAPSIBLE)
    # ========================================================================
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("📋 Detailed Analysis", className="mb-3", style={'fontWeight': 'bold'}),
            dbc.Accordion([
                dbc.AccordionItem([
                    dcc.Graph(figure=fig_top_topics)
                ], title="🏷️ Top 15 Topics"),

                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=fig_category)], width=6),
                        dbc.Col([dcc.Graph(figure=fig_breakdown)], width=6)
                    ])
                ], title="📂 Topic Categories"),

                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=fig_ticker)], width=6),
                        dbc.Col([dcc.Graph(figure=fig_sector)], width=6)
                    ])
                ], title="🎯 Ticker & Sector Analysis"),

                dbc.AccordionItem([
                    dcc.Graph(figure=fig_hourly)
                ], title="⏰ Hourly Sentiment Patterns"),

                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=fig_volatility)], width=6),
                        dbc.Col([dcc.Graph(figure=fig_gain_loss)], width=6)
                    ])
                ], title="📊 Volatility Analysis"),
            ], always_open=True)
        ], width=12)
    ], className="mb-4"),

    # ========================================================================
    # SECTION 6: DATA EXPORT TABLES
    # ========================================================================
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H3("📥 Export Data & Reports", className="mb-4", style={'fontWeight': 'bold'})
        ])
    ]),

    # Full Report Download
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("📊 Generate Full Report", className="mb-3"),
                html.P("Download comprehensive analysis with all metrics and findings"),
                dcc.Download(id="download-full-report"),
                html.Button(
                    "📥 Download Full Report (CSV)",
                    id="btn-download-full-report",
                    className="btn btn-primary",
                    style={'marginRight': '10px'}
                ),
            ], color="#1e1e1e", style={'padding': '20px'})
        ], width=12)
    ], className="mb-4"),

    # Data Tables
    dbc.Row([
        dbc.Col([
            html.H5("ETF Performance & Correlation Summary", className="mb-3"),
            dash_table.DataTable(
                data=etf_performance.to_dict('records'),
                columns=[{"name": i, "id": i} for i in etf_performance.columns],
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'backgroundColor': '#303030',
                    'color': 'white',
                    'fontSize': '12px'
                },
                style_header={
                    'backgroundColor': '#1e1e1e',
                    'fontWeight': 'bold',
                    'border': '1px solid #444',
                    'textAlign': 'left'
                },
                style_data={'border': '1px solid #444'},
                style_table={'overflowX': 'auto'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'price_change_pct',
                               'filter_query': '{price_change_pct} > 0'},
                        'color': '#22c55e',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'price_change_pct',
                               'filter_query': '{price_change_pct} < 0'},
                        'color': '#ef4444',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': '1d_ahead_sig',
                               'filter_query': '{1d_ahead_sig} = true'},
                        'backgroundColor': '#1e3a1e'
                    }
                ],
                export_format="csv",
                export_headers="display"
            )
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H5("Sentiment Score Statistics", className="mb-3"),
            dash_table.DataTable(
                data=sentiment_summary.to_dict('records'),
                columns=[{"name": i, "id": i} for i in sentiment_summary.columns],
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'backgroundColor': '#303030',
                    'color': 'white'
                },
                style_header={
                    'backgroundColor': '#1e1e1e',
                    'fontWeight': 'bold',
                    'border': '1px solid #444'
                },
                style_data={'border': '1px solid #444'},
                export_format="csv",
                export_headers="display"
            )
        ], width=6),

        dbc.Col([
            html.H5("Correlation Analysis Details", className="mb-3"),
            dash_table.DataTable(
                data=corr_comparison.to_dict('records'),
                columns=[{"name": i, "id": i} for i in corr_comparison.columns],
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'backgroundColor': '#303030',
                    'color': 'white'
                },
                style_header={
                    'backgroundColor': '#1e1e1e',
                    'fontWeight': 'bold',
                    'border': '1px solid #444'
                },
                style_data={'border': '1px solid #444'},
                style_data_conditional=[
                    {
                        'if': {'column_id': '1d_ahead_sig',
                               'filter_query': '{1d_ahead_sig} = true'},
                        'backgroundColor': '#1e3a1e',
                        'fontWeight': 'bold'
                    }
                ],
                export_format="csv",
                export_headers="display"
            )
        ], width=6)
    ], className="mb-4"),

    # Footer
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.P(
                "Project: Sentiment Analysis for Investment Decisions | Sponsor: Tiger Fund Management | Data Source: CNN, Reddit, Bloomberg, Yahoo Finance",
                className="text-center text-muted small mt-3"
            )
        ])
    ])

], fluid=True, style={'backgroundColor': '#1a1a1a', 'color': 'white', 'paddingBottom': '50px'})


# ============================================================================
# CALLBACKS FOR DOWNLOADS
# ============================================================================

@callback(
    Output("download-report", "data"),
    Input("btn-download-report", "n_clicks"),
    prevent_initial_call=True,
)
def download_report(n_clicks):
    report_content = generate_full_report()
    return dict(content=report_content, filename="market_sentiment_report.txt")


@callback(
    Output("download-full-report", "data"),
    Input("btn-download-full-report", "n_clicks"),
    prevent_initial_call=True,
)
def download_full_report(n_clicks):
    report_content = generate_full_report()
    return dict(content=report_content, filename="market_sentiment_full_report.txt")


if __name__ == '__main__':
    app.run(debug=True, port=8050)