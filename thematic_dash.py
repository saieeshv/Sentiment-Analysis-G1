
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

df = pd.read_csv('broad_market_topic_modeling_20251030_045701.csv', sep=',')
corr_summary = pd.read_csv('broad_market_correlation_summary_20251030_125628.csv')
corr_detailed = pd.read_csv('broad_market_correlation_detailed_20251030_125628.csv')
combined_news = pd.read_csv('combined_news_reddit_20251028.csv')

df = df[~df['topics'].str.contains('paywall|paylimitwall', case=False, na=False)]

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['hour'] = df['date'].dt.hour
df['date_only'] = pd.to_datetime(df['date'].dt.date)
sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)

combined_news['date'] = pd.to_datetime(combined_news['date'], errors='coerce')
combined_news['date_only'] = pd.to_datetime(combined_news['date'].dt.date)

# ============================================================================
# FIX: CREATE SENTIMENT COLUMN EARLY
# ============================================================================
combined_news['sentiment'] = combined_news['LLM_label']

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
# DATA FRESHNESS CALCULATIONS
# ============================================================================

today = pd.Timestamp.now(tz='UTC')
combined_news['days_old'] = (today - combined_news['date']).dt.days

def get_freshness_status(days):
    if days <= 7:
        return "Fresh", "#22c55e"
    elif days <= 30:
        return "Recent", "#eab308"
    elif days <= 60:
        return "Aging", "#f97316"
    else:
        return "Stale", "#ef4444"

# ============================================================================
# TIME FILTER FUNCTION
# ============================================================================

MIN_ARTICLE_THRESHOLD = 10

def filter_data_by_timeframe(timeframe):
    if timeframe == 'all':
        filtered_news = combined_news.copy()
        filtered_df = df.copy()
    elif timeframe == '1m':
        cutoff = today - pd.Timedelta(days=30)
        filtered_news = combined_news[combined_news['date'] >= cutoff].copy()
        filtered_df = df[df['date'] >= cutoff].copy()
    elif timeframe == '3m':
        cutoff = today - pd.Timedelta(days=90)
        filtered_news = combined_news[combined_news['date'] >= cutoff].copy()
        filtered_df = df[df['date'] >= cutoff].copy()
    else:
        filtered_news = combined_news.copy()
        filtered_df = df.copy()

    ticker_volume = filtered_news.groupby('ticker').size().reset_index(name='article_count')
    ticker_volume['sufficient_data'] = ticker_volume['article_count'] >= MIN_ARTICLE_THRESHOLD
    ticker_volume['warning'] = ticker_volume['article_count'].apply(
        lambda x: f"⚠️ Only {x} articles" if x < MIN_ARTICLE_THRESHOLD else ""
    )

    return filtered_news, filtered_df, ticker_volume

filtered_news, filtered_df, ticker_volume = filter_data_by_timeframe('all')

# ============================================================================
# FUNCTION TO GENERATE ALL FIGURES
# ============================================================================

def generate_figures(filtered_news, filtered_df, ticker_volume, timeframe_label):

    # ===== AGGREGATIONS =====
    daily_sentiment = filtered_df.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
    hourly_sentiment = filtered_df.groupby(['hour', 'sentiment']).size().reset_index(name='count')

    daily_avg = filtered_df.groupby('date_only')['sentiment_score'].mean().reset_index()
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

    filtered_df['topic_category'] = filtered_df['topics'].apply(categorize_topic)
    category_sentiment = filtered_df.groupby(['topic_category', 'sentiment']).size().reset_index(name='count')
    category_avg = filtered_df.groupby('topic_category')['sentiment_score'].agg(['mean', 'count']).reset_index()
    category_avg.columns = ['topic_category', 'avg_sentiment', 'article_count']

    sentiment_dist = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_dist.columns = ['Sentiment', 'Count']
    sentiment_dist['Percentage'] = (sentiment_dist['Count'] / sentiment_dist['Count'].sum() * 100).round(1)

    topic_sentiment = filtered_df.groupby(['topics', 'sentiment']).size().reset_index(name='count')
    top_topics_list = filtered_df['topics'].value_counts().head(15).index.tolist()
    top_topics_df = topic_sentiment[topic_sentiment['topics'].isin(top_topics_list)]

    if 'ticker' in filtered_df.columns:
        ticker_sentiment = filtered_df.groupby(['ticker', 'sentiment']).size().reset_index(name='count')
        ticker_avg = filtered_df.groupby('ticker')['sentiment_score'].agg(['mean', 'count']).reset_index()
        ticker_avg.columns = ['ticker', 'avg_sentiment', 'article_count']
        ticker_avg_filtered = ticker_avg[ticker_avg['article_count'] >= 10].sort_values('avg_sentiment')
    else:
        ticker_avg_filtered = pd.DataFrame()

    sector_sentiment = filtered_df.groupby(['gics_sector', 'sentiment']).size().reset_index(name='count')
    sector_avg = filtered_df.groupby('gics_sector')['sentiment_score'].agg(['mean', 'count']).reset_index()
    sector_avg.columns = ['gics_sector', 'avg_sentiment', 'article_count']
    sector_avg = sector_avg[sector_avg['gics_sector'] != 'Unknown'].sort_values('avg_sentiment')

    source_counts = filtered_news['source'].value_counts().reset_index()
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

    filtered_news['source_type'] = filtered_news['source'].apply(categorize_source)
    source_type_counts = filtered_news['source_type'].value_counts().reset_index()
    source_type_counts.columns = ['source_type', 'count']

    # Note: filtered_news already has 'sentiment' column created at top
    source_sentiment = filtered_news.groupby(['source_type', 'sentiment']).size().reset_index(name='count')

    # ===== INVESTMENT-CRITICAL FIGURES =====

    # 1. PRICE & SENTIMENT OVERLAY
    price_sentiment = corr_summary.sort_values('avg_sentiment').reset_index(drop=True)
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

    # 2. CORRELATION HEATMAP
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
        fig_signals.update_layout(title="<b>Statistically Significant Predictive Signals</b>", height=300, template='plotly_dark')

    # 4. PERFORMANCE vs SENTIMENT SCATTER
    fig_scatter = px.scatter(
        corr_summary,
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
        x=corr_summary['ETF'],
        y=corr_summary['same_day_corr'],
        name='Same Day',
        marker_color='#3b82f6'
    ))
    fig_corr_comp.add_trace(go.Bar(
        x=corr_summary['ETF'],
        y=corr_summary['1d_ahead_corr'],
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
        title=f"<b>Overall Sentiment Composition ({timeframe_label})</b>",
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
        title=f"<b>Daily Sentiment Volume Trend ({timeframe_label})</b>",
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
        title=f"<b>Sentiment Momentum ({timeframe_label})</b>",
        height=400,
        template='plotly_dark',
        hovermode='x unified'
    )

    # 9. NEWS VOLUME OVER TIME
    news_over_time = filtered_news.groupby('date_only').size().reset_index(name='article_count')
    fig_news_volume = px.line(
        news_over_time,
        x='date_only',
        y='article_count',
        title=f"<b>News & Social Media Volume ({timeframe_label})</b>",
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
        title=f"<b>Top 15 News Sources ({timeframe_label})</b>",
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
        title=f"<b>Content Sources by Type ({timeframe_label})</b>",
        hole=0.4,
        height=400
    )
    fig_source_types.update_layout(template='plotly_dark')

    # 12. SENTIMENT BY SOURCE
    fig_source_sentiment = px.bar(
        source_sentiment,
        x='source_type',
        y='count',
        color='sentiment',
        title=f"<b>Sentiment Distribution by Source Type ({timeframe_label})</b>",
        barmode='group',
        height=400,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308',
            'Negative': '#ef4444'
        }
    )
    fig_source_sentiment.update_layout(template='plotly_dark')

    # 13. TOP TOPICS
    fig_top_topics = px.bar(
        top_topics_df,
        y='topics',
        x='count',
        color='sentiment',
        orientation='h',
        title=f"<b>Top 15 Topics by Sentiment ({timeframe_label})</b>",
        barmode='stack',
        height=500,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308', 
            'Negative': '#ef4444'
        }
    )
    fig_top_topics.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')

    # 14. CATEGORY SENTIMENT
    fig_category = px.bar(
        category_avg.sort_values('avg_sentiment'),
        x="avg_sentiment",
        y="topic_category",
        orientation='h',
        title=f"<b>Average Sentiment by Topic Category ({timeframe_label})</b>",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        height=400
    )
    fig_category.update_layout(template='plotly_dark')

    # 15. CATEGORY BREAKDOWN
    fig_breakdown = px.bar(
        category_sentiment,
        x='topic_category',
        y='count',
        color='sentiment',
        title=f"<b>Topic Category Breakdown ({timeframe_label})</b>",
        barmode='stack',
        height=400,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308',
            'Negative': '#ef4444'
        }
    )
    fig_breakdown.update_layout(template='plotly_dark')

    # 16. HOURLY PATTERN
    fig_hourly = px.line(
        hourly_sentiment,
        x="hour",
        y="count",
        color="sentiment",
        title=f"<b>Sentiment Pattern by Hour of Day ({timeframe_label})</b>",
        height=400,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308',
            'Negative': '#ef4444'
        }
    )
    fig_hourly.update_layout(template='plotly_dark')

    # 17. TICKER SENTIMENT
    if not ticker_avg_filtered.empty:
        fig_ticker = px.bar(
            ticker_avg_filtered,
            x="avg_sentiment",
            y="ticker",
            orientation='h',
            title=f"<b>ETF Ticker Sentiment ({timeframe_label})</b>",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            height=500,
            hover_data=['article_count']
        )
        fig_ticker.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
    else:
        fig_ticker = go.Figure()
        fig_ticker.update_layout(title=f"<b>ETF Ticker Sentiment ({timeframe_label})</b>", template='plotly_dark')

    # 18. SECTOR SENTIMENT
    if not sector_avg.empty:
        fig_sector = px.bar(
            sector_avg,
            x="avg_sentiment",
            y="gics_sector",
            orientation='h',
            title=f"<b>GICS Sector Sentiment ({timeframe_label})</b>",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            height=400,
            hover_data=['article_count']
        )
        fig_sector.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
    else:
        fig_sector = go.Figure()
        fig_sector.update_layout(title=f"<b>GICS Sector Sentiment ({timeframe_label})</b>", template='plotly_dark')

    # 19. VOLATILITY
    fig_volatility = go.Figure()
    fig_volatility.add_trace(go.Scatter(
        x=corr_summary['ETF'],
        y=corr_summary['std_daily_return'],
        mode='markers',
        marker=dict(
            size=corr_summary['total_news_reddit'] / 10,
            color=corr_summary['avg_sentiment'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sentiment")
        ),
        text=corr_summary['ETF'],
        hovertemplate='<b>%{text}</b><br>Volatility: %{y:.2f}%<extra></extra>'
    ))
    fig_volatility.update_layout(
        title="<b>ETF Volatility (bubble size = news volume)</b>",
        xaxis_title="ETF",
        yaxis_title="Daily Return Std Dev (%)",
        height=400,
        template='plotly_dark'
    )

    # 20. GAIN vs LOSS
    fig_gain_loss = go.Figure()
    fig_gain_loss.add_trace(go.Bar(
        x=corr_summary['ETF'],
        y=corr_summary['max_gain'],
        name='Max Single-Day Gain',
        marker_color='#22c55e'
    ))
    fig_gain_loss.add_trace(go.Bar(
        x=corr_summary['ETF'],
        y=corr_summary['max_loss'],
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

    insufficient_tickers = ticker_volume[~ticker_volume['sufficient_data']]

    return {
        'fig_price_sentiment': fig_price_sentiment,
        'fig_corr_heatmap': fig_corr_heatmap,
        'fig_signals': fig_signals,
        'fig_scatter': fig_scatter,
        'fig_corr_comp': fig_corr_comp,
        'fig_sentiment_pie': fig_sentiment_pie,
        'fig_daily': fig_daily,
        'fig_momentum': fig_momentum,
        'fig_news_volume': fig_news_volume,
        'fig_sources': fig_sources,
        'fig_source_types': fig_source_types,
        'fig_source_sentiment': fig_source_sentiment,
        'fig_top_topics': fig_top_topics,
        'fig_category': fig_category,
        'fig_breakdown': fig_breakdown,
        'fig_hourly': fig_hourly,
        'fig_ticker': fig_ticker,
        'fig_sector': fig_sector,
        'fig_volatility': fig_volatility,
        'fig_gain_loss': fig_gain_loss,
        'insufficient_tickers': insufficient_tickers,
        'total_articles': len(filtered_news)
    }

initial_figs = generate_figures(filtered_news, filtered_df, ticker_volume, "All Time")

# ============================================================================
# CREATE SENTIMENT SUMMARY TABLE (after sentiment column exists)
# ============================================================================
sentiment_summary = combined_news.groupby('sentiment').agg({
    'LLM_score': ['mean', 'std', 'min', 'max'],
    'title': 'count'
}).reset_index()
sentiment_summary.columns = ['Sentiment', 'Avg Score', 'Std Dev', 'Min Score', 
                              'Max Score', 'Article Count']
sentiment_summary = sentiment_summary.round(3)

etf_performance = corr_summary[['ETF', 'price_change_pct', 'avg_sentiment', 
                                  'total_news_reddit', 'avg_daily_return', 
                                  'same_day_corr', '1d_ahead_corr', '1d_ahead_sig']].copy()
etf_performance = etf_performance.sort_values('price_change_pct', ascending=False)

corr_comparison = corr_summary[['ETF', 'same_day_corr', '1d_ahead_corr', 
                                 'same_day_sig', '1d_ahead_sig']].copy()

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def generate_full_report():
    report = io.StringIO()
    report.write("=" * 80 + "\n")
    report.write("MARKET SENTIMENT ANALYSIS REPORT\n")
    report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write("=" * 80 + "\n\n")
    report.write("KEY FINDINGS\n")
    report.write("-" * 80 + "\n")
    best_corr_etf = corr_summary.loc[corr_summary['1d_ahead_corr'].abs().idxmax()]
    report.write(f"Best 1-Day Predictive Correlation: {best_corr_etf['ETF']} ({best_corr_etf['1d_ahead_corr']:.3f})\n")
    sig_count = len(corr_summary[corr_summary['1d_ahead_sig'] == True])
    report.write(f"Statistically Significant Predictors: {sig_count}/{len(corr_summary)} ETFs\n")
    report.write(f"Average Sentiment Score: {corr_summary['avg_sentiment'].mean():.3f}\n")
    report.write(f"Data Coverage: {combined_news.shape[0]} articles/posts\n\n")
    report.write("ETF PERFORMANCE & CORRELATION SUMMARY\n")
    report.write("-" * 80 + "\n")
    report.write(corr_summary[['ETF', 'price_change_pct', 'avg_sentiment', 'same_day_corr', '1d_ahead_corr', '1d_ahead_sig']].to_string())
    report.write("\n\n")
    return report.getvalue()

# ============================================================================
# DASH APP
# ============================================================================
def get_thematic_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🚀 Market Sentiment Intelligence Dashboard",
                        className="text-center mb-2 mt-4", style={'fontWeight': 'bold'}),
                html.P("Investment-Focused Analysis: Does Sentiment Predict Stock Price Movement?",
                       className="text-center text-muted mb-4", style={'fontSize': '16px'})
            ])
        ]),

        # ===== TIME FILTER =====
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📅 Time Period Filter", className="mb-3"),
                        dbc.RadioItems(
                            id="time-filter",
                            options=[
                                {"label": " All Time (Full Dataset)", "value": "all"},
                                {"label": " Last 1 Month (30 days)", "value": "1m"},
                                {"label": " Last 3 Months (90 days)", "value": "3m"},
                            ],
                            value="all",
                            inline=True,
                            style={'fontSize': '14px'}
                        ),
                        html.Hr(),
                        html.Div(id="filter-info", className="small text-muted")
                    ])
                ], color="#1e1e1e", style={'marginBottom': '20px'})
            ], width=12)
        ]),

        html.Div(id="warning-alert"),

        # ===== EXECUTIVE SUMMARY =====
        html.Hr(),
        dbc.Row([dbc.Col([html.H3("📊 Executive Summary", className="mb-4", style={'fontWeight': 'bold'})])]),

        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Best Predictive Signal", className="text-muted"),
                html.H3("ARKX", style={'color': '#22c55e', 'fontWeight': 'bold'}),
                html.P("Correlation: 0.211 (p=0.03) ✓", className="text-muted small")
            ])], color="#1e1e1e", outline=False)], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Significant Predictors", className="text-muted"),
                html.H3("1 of 6", style={'color': '#f97316', 'fontWeight': 'bold'}),
                html.P("ETFs show statistically significant predictive power", className="text-muted small")
            ])], color="#1e1e1e", outline=False)], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Average Performance", className="text-muted"),
                html.H3("+36.1%", style={'color': '#22c55e', 'fontWeight': 'bold'}),
                html.P("Portfolio average price change", className="text-muted small")
            ])], color="#1e1e1e", outline=False)], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Market Sentiment", className="text-muted"),
                html.H3("+0.083", style={'color': '#eab308', 'fontWeight': 'bold'}),
                html.P("Overall slightly positive bias", className="text-muted small")
            ])], color="#1e1e1e", outline=False)], width=3),
        ], className="mb-4", style={'marginBottom': '30px'}),

        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Data Coverage", className="text-muted"),
                html.H3(id="total-articles-display", style={'color': '#3b82f6', 'fontWeight': 'bold'}),
                html.P("News & Reddit articles analyzed", className="text-muted small")
            ])], color="#1e1e1e", outline=False)], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Sentiment Split", className="text-muted"),
                html.H3("40.6% | 42.3%", style={'color': '#22c55e', 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.P("Positive | Neutral (17.1% Negative)", className="text-muted small")
            ])], color="#1e1e1e", outline=False)], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Avg Daily Coverage", className="text-muted"),
                html.H3("4.7", style={'color': '#3b82f6', 'fontWeight': 'bold'}),
                html.P("Articles per day (236 days avg)", className="text-muted small")
            ])], color="#1e1e1e", outline=False)], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H6("Export Report", className="text-muted"),
                dcc.Download(id="download-report"),
                html.Button("📥 Download CSV", id="btn-download-report",
                            className="btn btn-sm btn-outline-primary w-100", style={'marginTop': '5px'})
            ])], color="#1e1e1e", outline=False)], width=3),
        ], className="mb-4"),

        # ===== CORE INVESTMENT ANALYSIS =====
        html.Hr(),
        dbc.Row([dbc.Col([html.H3("📈 Core Investment Analysis", className="mb-4", style={'fontWeight': 'bold'})])]),

        dbc.Row([
            dbc.Col([dcc.Graph(id='fig-price-sentiment')], width=7),
            dbc.Col([dcc.Graph(id='fig-corr-heatmap')], width=5)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([dcc.Graph(id='fig-scatter')], width=6),
            dbc.Col([dcc.Graph(id='fig-corr-comp')], width=6)
        ], className="mb-4"),

        dbc.Row([dbc.Col([dcc.Graph(id='fig-signals')], width=12)], className="mb-4"),

        # ===== SENTIMENT TRENDS =====
        html.Hr(),
        dbc.Row([dbc.Col([html.H3("📊 Sentiment Trends & Distribution", className="mb-4", style={'fontWeight': 'bold'})])]),

        dbc.Row([
            dbc.Col([dcc.Graph(id='fig-sentiment-pie')], width=4),
            dbc.Col([dcc.Graph(id='fig-momentum')], width=8)
        ], className="mb-4"),

        dbc.Row([dbc.Col([dcc.Graph(id='fig-daily')], width=12)], className="mb-4"),

        # ===== DATA SOURCES =====
        html.Hr(),
        dbc.Row([dbc.Col([html.H3("📡 Data Sources & Quality", className="mb-4", style={'fontWeight': 'bold'})])]),

        dbc.Row([
            dbc.Col([dcc.Graph(id='fig-sources')], width=6),
            dbc.Col([dcc.Graph(id='fig-source-types')], width=6)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([dcc.Graph(id='fig-source-sentiment')], width=6),
            dbc.Col([dcc.Graph(id='fig-news-volume')], width=6)
        ], className="mb-4"),

        # ===== DETAILED ANALYSIS =====
        html.Hr(),
        dbc.Row([dbc.Col([
            html.H3("📋 Detailed Analysis", className="mb-3", style={'fontWeight': 'bold'}),
            dbc.Accordion([
                dbc.AccordionItem([dcc.Graph(id='fig-top-topics')], title="🏷️ Top 15 Topics"),
                dbc.AccordionItem([dbc.Row([
                    dbc.Col([dcc.Graph(id='fig-category')], width=6),
                    dbc.Col([dcc.Graph(id='fig-breakdown')], width=6)
                ])], title="📂 Topic Categories"),
                dbc.AccordionItem([dbc.Row([
                    dbc.Col([dcc.Graph(id='fig-ticker')], width=6),
                    dbc.Col([dcc.Graph(id='fig-sector')], width=6)
                ])], title="🎯 Ticker & Sector Analysis"),
                dbc.AccordionItem([dcc.Graph(id='fig-hourly')], title="⏰ Hourly Sentiment Patterns"),
                dbc.AccordionItem([dbc.Row([
                    dbc.Col([dcc.Graph(id='fig-volatility')], width=6),
                    dbc.Col([dcc.Graph(id='fig-gain-loss')], width=6)
                ])], title="📊 Volatility Analysis"),
            ], always_open=True)
        ])]),

        # ===== WARNINGS =====
        html.Hr(),
        dbc.Row([dbc.Col([
            html.H5("⚠️ Insufficient Data Warning Table", className="mb-3"),
            html.Div(id="warning-table")
        ]), ], className="mb-4"),

        # ===== EXPORT =====
        html.Hr(),
        dbc.Row([dbc.Col([html.H3("📥 Export Data & Reports", className="mb-4", style={'fontWeight': 'bold'})])]),

        dbc.Row([dbc.Col([dbc.Alert([
            html.H5("📊 Generate Full Report", className="mb-3"),
            html.P("Download comprehensive analysis"),
            dcc.Download(id="download-full-report"),
            html.Button("📥 Download Full Report (CSV)", id="btn-download-full-report",
                        className="btn btn-primary")
        ], color="#1e1e1e", style={'padding': '20px'})], width=12)], className="mb-4"),

        dbc.Row([dbc.Col([
            html.H5("ETF Performance & Correlation Summary", className="mb-3"),
            dash_table.DataTable(
                data=etf_performance.to_dict('records'),
                columns=[{"name": i, "id": i} for i in etf_performance.columns],
                style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white', 'fontSize': '12px'},
                style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold', 'border': '1px solid #444', 'textAlign': 'left'},
                style_data={'border': '1px solid #444'},
                style_table={'overflowX': 'auto'},
                style_data_conditional=[
                    {'if': {'column_id': 'price_change_pct', 'filter_query': '{price_change_pct} > 0'}, 'color': '#22c55e', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'price_change_pct', 'filter_query': '{price_change_pct} < 0'}, 'color': '#ef4444', 'fontWeight': 'bold'},
                    {'if': {'column_id': '1d_ahead_sig', 'filter_query': '{1d_ahead_sig} = true'}, 'backgroundColor': '#1e3a1e'}
                ],
                export_format="csv",
                export_headers="display"
            )
        ], width=12)], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H5("Sentiment Score Statistics", className="mb-3"),
                dash_table.DataTable(
                    data=sentiment_summary.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in sentiment_summary.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold', 'border': '1px solid #444'},
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
                    style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold', 'border': '1px solid #444'},
                    style_data={'border': '1px solid #444'},
                    style_data_conditional=[
                        {'if': {'column_id': '1d_ahead_sig', 'filter_query': '{1d_ahead_sig} = true'}, 'backgroundColor': '#1e3a1e', 'fontWeight': 'bold'}
                    ],
                    export_format="csv",
                    export_headers="display"
                )
            ], width=6)
        ], className="mb-4"),

        html.Hr(),

        dbc.Row([dbc.Col([
            html.P(f"Data Threshold: Minimum {MIN_ARTICLE_THRESHOLD} articles | Tiger Fund Management",
                   className="text-center text-muted small mt-3")
        ])])
    ], fluid=True, style={'backgroundColor': '#1a1a1a', 'color': 'white', 'paddingBottom': '50px'})



# ============================================================================
# CALLBACKS
# ============================================================================
def register_thematic_callbacks(app):

    @app.callback(
        [
            Output('fig-price-sentiment', 'figure'),
            Output('fig-corr-heatmap', 'figure'),
            Output('fig-signals', 'figure'),
            Output('fig-scatter', 'figure'),
            Output('fig-corr-comp', 'figure'),
            Output('fig-sentiment-pie', 'figure'),
            Output('fig-daily', 'figure'),
            Output('fig-momentum', 'figure'),
            Output('fig-news-volume', 'figure'),
            Output('fig-sources', 'figure'),
            Output('fig-source-types', 'figure'),
            Output('fig-source-sentiment', 'figure'),
            Output('fig-top-topics', 'figure'),
            Output('fig-category', 'figure'),
            Output('fig-breakdown', 'figure'),
            Output('fig-hourly', 'figure'),
            Output('fig-ticker', 'figure'),
            Output('fig-sector', 'figure'),
            Output('fig-volatility', 'figure'),
            Output('fig-gain-loss', 'figure'),
            Output('filter-info', 'children'),
            Output('warning-alert', 'children'),
            Output('warning-table', 'children'),
            Output('total-articles-display', 'children'),
        ],
        Input('time-filter', 'value')
    )
    def update_dashboard(timeframe):
        filtered_news, filtered_df, ticker_volume = filter_data_by_timeframe(timeframe)

        if timeframe == 'all':
            label = "All Time"
        elif timeframe == '1m':
            label = "Last 1 Month"
        elif timeframe == '3m':
            label = "Last 3 Months"
        else:
            label = "All Time"

        figs = generate_figures(filtered_news, filtered_df, ticker_volume, label)

        filter_info = html.Div([
            html.P(f"Showing: {figs['total_articles']} articles", className="mb-1"),
            html.P(f"Date range: {filtered_news['date'].min().strftime('%Y-%m-%d')} to {filtered_news['date'].max().strftime('%Y-%m-%d')}", className="mb-0")
        ])

        warning_count = len(figs['insufficient_tickers'])
        if warning_count > 0:
            warning_alert = dbc.Alert([
                html.H5(f"⚠️ {warning_count} Ticker(s) with Insufficient Data", className="mb-2"),
                html.P(f"These tickers have fewer than {MIN_ARTICLE_THRESHOLD} articles in the selected time period.")
            ], color="warning", className="mb-4")
        else:
            warning_alert = dbc.Alert([
                html.H5("✅ All Tickers Have Sufficient Data", className="mb-2"),
                html.P(f"All tickers have at least {MIN_ARTICLE_THRESHOLD} articles. Sentiment analysis is reliable.")
            ], color="success", className="mb-4")

        if warning_count > 0:
            warning_table = dash_table.DataTable(
                data=figs['insufficient_tickers'].to_dict('records'),
                columns=[{"name": "Ticker", "id": "ticker"},
                        {"name": "Article Count", "id": "article_count"},
                        {"name": "Status", "id": "warning"}],
                style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white'},
                style_header={'backgroundColor': '#7f1d1d', 'fontWeight': 'bold', 'border': '1px solid #ef4444'},
                style_data={'border': '1px solid #444'},
                style_data_conditional=[{'if': {'column_id': 'article_count'}, 'color': '#ef4444', 'fontWeight': 'bold'}]
            )
        else:
            warning_table = html.P("No warnings - all tickers have sufficient data coverage.", className="text-success")

        return (
            figs['fig_price_sentiment'], figs['fig_corr_heatmap'], figs['fig_signals'],
            figs['fig_scatter'], figs['fig_corr_comp'], figs['fig_sentiment_pie'],
            figs['fig_daily'], figs['fig_momentum'], figs['fig_news_volume'],
            figs['fig_sources'], figs['fig_source_types'], figs['fig_source_sentiment'],
            figs['fig_top_topics'], figs['fig_category'], figs['fig_breakdown'],
            figs['fig_hourly'], figs['fig_ticker'], figs['fig_sector'],
            figs['fig_volatility'], figs['fig_gain_loss'],
            filter_info, warning_alert, warning_table, str(figs['total_articles'])
        )

    @app.callback(Output("download-report", "data"), Input("btn-download-report", "n_clicks"), prevent_initial_call=True)
    def download_report(n_clicks):
        return dict(content=generate_full_report(), filename="market_sentiment_report.txt")

    @app.callback(Output("download-full-report", "data"), Input("btn-download-full-report", "n_clicks"), prevent_initial_call=True)
    def download_full_report_callback(n_clicks):
        return dict(content=generate_full_report(), filename="market_sentiment_full_report.txt")
