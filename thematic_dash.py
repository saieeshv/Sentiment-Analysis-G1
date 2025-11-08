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



df = pd.read_csv('broad_market_topic_modeling_20251102_084257.csv', sep=',')
corr_summary = pd.read_csv('broad_market_correlation_summary_20251102_081024.csv')
corr_detailed = pd.read_csv('broad_market_correlation_detailed_20251102_081024.csv')
combined_news = pd.read_csv('combined_news_reddit_20251101.csv')



df = df[~df['topics'].str.contains('paywall|paylimitwall', case=False, na=False)]
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['hour'] = df['date'].dt.hour
df['date_only'] = pd.to_datetime(df['date'].dt.date)



sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)



combined_news['date'] = pd.to_datetime(combined_news['date'], errors='coerce')
combined_news['date_only'] = pd.to_datetime(combined_news['date'].dt.date)
combined_news['sentiment'] = combined_news['LLM_label']
combined_news['sentiment_score'] = combined_news['sentiment'].map(sentiment_map)



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



# ============================================================================
# CALCULATE EXECUTIVE SUMMARY METRICS
# ============================================================================

MIN_ARTICLE_THRESHOLD = 10

# Overall sentiment (all-time)
if len(combined_news) > 0:
    sentiment_avg = combined_news['sentiment_score'].mean()
    sentiment_pos_pct = (combined_news['sentiment'] == 'Positive').sum() / len(combined_news) * 100
    sentiment_neg_pct = (combined_news['sentiment'] == 'Negative').sum() / len(combined_news) * 100
    sentiment_neutral_pct = (combined_news['sentiment'] == 'Neutral').sum() / len(combined_news) * 100
    total_articles = len(combined_news)
else:
    sentiment_avg = 0
    sentiment_pos_pct = 0
    sentiment_neg_pct = 0
    sentiment_neutral_pct = 0
    total_articles = 0

# Most recurring topic in LAST 7 DAYS (to match Top Topics chart)
cutoff_7d = today - pd.Timedelta(days=7)
last_7d_df = df[df['date'] >= cutoff_7d].copy()

if len(last_7d_df) > 0 and 'topic_theme' in last_7d_df.columns:
    top_topic_7d = last_7d_df['topic_theme'].value_counts().head(1)
    if len(top_topic_7d) > 0:
        top_topic_name = top_topic_7d.index[0]
        top_topic_count = top_topic_7d.values[0]
        # Get sentiment for this topic
        top_topic_sentiment = last_7d_df[last_7d_df['topic_theme'] == top_topic_name]['sentiment_score'].mean()
    else:
        top_topic_name = "N/A"
        top_topic_count = 0
        top_topic_sentiment = 0
else:
    top_topic_name = "N/A"
    top_topic_count = 0
    top_topic_sentiment = 0

# Average correlation strength
avg_corr = corr_summary['same_day_corr'].mean()



# ============================================================================
# HELPER FUNCTION TO FORMAT DATE RANGES
# ============================================================================

def format_date_range(data, date_column='date_only'):
    """
    Formats date range for chart titles
    Returns string like "Oct 15, 2024 - Nov 07, 2025"
    """
    if len(data) == 0 or date_column not in data.columns:
        return "No Data"
    
    dates = data[date_column].dropna()
    if len(dates) == 0:
        return "No Data"
    
    min_date = dates.min()
    max_date = dates.max()
    
    return f"{min_date.strftime('%b %d, %Y')} - {max_date.strftime('%b %d, %Y')}"



# ============================================================================
# FUNCTION TO GENERATE ALL FIGURES
# ============================================================================

def generate_figures(filtered_news, filtered_df):
    
    # ===== CALCULATE DATE RANGE FOR ALL CHARTS =====
    date_range_news = format_date_range(filtered_news, 'date_only')
    date_range_df = format_date_range(filtered_df, 'date_only')
    
    # ===== FILTER TO LAST 7 DAYS FOR TOP TOPICS =====
    cutoff_7d = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
    filtered_df_7d = filtered_df[filtered_df['date'] >= cutoff_7d].copy()
    date_range_7d = format_date_range(filtered_df_7d, 'date_only')
    
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
    
    # USE LAST 7 DAYS DATA FOR TOP TOPICS TO MATCH EXECUTIVE SUMMARY
    if 'topic_theme' in filtered_df_7d.columns:
        topic_sentiment = filtered_df_7d.groupby(['topic_theme', 'sentiment']).size().reset_index(name='count')
        top_topics_list = filtered_df_7d['topic_theme'].value_counts().head(15).index.tolist()
        top_topics_df = topic_sentiment[topic_sentiment['topic_theme'].isin(top_topics_list)]
    else:
        # Fallback to empty dataframe if topic_theme doesn't exist
        top_topics_df = pd.DataFrame(columns=['topic_theme', 'sentiment', 'count'])
    
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
        title=f"ETF Price Performance vs Average Sentiment<br><sub>{date_range_news}</sub>",
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
        title=f"Sentiment-Price Correlation Strength Matrix<br><sub>{date_range_news}</sub>",
        height=400,
        template='plotly_dark'
    )
    
    # 3. PERFORMANCE vs SENTIMENT SCATTER
    fig_scatter = px.scatter(
        corr_summary,
        x='avg_sentiment',
        y='price_change_pct',
        size='total_news_reddit',
        color='ETF',
        title=f"ETF Performance vs Sentiment (bubble size = news volume)<br><sub>{date_range_news}</sub>",
        height=500,
        labels={'price_change_pct': 'Price Change (%)', 'avg_sentiment': 'Average Sentiment'},
        hover_data=['same_day_corr']
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.update_layout(template='plotly_dark')
    
    # 4. CORRELATION COMPARISON
    fig_corr_comp = go.Figure()
    fig_corr_comp.add_trace(go.Bar(
        x=corr_summary['ETF'],
        y=corr_summary['same_day_corr'],
        name='Same Day',
        marker_color='#3b82f6'
    ))
    fig_corr_comp.update_layout(
        title=f"Same Day Sentiment-Price Correlation<br><sub>{date_range_news}</sub>",
        barmode='group',
        height=400,
        template='plotly_dark',
        yaxis_title="Correlation Coefficient",
        hovermode='x unified'
    )
    fig_corr_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # 5. SENTIMENT DISTRIBUTION
    fig_sentiment_pie = px.pie(
        sentiment_dist,
        names='Sentiment',
        values='Count',
        title=f"Overall Sentiment Composition<br><sub>{date_range_df}</sub>",
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
    
    # 6. DAILY SENTIMENT TREND
    daily_sentiment_news = filtered_news.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
    
    all_dates_news = sorted(filtered_news['date_only'].dropna().unique())
    last_7_days = all_dates_news[-7:] if len(all_dates_news) >= 7 else all_dates_news
    
    daily_sentiment_last7 = daily_sentiment_news[daily_sentiment_news['date_only'].isin(last_7_days)]
    
    if len(last_7_days) > 0:
        date_range_str_daily = f"{last_7_days[0].strftime('%b %d')} - {last_7_days[-1].strftime('%b %d, %Y')}"
    else:
        date_range_str_daily = "No Data"
    
    fig_daily = px.bar(
        daily_sentiment_last7,
        x="date_only",
        y="count",
        color="sentiment",
        title=f"Daily Sentiment Volume Trend - Last 7 Days<br><sub>{date_range_str_daily}</sub>",
        barmode='stack',
        height=400,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308',
            'Negative': '#ef4444'
        }
    )
    fig_daily.update_layout(template='plotly_dark', xaxis_title="Date", yaxis_title="Article Count")
    
    # 7. SENTIMENT MOMENTUM
    daily_avg_news = filtered_news.groupby('date_only')['sentiment_score'].mean().reset_index()
    daily_avg_news.columns = ['date', 'avg_sentiment']
    daily_avg_news['sentiment_ma_7'] = daily_avg_news['avg_sentiment'].rolling(window=7, min_periods=1).mean()
    
    all_dates = sorted(daily_avg_news['date'].unique())
    last_7_days_momentum = all_dates[-7:] if len(all_dates) >= 7 else all_dates
    daily_avg_last7 = daily_avg_news[daily_avg_news['date'].isin(last_7_days_momentum)].copy()
    
    daily_volume = filtered_news.groupby('date_only').size().reset_index(name='article_count')
    daily_volume.columns = ['date', 'article_count']
    daily_volume_last7 = daily_volume[daily_volume['date'].isin(last_7_days_momentum)].copy()
    
    fig_momentum = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_momentum.add_trace(go.Scatter(
        x=daily_avg_last7['date'],
        y=daily_avg_last7['avg_sentiment'],
        mode='lines+markers',
        name='Daily Avg',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8)
    ), secondary_y=False)
    
    fig_momentum.add_trace(go.Scatter(
        x=daily_avg_last7['date'],
        y=daily_avg_last7['sentiment_ma_7'],
        mode='lines+markers',
        name='7-Day MA',
        line=dict(color='#f97316', width=3),
        marker=dict(size=8)
    ), secondary_y=False)
    
    fig_momentum.add_trace(go.Bar(
        x=daily_volume_last7['date'],
        y=daily_volume_last7['article_count'],
        name='Daily Volume',
        marker=dict(color='rgba(107, 114, 128, 0.3)'),
        hovertemplate='<b>%{x}</b><br>Articles: %{y}<extra></extra>'
    ), secondary_y=True)
    
    if len(last_7_days_momentum) > 0:
        date_range_str = f"{last_7_days_momentum[0].strftime('%b %d')} - {last_7_days_momentum[-1].strftime('%b %d, %Y')}"
    else:
        date_range_str = "No Data"
    
    fig_momentum.update_layout(
        title=f"Sentiment Momentum - Last 7 Days<br><sub>{date_range_str}</sub>",
        height=400,
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title="Date"
    )
    fig_momentum.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    fig_momentum.update_yaxes(title_text="Article Volume", side='right', showgrid=False, secondary_y=True)
    
    # 8. NEWS VOLUME OVER TIME
    news_over_time = filtered_news.groupby('date_only').size().reset_index(name='article_count')
    fig_news_volume = px.line(
        news_over_time,
        x='date_only',
        y='article_count',
        title=f"News & Social Media Volume<br><sub>{date_range_news}</sub>",
        height=400,
        markers=True
    )
    fig_news_volume.update_traces(line_color='#3b82f6', marker=dict(size=5))
    fig_news_volume.update_layout(template='plotly_dark')
    
    # 9. TOP SOURCES
    fig_sources = px.bar(
        top_sources,
        x='count',
        y='source',
        orientation='h',
        title=f"Top 15 News Sources<br><sub>{date_range_news}</sub>",
        height=500,
        color='count',
        color_continuous_scale='Blues'
    )
    fig_sources.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
    
    # 10. SOURCE TYPE BREAKDOWN
    fig_source_types = px.pie(
        source_type_counts,
        names='source_type',
        values='count',
        title=f"Content Sources by Type<br><sub>{date_range_news}</sub>",
        hole=0.4,
        height=400
    )
    fig_source_types.update_layout(template='plotly_dark')
    
    # 11. SENTIMENT BY SOURCE
    fig_source_sentiment = px.bar(
        source_sentiment,
        x='source_type',
        y='count',
        color='sentiment',
        title=f"Sentiment Distribution by Source Type<br><sub>{date_range_news}</sub>",
        barmode='group',
        height=400,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308',
            'Negative': '#ef4444'
        }
    )
    fig_source_sentiment.update_layout(template='plotly_dark')
    
    # 12. TOP TOPICS (USING LAST 7 DAYS DATA)
    if len(top_topics_df) > 0:
        fig_top_topics = px.bar(
            top_topics_df,
            y='topic_theme',
            x='count',
            color='sentiment',
            orientation='h',
            title=f"Top Topics by Sentiment - Last 7 Days<br><sub>{date_range_7d}</sub>",
            barmode='stack',
            height=500,
            color_discrete_map={
                'Positive': '#22c55e',
                'Neutral': '#eab308',
                'Negative': '#ef4444'
            }
        )
        fig_top_topics.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
    else:
        fig_top_topics = go.Figure()
        fig_top_topics.update_layout(
            title=f"Top Topics by Sentiment - Last 7 Days<br><sub>{date_range_7d}</sub>",
            template='plotly_dark'
        )
    
    # 13. CATEGORY BREAKDOWN
    fig_breakdown = px.bar(
        category_sentiment,
        x='topic_category',
        y='count',
        color='sentiment',
        title=f"Topic Category Breakdown<br><sub>{date_range_df}</sub>",
        barmode='stack',
        height=400,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308',
            'Negative': '#ef4444'
        }
    )
    fig_breakdown.update_layout(template='plotly_dark')
    
    # 14. HOURLY PATTERN
    fig_hourly = px.line(
        hourly_sentiment,
        x="hour",
        y="count",
        color="sentiment",
        title=f"Sentiment Pattern by Hour of Day<br><sub>{date_range_df}</sub>",
        height=400,
        color_discrete_map={
            'Positive': '#22c55e',
            'Neutral': '#eab308',
            'Negative': '#ef4444'
        }
    )
    fig_hourly.update_layout(template='plotly_dark')
    
    # 15. TICKER SENTIMENT
    if not ticker_avg_filtered.empty:
        fig_ticker = px.bar(
            ticker_avg_filtered,
            x="avg_sentiment",
            y="ticker",
            orientation='h',
            title=f"ETF Ticker Sentiment<br><sub>{date_range_df}</sub>",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            height=500,
            hover_data=['article_count']
        )
        fig_ticker.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
    else:
        fig_ticker = go.Figure()
        fig_ticker.update_layout(title=f"ETF Ticker Sentiment<br><sub>{date_range_df}</sub>", template='plotly_dark')
    
    # 16. SECTOR SENTIMENT
    if not sector_avg.empty:
        fig_sector = px.bar(
            sector_avg,
            x="avg_sentiment",
            y="gics_sector",
            orientation='h',
            title=f"GICS Sector Sentiment<br><sub>{date_range_df}</sub>",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            height=400,
            hover_data=['article_count']
        )
        fig_sector.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
    else:
        fig_sector = go.Figure()
        fig_sector.update_layout(title=f"GICS Sector Sentiment<br><sub>{date_range_df}</sub>", template='plotly_dark')
    
    # 17. GAIN vs LOSS
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
        title=f"ETF Max Single-Day Movements<br><sub>{date_range_news}</sub>",
        barmode='group',
        height=400,
        template='plotly_dark',
        yaxis_title="Percentage (%)"
    )
    
    # Calculate data quality warnings
    ticker_volume = filtered_news.groupby('ticker').size().reset_index(name='article_count')
    ticker_volume['sufficient_data'] = ticker_volume['article_count'] >= MIN_ARTICLE_THRESHOLD
    insufficient_tickers = ticker_volume[~ticker_volume['sufficient_data']]
    
    return {
        'fig_price_sentiment': fig_price_sentiment,
        'fig_corr_heatmap': fig_corr_heatmap,
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
        'fig_breakdown': fig_breakdown,
        'fig_hourly': fig_hourly,
        'fig_ticker': fig_ticker,
        'fig_sector': fig_sector,
        'fig_gain_loss': fig_gain_loss,
        'insufficient_tickers': insufficient_tickers,
        'total_articles': len(filtered_news),
        'date_range_display': date_range_news
    }



initial_figs = generate_figures(combined_news, df)



# ============================================================================
# CREATE SENTIMENT SUMMARY TABLE
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
                                'same_day_corr']].copy()
etf_performance = etf_performance.sort_values('price_change_pct', ascending=False)



corr_comparison = corr_summary[['ETF', 'same_day_corr', 'same_day_sig']].copy()



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
    report.write(f"Average Correlation Strength: {avg_corr:.3f}\n")
    report.write(f"Overall Sentiment: {sentiment_avg:.3f}\n")
    report.write(f"Top Topic (7d): {top_topic_name} (Sentiment: {top_topic_sentiment:.3f})\n")
    report.write(f"Data Coverage: {total_articles} articles/posts\n\n")
    
    report.write("ETF PERFORMANCE & CORRELATION SUMMARY\n")
    report.write("-" * 80 + "\n")
    report.write(corr_summary[['ETF', 'price_change_pct', 'avg_sentiment', 'same_day_corr', 'same_day_sig']].to_string())
    report.write("\n\n")
    
    return report.getvalue()



# ============================================================================
# DASH APP
# ============================================================================



app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])



app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚀 Market Sentiment Intelligence Dashboard",
                    className="text-center mb-2 mt-4", style={'fontWeight': 'bold'}),
            html.P("Investment-Focused Analysis: Sentiment and Market Performance",
                   className="text-center text-muted mb-4", style={'fontSize': '16px'})
        ])
    ]),
    
    # ===== EXECUTIVE SUMMARY (UNIFORM CARD HEIGHTS) =====
    html.Hr(),
    dbc.Row([dbc.Col([html.H3("📊 Executive Summary", className="mb-4", style={'fontWeight': 'bold'})])]),
    
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([
            html.H6("Overall Sentiment", className="text-muted", style={'marginBottom': '12px'}),
            html.H3(f"{sentiment_avg:+.3f}", style={
                'color': '#22c55e' if sentiment_avg > 0.1 else '#ef4444' if sentiment_avg < -0.1 else '#eab308',
                'fontWeight': 'bold',
                'marginBottom': '12px'
            }),
            html.P(f"{sentiment_pos_pct:.1f}% Pos | {sentiment_neg_pct:.1f}% Neg", 
                   className="text-muted small",
                   style={'marginBottom': '0'})
        ])], color="#1e1e1e", outline=False, style={'height': '180px'})], width=3),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.H6("Top Trending Topic (7d)", className="text-muted", style={'marginBottom': '12px'}),
            html.P(f"{top_topic_name}", 
                    style={'color': '#3b82f6', 'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '12px', 'minHeight': '40px'}),
            html.P(f"Sentiment: {top_topic_sentiment:+.3f} | {top_topic_count} mentions", 
                   className="small",
                   style={'color': '#22c55e' if top_topic_sentiment > 0 else '#ef4444' if top_topic_sentiment < 0 else '#eab308',
                          'marginBottom': '0'})
        ])], color="#1e1e1e", outline=False, style={'height': '180px'})], width=3),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.H6("Avg Correlation Strength", className="text-muted", style={'marginBottom': '12px'}),
            html.H3(f"{avg_corr:.3f}", style={
                'color': '#22c55e' if abs(avg_corr) > 0.3 else '#eab308' if abs(avg_corr) > 0.15 else '#ef4444',
                'fontWeight': 'bold',
                'marginBottom': '12px'
            }),
            html.P("Portfolio-wide sentiment signal", className="text-muted small", style={'marginBottom': '0'})
        ])], color="#1e1e1e", outline=False, style={'height': '180px'})], width=3),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.H6("Data Coverage", className="text-muted", style={'marginBottom': '12px'}),
            html.H3(f"{total_articles:,}", style={'color': '#3b82f6', 'fontWeight': 'bold', 'marginBottom': '12px'}),
            html.P(f"Total articles analyzed", className="text-muted small", style={'marginBottom': '0'})
        ])], color="#1e1e1e", outline=False, style={'height': '180px'})], width=3),
        
    ], className="mb-4", style={'marginBottom': '30px'}),
    
    # ===== DATA COVERAGE INFO =====
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Strong("📅 Full Dataset Coverage: ", style={'fontSize': '14px', 'color': '#ffffff'}),
                        html.Span(f"{initial_figs['date_range_display']}", style={'color': '#000000', 'fontWeight': 'bold', 'backgroundColor': '#ffffff', 'padding': '2px 8px', 'borderRadius': '4px'})
                    ])
                ])
            ], color="#1e1e1e", className="mb-4")
        ])
    ]),
    
    html.Div(id="warning-alert"),
    
    # ===== CORE INVESTMENT ANALYSIS =====
    html.Hr(),
    dbc.Row([dbc.Col([html.H3("📈 Core Investment Analysis", className="mb-4", style={'fontWeight': 'bold'})])]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='fig-price-sentiment', figure=initial_figs['fig_price_sentiment'])], width=7),
        dbc.Col([dcc.Graph(id='fig-corr-heatmap', figure=initial_figs['fig_corr_heatmap'])], width=5)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='fig-scatter', figure=initial_figs['fig_scatter'])], width=6),
        dbc.Col([dcc.Graph(id='fig-corr-comp', figure=initial_figs['fig_corr_comp'])], width=6)
    ], className="mb-4"),
    
    # ===== SENTIMENT TRENDS =====
    html.Hr(),
    dbc.Row([dbc.Col([html.H3("📊 Sentiment Trends & Distribution", className="mb-4", style={'fontWeight': 'bold'})])]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='fig-sentiment-pie', figure=initial_figs['fig_sentiment_pie'])], width=4),
        dbc.Col([dcc.Graph(id='fig-momentum', figure=initial_figs['fig_momentum'])], width=8)
    ], className="mb-4"),
    
    dbc.Row([dbc.Col([dcc.Graph(id='fig-daily', figure=initial_figs['fig_daily'])], width=12)], className="mb-4"),
    
    # ===== DETAILED ANALYSIS =====
    html.Hr(),
    dbc.Row([dbc.Col([
        html.H3("📋 Detailed Analysis", className="mb-3", style={'fontWeight': 'bold'}),
        dbc.Accordion([
            dbc.AccordionItem([dcc.Graph(id='fig-top-topics', figure=initial_figs['fig_top_topics'])], title="🏷️ Top Topics (Last 7 Days)"),
            dbc.AccordionItem([dcc.Graph(id='fig-breakdown', figure=initial_figs['fig_breakdown'])], title="📂 Topic Categories"),
            dbc.AccordionItem([dbc.Row([
                dbc.Col([dcc.Graph(id='fig-ticker', figure=initial_figs['fig_ticker'])], width=6),
                dbc.Col([dcc.Graph(id='fig-sector', figure=initial_figs['fig_sector'])], width=6)
            ])], title="🎯 Ticker & Sector Analysis"),
            dbc.AccordionItem([dcc.Graph(id='fig-hourly', figure=initial_figs['fig_hourly'])], title="⏰ Hourly Sentiment Patterns"),
            dbc.AccordionItem([dcc.Graph(id='fig-gain-loss', figure=initial_figs['fig_gain_loss'])], title="📊 Max Daily Movements"),
        ], always_open=True)
    ])]),
    
    # ===== DATA SOURCES =====
    html.Hr(),
    dbc.Row([dbc.Col([html.H3("📡 Data Sources & Quality", className="mb-4", style={'fontWeight': 'bold'})])]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='fig-sources', figure=initial_figs['fig_sources'])], width=6),
        dbc.Col([dcc.Graph(id='fig-source-types', figure=initial_figs['fig_source_types'])], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='fig-source-sentiment', figure=initial_figs['fig_source_sentiment'])], width=6),
        dbc.Col([dcc.Graph(id='fig-news-volume', figure=initial_figs['fig_news_volume'])], width=6)
    ], className="mb-4"),
    
    # ===== WARNINGS =====
    html.Hr(),
    dbc.Row([dbc.Col([
        html.H5("⚠️ Data Quality Warnings", className="mb-3"),
        html.Div(id="warning-table")
    ]), ], className="mb-4"),
    
    # ===== EXPORT =====
    html.Hr(),
    dbc.Row([dbc.Col([html.H3("📥 Export Data & Reports", className="mb-4", style={'fontWeight': 'bold'})])]),
    
    dbc.Row([dbc.Col([dbc.Alert([
        html.H5("📊 Generate Full Report", className="mb-3"),
        html.P("Download comprehensive analysis"),
        dcc.Download(id="download-full-report"),
        html.Button("📥 Download Full Report (TXT)", id="btn-download-full-report",
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
# CALLBACK FOR DATA QUALITY WARNINGS
# ============================================================================

# @callback(
#     [Output('warning-alert', 'children'),
#      Output('warning-table', 'children')],
#     Input('warning-alert', 'id')  # Dummy input to trigger on load
# )
# def update_warnings(_):
#     ticker_volume = combined_news.groupby('ticker').size().reset_index(name='article_count')
#     ticker_volume['sufficient_data'] = ticker_volume['article_count'] >= MIN_ARTICLE_THRESHOLD
#     ticker_volume['warning'] = ticker_volume['article_count'].apply(
#         lambda x: f"⚠️ Only {x} articles" if x < MIN_ARTICLE_THRESHOLD else ""
#     )
#     insufficient_tickers = ticker_volume[~ticker_volume['sufficient_data']]
    
#     warning_count = len(insufficient_tickers)
#     if warning_count > 0:
#         warning_alert = dbc.Alert([
#             html.H5(f"⚠️ {warning_count} Ticker(s) with Insufficient Data", className="mb-2"),
#             html.P(f"These tickers have fewer than {MIN_ARTICLE_THRESHOLD} articles in the dataset.")
#         ], color="warning", className="mb-4")
#     else:
#         warning_alert = dbc.Alert([
#             html.H5("✅ All Tickers Have Sufficient Data", className="mb-2"),
#             html.P(f"All tickers have at least {MIN_ARTICLE_THRESHOLD} articles. Sentiment analysis is reliable.")
#         ], color="success", className="mb-4")
    
#     if warning_count > 0:
#         warning_table = dash_table.DataTable(
#             data=insufficient_tickers.to_dict('records'),
#             columns=[{"name": "Ticker", "id": "ticker"},
#                     {"name": "Article Count", "id": "article_count"},
#                     {"name": "Status", "id": "warning"}],
#             style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white'},
#             style_header={'backgroundColor': '#7f1d1d', 'fontWeight': 'bold', 'border': '1px solid #ef4444'},
#             style_data={'border': '1px solid #444'},
#             style_data_conditional=[{'if': {'column_id': 'article_count'}, 'color': '#ef4444', 'fontWeight': 'bold'}]
#         )
#     else:
#         warning_table = html.P("No warnings - all tickers have sufficient data coverage.", className="text-success")
    
#     return warning_alert, warning_table



# @callback(Output("download-full-report", "data"), Input("btn-download-full-report", "n_clicks"), prevent_initial_call=True)
# def download_full_report_callback(n_clicks):
#     return dict(content=generate_full_report(), filename="market_sentiment_full_report.txt")

# ============================================================================
# EXPORT FUNCTIONS FOR INTEGRATION WITH MULTI-TAB APP
# ============================================================================

# ============================================================================
# EXPORT FUNCTIONS FOR INTEGRATION WITH MULTI-TAB APP
# ============================================================================

def get_thematic_layout():
    """Returns the layout for integration with multi-tab applications"""
    return app.layout

def register_thematic_callbacks(external_app):
    """Register callbacks with an external app instance for multi-tab integration"""
    
    @external_app.callback(
        [Output('warning-alert', 'children'),
         Output('warning-table', 'children')],
        Input('warning-alert', 'id')
    )
    def update_warnings(_):
        ticker_volume = combined_news.groupby('ticker').size().reset_index(name='article_count')
        ticker_volume['sufficient_data'] = ticker_volume['article_count'] >= MIN_ARTICLE_THRESHOLD
        ticker_volume['warning'] = ticker_volume['article_count'].apply(
            lambda x: f"⚠️ Only {x} articles" if x < MIN_ARTICLE_THRESHOLD else ""
        )
        
        insufficient_tickers = ticker_volume[~ticker_volume['sufficient_data']]
        warning_count = len(insufficient_tickers)
        
        if warning_count > 0:
            warning_alert = dbc.Alert([
                html.H5(f"⚠️ {warning_count} Ticker(s) with Insufficient Data", className="mb-2"),
                html.P(f"These tickers have fewer than {MIN_ARTICLE_THRESHOLD} articles in the dataset.")
            ], color="warning", className="mb-4")
        else:
            warning_alert = dbc.Alert([
                html.H5("✅ All Tickers Have Sufficient Data", className="mb-2"),
                html.P(f"All tickers have at least {MIN_ARTICLE_THRESHOLD} articles. Sentiment analysis is reliable.")
            ], color="success", className="mb-4")
        
        if warning_count > 0:
            warning_table = dash_table.DataTable(
                data=insufficient_tickers.to_dict('records'),
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
        
        return warning_alert, warning_table
    
    @external_app.callback(
        Output("download-full-report", "data"),
        Input("btn-download-full-report", "n_clicks"),
        prevent_initial_call=True
    )
    def download_full_report_callback(n_clicks):
        return dict(content=generate_full_report(), filename="market_sentiment_full_report.txt")


if __name__ == '__main__':
    # When running standalone, register callbacks with the local app instance
    register_thematic_callbacks(app)
    app.run(debug=True, port=8050)

