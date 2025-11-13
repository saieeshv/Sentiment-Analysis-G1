import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, Input, Output
import re

# Load Data
# summary_df = pd.read_csv("security_correlation_summary_20251031_234833.csv")
# detailed_df = pd.read_csv("security_correlation_detailed_20251031_234833.csv")
# trending_df = pd.read_csv("trending_stocks_20251101_140418.csv")

# Load Data
summary_df = pd.read_csv("/content/drive/MyDrive/IS484_FYP/Google_Colab/Outputs/security_correlation_summary.csv")
detailed_df = pd.read_csv("/content/drive/MyDrive/IS484_FYP/Google_Colab/Outputs/security_correlation_detailed.csv")
trending_df = pd.read_csv("/content/drive/MyDrive/IS484_FYP/Google_Colab/Raw_Data/trending_stocks.csv")

# Convert date columns to datetime
summary_df['date_range_start'] = pd.to_datetime(summary_df['date_range_start'])
summary_df['date_range_end'] = pd.to_datetime(summary_df['date_range_end'])

# Prepare Sentiment Score Statistics
sentiment_stats_df = detailed_df.groupby('ticker').agg(
    Avg_Score=('avg_sentiment', 'mean'),
    Min_Score=('avg_sentiment', 'min'),
    Max_Score=('avg_sentiment', 'max'),
    Article_Count=('valid_data_points', 'sum')
).reset_index()

# Correlation Analysis Details - REMOVED unwanted columns
corr_cols = ['ticker', 'same_day_corr', 'same_day_significant']
corr_stats_df = summary_df[corr_cols]

perf_summary_df = summary_df[
    ['ticker', 'price_change_pct', 'avg_sentiment', 'data_points', 'avg_daily_return',
     'same_day_corr', '1_day_ahead_corr', '1_day_ahead_significant']
]

# ============================================================================
# HELPER FUNCTIONS TO FORMAT DATE RANGES
# ============================================================================
def format_date_range_from_summary(summary_df):
    """
    Formats date range for chart titles using min/max from date_range_start and date_range_end
    Returns string like "Sep 09, 2025 - Oct 30, 2025"
    """
    if len(summary_df) == 0:
        return "No Data"
    
    min_date = summary_df['date_range_start'].min()
    max_date = summary_df['date_range_end'].max()
    
    return f"{min_date.strftime('%b %d, %Y')} - {max_date.strftime('%b %d, %Y')}"


def format_trending_date_range(trending_df):
    """
    Formats the date range for trending stocks chart
    Returns formatted string like "Last 7 Days"
    """
    if len(trending_df) == 0 or 'date_range' not in trending_df.columns:
        return "No Data"
    
    date_range_value = trending_df['date_range'].iloc[0]
    
    import re
    match = re.search(r'last(\d+)days', date_range_value.lower())
    if match:
        num_days = match.group(1)
        return f"Last {num_days} Days"
    else:
        return date_range_value.replace('_', ' ').title()



def get_security_layout():
    return dbc.Container([
        # HEADER SECTION WITH ICON
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="bi bi-graph-up-arrow me-3",
                           style={"fontSize": "48px", "color": "#22c55e"}),
                    html.Span("Securities Sentiment & Performance Dashboard",
                              style={"fontSize": "36px", "fontWeight": "bold", "verticalAlign": "middle"})
                ], style={
                    "backgroundColor": "#2a2a2a",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "marginTop": "20px",
                    "marginBottom": "30px",
                    "border": "2px solid #22c55e",
                    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center"
                })
            ], width=12)
        ]),
        
        dcc.Interval(id='security-interval', interval=60000, n_intervals=0),
        
        # Summary Section (Always visible)
        dbc.Row([dbc.Col(id="security-summary-section", width=12)], className="mb-4"),
        
        # SECTION 1: Trending Stocks (Always visible)
        dbc.Row([
            dbc.Col(html.H4("🔥 Trending Stocks: Buzz vs Sentiment",
                            className="mb-3", style={"fontSize": "28px", "fontWeight": "600"}), width=12)
        ]),
        dbc.Row([dbc.Col(dcc.Graph(id="trending-stocks-bubble"), width=12)], className="mb-5"),
        
        # SECTION 2: Sentiment vs Price Performance (Always visible)
        dbc.Row([
            dbc.Col(html.H4("📊 Sentiment vs Price Performance",
                            className="mb-3", style={"fontSize": "28px", "fontWeight": "600"}), width=12)
        ]),
        dbc.Row([dbc.Col(dcc.Graph(id="security-bubble-sentiment-price"), width=12)], className="mb-5"),
        
        # SECTION 3: Risk vs Reward (Always visible)
        dbc.Row([
            dbc.Col(html.H4("⚖️ Risk vs Reward Analysis",
                            className="mb-3", style={"fontSize": "28px", "fontWeight": "600"}), width=12)
        ]),
        dbc.Row([dbc.Col(dcc.Graph(id="security-scatter-vol-return"), width=12)], className="mb-5"),
        
        # COLLAPSIBLE SECTION: Correlation Analysis
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    html.Div([
                        html.Span("⏳ Correlation Analysis Over Time Horizons",
                                  style={"flex": "1", "textAlign": "left"}),
                        html.I(id="correlation-arrow", className="bi bi-chevron-down",
                               style={"fontSize": "20px"})
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
                    id="collapse-correlation-button",
                    color="dark",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "500",
                        "width": "100%",
                        "backgroundColor": "#2a2a2a",
                        "border": "1px solid #444",
                        "borderRadius": "8px",
                        "padding": "12px 15px",
                        "marginBottom": "0px"
                    }
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id="security-grouped-bar-corr")
                        ], style={"backgroundColor": "#1a1a1a", "border": "none", "padding": "10px 0px"}),
                        style={"backgroundColor": "#1a1a1a", "border": "none"}
                    ),
                    id="collapse-correlation",
                    is_open=False,
                ),
            ], width=12)
        ], className="mb-3"),
        
        # COLLAPSIBLE SECTION: News Coverage
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    html.Div([
                        html.Span("📰 News & Social Media Coverage by Security",
                                  style={"flex": "1", "textAlign": "left"}),
                        html.I(id="news-arrow", className="bi bi-chevron-down",
                               style={"fontSize": "20px"})
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
                    id="collapse-news-button",
                    color="dark",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "500",
                        "width": "100%",
                        "backgroundColor": "#2a2a2a",
                        "border": "1px solid #444",
                        "borderRadius": "8px",
                        "padding": "12px 15px",
                        "marginBottom": "0px"
                    }
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id="security-bar-news-coverage")
                        ], style={"backgroundColor": "#1a1a1a", "border": "none", "padding": "10px 0px"}),
                        style={"backgroundColor": "#1a1a1a", "border": "none"}
                    ),
                    id="collapse-news",
                    is_open=False,
                ),
            ], width=12)
        ], className="mb-3"),
        
        # SECTION: Data Tables
        dbc.Row([
            dbc.Col(html.H4("📋 Security Performance & Correlation Summary",
                            style={"fontSize": "28px", "fontWeight": "600", "marginTop": "20px"}), width=12)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='table-summary',
                    data=perf_summary_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in perf_summary_df.columns],
                    style_cell={'textAlign': 'left', 'padding': '12px', 'backgroundColor': '#303030',
                                'color': 'white', 'fontSize': '15px'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold',
                                  'border': '1px solid #444', 'textAlign': 'left', 'fontSize': '16px'},
                    style_data={'border': '1px solid #444'},
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=[
                        {'if': {'column_id': 'price_change_pct', 'filter_query': '{price_change_pct} > 0'},
                         'color': '#22c55e', 'fontWeight': 'bold'},
                        {'if': {'column_id': 'price_change_pct', 'filter_query': '{price_change_pct} < 0'},
                         'color': '#ef4444', 'fontWeight': 'bold'},
                        {'if': {'column_id': '1_day_ahead_significant', 'filter_query': '{1_day_ahead_significant} = true'},
                         'backgroundColor': '#1e3a1e'}
                    ],
                    export_format='csv',
                    export_headers='display',
                    page_size=10,
                )
            ], width=12, className="mb-5")
        ]),
        
        dbc.Row([
            dbc.Col(html.H4("📈 Sentiment Score Distribution & Statistics",
                            style={"fontSize": "28px", "fontWeight": "600"}), width=12)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='table-sentiment',
                    data=sentiment_stats_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in sentiment_stats_df.columns],
                    style_cell={'textAlign': 'left', 'padding': '12px', 'backgroundColor': '#303030',
                                'color': 'white', 'fontSize': '15px'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold',
                                  'border': '1px solid #444', 'textAlign': 'left', 'fontSize': '16px'},
                    style_data={'border': '1px solid #444'},
                    style_table={'overflowX': 'auto'},
                    export_format='csv',
                    export_headers='display',
                    page_size=10,
                )
            ], width=12, className="mb-5")
        ]),
        
        dbc.Row([
            dbc.Col(html.H4("🔗 Correlation Analysis Details",
                            style={"fontSize": "28px", "fontWeight": "600"}), width=12)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='table-correlation',
                    data=corr_stats_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in corr_stats_df.columns],
                    style_cell={'textAlign': 'left', 'padding': '12px', 'backgroundColor': '#303030',
                                'color': 'white', 'fontSize': '15px'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold',
                                  'border': '1px solid #444', 'textAlign': 'left', 'fontSize': '16px'},
                    style_data={'border': '1px solid #444'},
                    style_table={'overflowX': 'auto'},
                    export_format='csv',
                    export_headers='display',
                    page_size=10,
                )
            ], width=12, className="mb-4")
        ]),
        
    ], fluid=True, style={"backgroundColor": "#1a1a1a", "color": "white", "paddingBottom": "30px"})


def register_security_callbacks(app):
    # Collapse toggle callbacks WITH ARROW ANIMATION
    @app.callback(
        [Output("collapse-correlation", "is_open"),
         Output("correlation-arrow", "className")],
        Input("collapse-correlation-button", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_correlation(n):
        if n is None:
            return False, "bi bi-chevron-down"
        is_open = n % 2 == 1
        arrow = "bi bi-chevron-up" if is_open else "bi bi-chevron-down"
        return is_open, arrow
    
    @app.callback(
        [Output("collapse-news", "is_open"),
         Output("news-arrow", "className")],
        Input("collapse-news-button", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_news(n):
        if n is None:
            return False, "bi bi-chevron-down"
        is_open = n % 2 == 1
        arrow = "bi bi-chevron-up" if is_open else "bi bi-chevron-down"
        return is_open, arrow
    
    # Main data update callback
    @app.callback(
        [
            Output("security-summary-section", "children"),
            Output("trending-stocks-bubble", "figure"),
            Output("security-bubble-sentiment-price", "figure"),
            Output("security-scatter-vol-return", "figure"),
            Output("security-grouped-bar-corr", "figure"),
            Output("security-bar-news-coverage", "figure"),
        ],
        Input("security-interval", "n_intervals")
    )
    def update_all(n):
        # Calculate date ranges for charts
        date_range_str = format_date_range_from_summary(summary_df)
        trending_date_range = format_trending_date_range(trending_df)
        
        # Summary card with enhanced styling
        total_news = summary_df["total_news_reddit"].sum()
        avg_sentiment = summary_df["avg_sentiment"].mean()
        price_change = summary_df["price_change_pct"].mean()
        num_securities = len(summary_df)
        
        # Color coding for sentiment and price change
        sentiment_color = "#22c55e" if avg_sentiment > 0 else "#ef4444" if avg_sentiment < 0 else "#fbbf24"
        price_color = "#22c55e" if price_change > 0 else "#ef4444"
        
        summary_card = dbc.Card([
            dbc.CardHeader(html.H4("📊 Overall Summary", style={"fontSize": "26px", "fontWeight": "bold", "margin": "0"})),
            dbc.CardBody([
                html.P([
                    html.Span("Total News/Reddit Volume: ", style={"fontSize": "18px"}),
                    html.Span(f"{total_news:,}", style={"fontSize": "18px", "fontWeight": "bold", "color": "#60a5fa"})
                ], style={"marginBottom": "12px"}),
                html.P([
                    html.Span("Average Sentiment: ", style={"fontSize": "18px"}),
                    html.Span(f"{avg_sentiment:.3f}", style={"fontSize": "18px", "fontWeight": "bold", "color": sentiment_color})
                ], style={"marginBottom": "12px"}),
                html.P([
                    html.Span("Average Price Change: ", style={"fontSize": "18px"}),
                    html.Span(f"{price_change:+.2f}%", style={"fontSize": "18px", "fontWeight": "bold", "color": price_color})
                ], style={"marginBottom": "12px"}),
                html.P([
                    html.Span("Number of Securities: ", style={"fontSize": "18px"}),
                    html.Span(f"{num_securities}", style={"fontSize": "18px", "fontWeight": "bold", "color": "#60a5fa"})
                ], style={"marginBottom": "0px"}),
            ], style={"color": "white", "padding": "20px"})
        ], style={"backgroundColor": "#212121", "marginBottom": "1rem", "border": "1px solid #444"})
        
        # Trending Stocks Bubble Chart - UPDATED WITH DYNAMIC DATE RANGE
        trending_bubble_fig = px.scatter(
            trending_df,
            x="sentiment_score",
            y="total_mentions",
            size="total_mentions",
            color="category",
            hover_name="ticker",
            hover_data={
                "name": True,
                "sentiment_score": ":.3f",
                "total_mentions": True,
                "positive_mentions": True,
                "negative_mentions": True,
                "category": True
            },
            title=f"Trending Stocks: Social Buzz vs Sentiment<br><sub>{trending_date_range}</sub>",
            labels={
                "sentiment_score": "Sentiment Score (Bullish →)",
                "total_mentions": "Total Mentions (Buzz)"
            },
            template="plotly_dark",
            height=500
        )
        trending_bubble_fig.update_layout(
            title_font_size=22,
            font=dict(size=14),
            xaxis=dict(title_font_size=16, tickfont_size=14),
            yaxis=dict(title_font_size=16, tickfont_size=14),
            legend=dict(font_size=14)
        )
        trending_bubble_fig.add_hline(y=trending_df['total_mentions'].median(),
                                       line_dash="dash", line_color="gray", opacity=0.5)
        trending_bubble_fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        trending_bubble_fig.add_annotation(
            x=0.7, y=trending_df['total_mentions'].max() * 0.95,
            text="🚀 Hot & Bullish", showarrow=False,
            font=dict(size=15, color="lightgreen")
        )
        trending_bubble_fig.add_annotation(
            x=-0.2, y=trending_df['total_mentions'].max() * 0.95,
            text="⚠️ High Buzz but Bearish", showarrow=False,
            font=dict(size=15, color="orange")
        )
        
        # Sentiment vs Price Performance - UPDATED WITH DATE RANGE
        bubble_fig = px.scatter(
            detailed_df,
            x="avg_sentiment",
            y="price_change_pct",
            size="valid_data_points",
            color="ticker",
            title=f"Sentiment vs Price Change by Security<br><sub>{date_range_str}</sub>",
            labels={
                "avg_sentiment": "Average Sentiment Score",
                "price_change_pct": "Price Change %"
            },
            template="plotly_dark",
            height=500
        )
        bubble_fig.update_layout(
            title_font_size=22, font=dict(size=14),
            xaxis=dict(title_font_size=16, tickfont_size=14),
            yaxis=dict(title_font_size=16, tickfont_size=14),
            legend=dict(font_size=14)
        )
        
        # Risk vs Reward - UPDATED WITH DATE RANGE
        vol_return_fig = px.scatter(
            summary_df,
            x="return_volatility",
            y="avg_daily_return",
            color="ticker",
            title=f"Volatility vs Average Daily Return<br><sub>{date_range_str}</sub>",
            labels={
                "return_volatility": "Return Volatility (Risk)",
                "avg_daily_return": "Average Daily Return (%)"
            },
            template="plotly_dark",
            height=500
        )
        vol_return_fig.update_layout(
            title_font_size=22, font=dict(size=14),
            xaxis=dict(title_font_size=16, tickfont_size=14),
            yaxis=dict(title_font_size=16, tickfont_size=14),
            legend=dict(font_size=14)
        )
        vol_return_fig.add_shape(
            type='line',
            x0=summary_df["return_volatility"].min(),
            x1=summary_df["return_volatility"].max(),
            y0=summary_df["avg_daily_return"].min(),
            y1=summary_df["avg_daily_return"].max(),
            line=dict(color='lightgray', dash='dash')
        )
        
        # Correlation Analysis (Collapsible) - UPDATED WITH DATE RANGE
        corr_lags = ["same_day_corr", "1_day_ahead_corr", "2_day_ahead_corr", "3_day_ahead_corr", "5_day_ahead_corr"]
        corr_data = summary_df.melt(id_vars=["ticker"], value_vars=corr_lags, var_name="Lag", value_name="Correlation")
        grouped_bar_fig = px.bar(
            corr_data,
            x="ticker",
            y="Correlation",
            color="Lag",
            barmode="group",
            title=f"Sentiment-Price Correlation by Time Horizon<br><sub>{date_range_str}</sub>",
            labels={
                "ticker": "Security Ticker",
                "Correlation": "Correlation Coefficient"
            },
            template="plotly_dark",
            height=500
        )
        grouped_bar_fig.update_layout(
            title_font_size=22, font=dict(size=14),
            xaxis=dict(title_font_size=16, tickfont_size=14),
            yaxis=dict(title_font_size=16, tickfont_size=14),
            legend=dict(font_size=14)
        )
        
        # News Coverage (Collapsible) - UPDATED WITH DATE RANGE
        bar_fig = px.bar(
            summary_df,
            x="ticker",
            y="total_news_reddit",
            title=f"Total Media Coverage Volume<br><sub>{date_range_str}</sub>",
            labels={
                "ticker": "Security Ticker",
                "total_news_reddit": "Total Articles & Posts"
            },
            template="plotly_dark",
            height=500
        )
        bar_fig.update_layout(
            title_font_size=22, font=dict(size=14),
            xaxis=dict(title_font_size=16, tickfont_size=14),
            yaxis=dict(title_font_size=16, tickfont_size=14)
        )
        
        return summary_card, trending_bubble_fig, bubble_fig, vol_return_fig, grouped_bar_fig, bar_fig
