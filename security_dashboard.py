# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from dash import html, dcc, Input, Output
# import dash_bootstrap_components as dbc

# # Load your data
# summary_df = pd.read_csv("security_correlation_summary_20251031_234833.csv")
# detailed_df = pd.read_csv("security_correlation_detailed_20251031_234833.csv")

# def get_security_layout():
#     return dbc.Container([
#         dbc.Row(dbc.Col(html.H2("All Securities Sentiment & Performance Dashboard", className="text-center mt-3 mb-4", style={"fontWeight": "bold"}))),
#         dcc.Interval(id='security-interval', interval=60000, n_intervals=0),
#         dbc.Row([dbc.Col(id="security-summary-section", width=12)], className="mb-4"),
#         dbc.Row([dbc.Col(dcc.Graph(id="security-bubble-sentiment-price"), width=12)], className="mb-4"),
#         dbc.Row([dbc.Col(dcc.Graph(id="security-grouped-bar-corr"), width=12)], className="mb-4"),
#         dbc.Row([dbc.Col(dcc.Graph(id="security-bar-news-coverage"), width=12)], className="mb-4"),
#         dbc.Row([dbc.Col(dcc.Graph(id="security-scatter-vol-return"), width=12)], className="mb-4"),
#     ], fluid=True, style={"backgroundColor": "#1a1a1a", "color": "white", "paddingBottom": "30px"})

# def register_security_callbacks(app):
#     @app.callback(
#         [
#             Output("security-summary-section", "children"),
#             Output("security-bubble-sentiment-price", "figure"),
#             Output("security-grouped-bar-corr", "figure"),
#             Output("security-bar-news-coverage", "figure"),
#             Output("security-scatter-vol-return", "figure"),
#         ],
#         Input("security-interval", "n_intervals")
#     )
#     def update_all(n):
#         total_news = summary_df["total_news_reddit"].sum()
#         avg_sentiment = summary_df["avg_sentiment"].mean()
#         price_change = summary_df["price_change_pct"].mean()
#         num_securities = len(summary_df)

#         summary_card = dbc.Card([
#             dbc.CardHeader(html.H5("Overall Summary")),
#             dbc.CardBody([
#                 html.P(f"Total News/Reddit Volume: {total_news}"),
#                 html.P(f"Average Sentiment: {avg_sentiment:.3f}"),
#                 html.P(f"Average Price Change %: {price_change:.2f}%"),
#                 html.P(f"Number of Securities: {num_securities}"),
#             ], style={"color": "white"})
#         ], style={"backgroundColor": "#212121", "marginBottom": "1rem"})

#         bubble_fig = px.scatter(
#             detailed_df,
#             x="avg_sentiment",
#             y="price_change_pct",
#             size="valid_data_points",
#             color="ticker",
#             title="Sentiment vs Price with Data Points by Security",
#             template="plotly_dark",
#             height=400
#         )

#         corr_lags = ["same_day_corr", "1_day_ahead_corr", "2_day_ahead_corr", "3_day_ahead_corr", "5_day_ahead_corr"]
#         corr_data = summary_df.melt(id_vars=["ticker"], value_vars=corr_lags, var_name="Lag", value_name="Correlation")
#         grouped_bar_fig = px.bar(
#             corr_data,
#             x="ticker",
#             y="Correlation",
#             color="Lag",
#             barmode="group",
#             title="Correlation Over Time Horizons for All Securities",
#             template="plotly_dark",
#             height=400
#         )

#         bar_fig = px.bar(
#             summary_df,
#             x="ticker",
#             y="total_news_reddit",
#             title="Total News/Reddit Coverage by Security",
#             template="plotly_dark",
#             height=400
#         )

#         vol_return_fig = px.scatter(
#             summary_df,
#             x="return_volatility",
#             y="avg_daily_return",
#             color="ticker",
#             title="Risk vs Reward for All Securities",
#             template="plotly_dark",
#             height=400
#         )
#         vol_return_fig.add_shape(
#             type='line',
#             x0=summary_df["return_volatility"].min(),
#             x1=summary_df["return_volatility"].max(),
#             y0=summary_df["avg_daily_return"].min(),
#             y1=summary_df["avg_daily_return"].max(),
#             line=dict(color='lightgray', dash='dash')
#         )

#         return summary_card, bubble_fig, grouped_bar_fig, bar_fig, vol_return_fig

import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, Input, Output

# Load Data
summary_df = pd.read_csv("security_correlation_summary_20251031_234833.csv")
detailed_df = pd.read_csv("security_correlation_detailed_20251031_234833.csv")

# Prepare Sentiment Score Statistics (example aggregation)
sentiment_stats_df = detailed_df.groupby('ticker').agg(
    Avg_Score=('avg_sentiment', 'mean'),
    Min_Score=('avg_sentiment', 'min'),
    Max_Score=('avg_sentiment', 'max'),
    Article_Count=('valid_data_points', 'sum')
).reset_index()

# Correlation Analysis Details columns
corr_cols = ['ticker', 'same_day_corr', '1_day_ahead_corr', 'same_day_significant', '1_day_ahead_significant']
corr_stats_df = summary_df[corr_cols]
perf_summary_df = summary_df[
    ['ticker', 'price_change_pct', 'avg_sentiment', 'data_points', 'avg_daily_return', 
     'same_day_corr', '1_day_ahead_corr', '1_day_ahead_significant']
]


def get_security_layout():
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("All Securities Sentiment & Performance Dashboard",
                                className="text-center mt-3 mb-4", style={"fontWeight": "bold"}))),
        dcc.Interval(id='security-interval', interval=60000, n_intervals=0),

        # Summary Section placeholder
        dbc.Row([dbc.Col(id="security-summary-section", width=12)], className="mb-4"),

        # Include your existing graphs here
        dbc.Row([dbc.Col(dcc.Graph(id="security-bubble-sentiment-price"), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id="security-grouped-bar-corr"), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id="security-bar-news-coverage"), width=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id="security-scatter-vol-return"), width=12)], className="mb-4"),
        
        # New Tables section
        dbc.Row([
            dbc.Col([
                html.H4("Security Performance & Correlation Summary"),
                dash_table.DataTable(
                    id='table-summary',
                    data=perf_summary_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in perf_summary_df.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white', 'fontSize': '12px'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold', 'border': '1px solid #444', 'textAlign': 'left'},
                    style_data={'border': '1px solid #444'},
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=[
                    {'if': {'column_id': 'price_change_pct', 'filter_query': '{price_change_pct} > 0'}, 'color': '#22c55e', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'price_change_pct', 'filter_query': '{price_change_pct} < 0'}, 'color': '#ef4444', 'fontWeight': 'bold'},
                    {'if': {'column_id': '1_day_ahead_significant', 'filter_query': '{1_day_ahead_significant} = true'}, 'backgroundColor': '#1e3a1e'}
                ],
                    export_format='csv',
                    export_headers='display',
                    page_size=10,
                )
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H4("Sentiment Score Distribution & Statistics"),
                dash_table.DataTable(
                    id='table-sentiment',
                    data=sentiment_stats_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in sentiment_stats_df.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white', 'fontSize': '12px'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold', 'border': '1px solid #444', 'textAlign': 'left'},
                    style_data={'border': '1px solid #444'},
                    style_table={'overflowX': 'auto'},
                    export_format='csv',
                    export_headers='display',
                    page_size=10,
                )
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H4("Correlation Analysis Details"),
                dash_table.DataTable(
                    id='table-correlation',
                    data=corr_stats_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in corr_stats_df.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white', 'fontSize': '12px'},
                    style_header={'backgroundColor': '#1e1e1e', 'fontWeight': 'bold', 'border': '1px solid #444', 'textAlign': 'left'},
                    style_data={'border': '1px solid #444'},
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=[
                        {'if': {'column_id': 'same_day_sig', 'filter_query': '{same_day_sig} = true'},
                         'backgroundColor': '#1e3a1e', 'fontWeight': 'bold'},
                        {'if': {'column_id': '1d_ahead_sig', 'filter_query': '{1d_ahead_sig} = true'},
                         'backgroundColor': '#1e3a1e', 'fontWeight': 'bold'}
                    ],
                    export_format='csv',
                    export_headers='display',
                    page_size=10,
                )
            ])
        ], className="mb-4")

    ], fluid=True, style={"backgroundColor": "#1a1a1a", "color": "white", "paddingBottom": "30px"})


def register_security_callbacks(app):
    @app.callback(
        [
            Output("security-summary-section", "children"),
            Output("security-bubble-sentiment-price", "figure"),
            Output("security-grouped-bar-corr", "figure"),
            Output("security-bar-news-coverage", "figure"),
            Output("security-scatter-vol-return", "figure"),
        ],
        Input("security-interval", "n_intervals")
    )
    def update_all(n):
        total_news = summary_df["total_news_reddit"].sum()
        avg_sentiment = summary_df["avg_sentiment"].mean()
        price_change = summary_df["price_change_pct"].mean()
        num_securities = len(summary_df)

        summary_card = dbc.Card([
            dbc.CardHeader(html.H5("Overall Summary")),
            dbc.CardBody([
                html.P(f"Total News/Reddit Volume: {total_news}"),
                html.P(f"Average Sentiment: {avg_sentiment:.3f}"),
                html.P(f"Average Price Change %: {price_change:.2f}%"),
                html.P(f"Number of Securities: {num_securities}"),
            ], style={"color": "white"})
        ], style={"backgroundColor": "#212121", "marginBottom": "1rem"})

        bubble_fig = px.scatter(
            detailed_df,
            x="avg_sentiment",
            y="price_change_pct",
            size="valid_data_points",
            color="ticker",
            title="Sentiment vs Price with Data Points by Security",
            template="plotly_dark",
            height=400
        )

        corr_lags = ["same_day_corr", "1_day_ahead_corr", "2_day_ahead_corr", "3_day_ahead_corr", "5_day_ahead_corr"]
        corr_data = summary_df.melt(id_vars=["ticker"], value_vars=corr_lags, var_name="Lag", value_name="Correlation")

        grouped_bar_fig = px.bar(
            corr_data,
            x="ticker",
            y="Correlation",
            color="Lag",
            barmode="group",
            title="Correlation Over Time Horizons for All Securities",
            template="plotly_dark",
            height=400
        )

        bar_fig = px.bar(
            summary_df,
            x="ticker",
            y="total_news_reddit",
            title="Total News/Reddit Coverage by Security",
            template="plotly_dark",
            height=400
        )

        vol_return_fig = px.scatter(
            summary_df,
            x="return_volatility",
            y="avg_daily_return",
            color="ticker",
            title="Risk vs Reward for All Securities",
            template="plotly_dark",
            height=400
        )
        vol_return_fig.add_shape(
            type='line',
            x0=summary_df["return_volatility"].min(),
            x1=summary_df["return_volatility"].max(),
            y0=summary_df["avg_daily_return"].min(),
            y1=summary_df["avg_daily_return"].max(),
            line=dict(color='lightgray', dash='dash')
        )
        return summary_card, bubble_fig, grouped_bar_fig, bar_fig, vol_return_fig
