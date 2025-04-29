import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os

# Load the data
DATA_PATH = os.path.join(os.path.dirname(__file__), "2015---Friuli-Venezia-Giulia---Gestione-finanziaria-Spese-Enti-Locali.csv")
df = pd.read_csv(DATA_PATH, delimiter=';', encoding='latin1', decimal=',', low_memory=False)

# Identify numeric columns for plotting
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
category_cols = [col for col in df.columns if col not in numeric_cols]

print("Numeric columns:", numeric_cols)
print("Category columns:", category_cols)
print("Head of DataFrame:")
print(df.head())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Dynamic Data Dashboard (Friuli-Venezia Giulia Financial Data)"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Chart Type"),
            dcc.Dropdown(
                id='chart-type',
                options=[
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Pie', 'value': 'pie'},
                    {'label': 'Scatter', 'value': 'scatter'},
                ],
                value='bar',
                clearable=False
            ),
            html.Br(),
            html.Label("X Axis (category)"),
            dcc.Dropdown(
                id='x-axis',
                options=[{'label': col, 'value': col} for col in category_cols],
                value=category_cols[0] if category_cols else None,
                placeholder='Select X axis',
            ),
            html.Br(),
            html.Label("Y Axis (numeric)"),
            dcc.Dropdown(
                id='y-axis',
                options=[{'label': col, 'value': col} for col in numeric_cols],
                value=numeric_cols[0] if numeric_cols else None,
                placeholder='Select Y axis',
            ),
            html.Br(),
            html.Label("Color/Group By (optional)"),
            dcc.Dropdown(
                id='color',
                options=[{'label': col, 'value': col} for col in category_cols],
                value=None,
                clearable=True
            ),
        ], width=3),
        dbc.Col([
            dcc.Graph(id='main-graph'),
        ], width=9)
    ])
], fluid=True)

@app.callback(
    Output('main-graph', 'figure'),
    [Input('chart-type', 'value'),
     Input('x-axis', 'value'),
     Input('y-axis', 'value'),
     Input('color', 'value')]
)
def update_graph(chart_type, x_axis, y_axis, color):
    if not (x_axis and y_axis):
        return {
            "layout": {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations": [
                    {
                        "text": "Please select both X and Y axes to display the chart.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 18}
                    }
                ]
            }
        }
    dff = df.copy()
    if chart_type == 'bar':
        fig = px.bar(dff, x=x_axis, y=y_axis, color=color)
    elif chart_type == 'line':
        fig = px.line(dff, x=x_axis, y=y_axis, color=color)
    elif chart_type == 'pie':
        fig = px.pie(dff, names=x_axis, values=y_axis, color=color)
    elif chart_type == 'scatter':
        fig = px.scatter(dff, x=x_axis, y=y_axis, color=color)
    else:
        fig = {}
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
