# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from app import app
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import mysql.connector as MySQLdb



'''df = pd.read_csv("stock_data.csv")'''
df2 = pd.read_csv("dataset_Facebook.csv",";")

conn = MySQLdb.connect(host="database-12.c3akcdosls7f.ap-southeast-1.rds.amazonaws.com", user="admin", passwd="Mathi#123", db="company")
cursor = conn.cursor()
cursor.execute('select Date, Open, High, Low, Close, Volume, OpenInt, Stock from stock_data');

rows = cursor.fetchall()
str(rows)[1:300]

df = pd.DataFrame( [[ij for ij in i] for i in rows] )
df.rename(columns={0: 'Date', 1: 'Open', 2: 'High', 3: 'Low', 4:'Close', 5:'Volume', 6:'OpenInt', 7:'Stock'}, inplace=True);


app.layout = html.Div([html.H1("Facebook Data Analysis", style={"textAlign": "center"}), dcc.Markdown('''Jeffin
'''),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Stock Prices', children=[
html.Div([html.H1("Dataset Introduction", style={'textAlign': 'center'}),
dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.iloc[0:5,:].to_dict("rows"),
),
    html.H1("Facebook Stocks High vs Lows", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown',options=[{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
        multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='highlow'),  dash_table.DataTable(
    id='table2',
    columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
    data= df.describe().reset_index().to_dict("rows"),
),
    html.H1("Facebook Market Volume", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown2',options=[{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
        multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='volume'),
    html.H1("Scatter Analysis", style={'textAlign': 'center', 'padding-top': -10}),
    dcc.Dropdown(id='my-dropdown3',
                 options=[{'label': 'Tesla', 'value': 'TSLA'}, {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Facebook', 'value': 'FB'}, {'label': 'Microsoft', 'value': 'MSFT'}],
                 value= 'FB',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
    dcc.Dropdown(id='my-dropdown4',
                 options=[{'label': 'Tesla', 'value': 'TSLA'}, {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Facebook', 'value': 'FB'}, {'label': 'Microsoft', 'value': 'MSFT'}],
                 value= 'AAPL',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
  dcc.RadioItems(id="radiob", value= "High", labelStyle={'display': 'inline-block', 'padding': 10},
                 options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"} , {'label': "Volume", 'value': "Volume"}],
 style={'textAlign': "center", }),
    dcc.Graph(id='scatter')
], className="container"),
]),


dcc.Tab(label='Performance Metrics', children=[
html.Div([html.H1("Facebook Metrics Distributions", style={"textAlign": "center"}),
            html.Div([html.Div([dcc.Dropdown(id='feature-selected1',
                                             options=[{'label': i.title(), 'value': i} for i in
                                                      df2.columns.values[1:]],
                                             value="Type")],
                               style={"display": "block", "margin-left": "auto", "margin-right": "auto",
                                      "width": "80%"}),
                      ],),
            dcc.Graph(id='my-graph2'),
dash_table.DataTable(
    id='table3',
    columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
    data= df.describe().reset_index().to_dict("rows"),
),
            html.H1("Paid vs Free Posts by Category", style={'textAlign': "center", 'padding-top': 5}),
     html.Div([
         dcc.RadioItems(id="select-survival", value=str(1), labelStyle={'display': 'inline-block', 'padding': 10},
                        options=[{'label': "Paid", 'value': str(1)}, {'label': "Free", 'value': str(0)}], )],
         style={'textAlign': "center", }),
     html.Div([html.Div([dcc.Graph(id="hist-graph", clear_on_unhover=True, )]), ]),
        ], className="container"),
]),

])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["High"],mode='lines',
            opacity=0.7,name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Low"],mode='lines',
            opacity=0.6,name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Volume"],mode='lines',
            opacity=0.7,name=f'Volume {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Transactions Volume"} ,   paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('scatter', 'figure'),
              [Input('my-dropdown3', 'value'), Input('my-dropdown4', 'value'), Input("radiob", "value"),])
def update_graph(stock, stock2, radioval):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    if (stock == None) or (stock2 == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock][radioval][-1000:], y=df[df["Stock"] == stock2][radioval][-1000:],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} of {dropdown[stock]} vs {dropdown[stock2]} Over Time (1000 iterations)",
                xaxis={"title": stock,}, yaxis={"title": stock2},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('feature-selected1', 'value')])
def update_graph(selected_feature1):
    if selected_feature1 == None:
        selected_feature1 = 'Type'
        trace = go.Histogram(x= df2.Type,
                             marker=dict(color='rgb(0, 0, 100)'))
    else:
        trace = go.Histogram(x=df2[selected_feature1],
                         marker=dict(color='rgb(0, 0, 100)'))

    return {
        'data': [trace],
        'layout': go.Layout(title=f'Metric: {selected_feature1.title()}',
                            colorway=["#EF963B", "#EF533B"], hovermode="closest",
                            xaxis={'title': "Distribution", 'titlefont': {'color': 'black', 'size': 14},
                                   'tickfont': {'size': 14, 'color': 'black'}},
                            yaxis={'title': "Frequency", 'titlefont': {'color': 'black', 'size': 14, },
                                   'tickfont': {'color': 'black'}},     paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}


@app.callback(
    dash.dependencies.Output("hist-graph", "figure"),
    [dash.dependencies.Input("select-survival", "value"),])
def update_graph(selected):
    dff = df2[df2["Paid"] == int(selected)]
    trace = go.Histogram(x=dff["Type"], marker=dict(color='rgb(0, 0, 100)'))
    layout = go.Layout(xaxis={"title": "Post distribution categories", "showgrid": False},
                       yaxis={"title": "Frequency", "showgrid": False},    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)' )
    figure2 = {"data": [trace], "layout": layout}

    return figure2


def update_graph(stock , radioval):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    trace2 = []
    train_data = df[df['Stock'] == stock][-1000:][0:int(1000 * 0.8)]
    test_data = df[df['Stock'] == stock][-1000:][int(1000 * 0.8):]
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=train_data['Date'],y=train_data[radioval], mode='lines',
            opacity=0.7,name=f'Training Set',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y=test_data[radioval],mode='lines',
            opacity=0.6,name=f'Test Set',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} Train-Test Sets for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


def update_graph(stock, radioval):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        test_data = df[df['Stock'] == stock][-1000:][int(1000 * 0.8):]
        train_data = df[df['Stock'] == stock][-1000:][0:int(1000 * 0.8)]
        train_ar = train_data[radioval].values
        test_ar = test_data[radioval].values
        history = [x for x in train_ar]
        predictions = list()
        for t in range(len(test_ar)):
            model = ARIMA(history, order=(3, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_ar[t]
            history.append(obs)
        error = mean_squared_error(test_ar, predictions)
        trace1.append(go.Scatter(x=test_data['Date'],y=test_data['High'],mode='lines',
            opacity=0.6,name=f'Actual Series',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y= np.concatenate(predictions).ravel(), mode='lines',
            opacity=0.7,name=f'Predicted Series (MSE: {error})',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} ARIMA Predictions vs Actual for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True, port = 8050, host='0.0.0.0')
	
