# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
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

external_scripts = ['/assets/style.css']
app = dash.Dash(__name__,external_scripts=external_scripts)
server = app.server

conn = MySQLdb.connect(host="database-12.c3akcdosls7f.ap-southeast-1.rds.amazonaws.com", user="admin", passwd="Mathi#123", db="company")
cursor = conn.cursor()
cursor.execute('select Region, Country, Item_Type, Sales_Channel, Order_Date from sales_full_jeffin');

rows = cursor.fetchall()
str(rows)[0:300]

df1 = pd.DataFrame( [[ij for ij in i] for i in rows] )
df1.rename(columns={0: 'Region', 1: 'Country', 2: 'Type', 3: 'Sales Channel', 4:'Order Date'}, inplace=True);
print(df1)
df = pd.read_csv("stock_data.csv")

app.layout = html.Div([html.H1("Facebook Data Analysis", style={"textAlign": "center"}),
	dcc.Tab(label='Stock Prices', children=[
		html.Div([html.H1("Dataset Introduction", style={'textAlign': 'center'}),
		dash_table.DataTable(
			id='table',
			columns=[{"name": i, "id": i} for i in df.columns],
			data=df.iloc[0:5,:].to_dict("rows"),
		),
		dcc.Markdown('''Jeffin'''),
		dash_table.DataTable(
			id='tabledf1',
			columns=[{"name": i, "id": i} for i in df1.columns],
			data=df1.iloc[0:5,:].to_dict("rows"),

		),
		html.H1("Facebook Stocks High vs Lows", style={'textAlign': 'center', 'padding-top': 5}),
		dcc.Dropdown(id='my-dropdown',options=[{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
			multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
		dcc.Graph(id='highlow'),

])
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
	


if __name__ == '__main__':
    app.run_server(debug=True, port = 8050, host='0.0.0.0')
