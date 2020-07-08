import base64
import datetime
import io
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

app = dash.Dash()
server = app.server

#df2 = pd.read_csv("dataset_Stock.csv")
#df2 = pd.read_csv("JiraReport_jeffin.csv")
#df2 = pd.read_csv("JiraReport_Jithu.csv")
#df2 = pd.read_csv("dataset_Facebook.csv")
df2 = pd.read_csv("Jira.csv")



app.layout = html.Div([
    # Setting the main title of the Dashboard
    html.H1("Jira Report Analysis", style={"textAlign": "center"}),
    
    # Dividing the dashboard in tabs

        # Defining the layout of the second tab
        dcc.Tab(label='Performance Metrics', children=[
            html.H1(" ", 
                    style={"textAlign": "center"}),
            # Adding a dropdown menu and the subsequent histogram graph
            html.Div([
                      html.Div([dcc.Dropdown(id='feature-selected1',
                      options=[{'label': i.title(),
                                'value': i} for i in df2.columns.values[0:]], 
                                 value='Status')],
                                 className="five columns", 
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "60%"}),
                    ], className="row",
                    style={"padding": 50, "width": "60%", 
                           "margin-left": "auto", "margin-right": "auto"}),
                    dcc.Graph(id='my-graph2'),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ', html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-data-upload'),        
        ])
      
    ])


@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('feature-selected1', 'value')])
    
               
def update_graph(selected_feature1):
    if selected_feature1 == None:
        selected_feature1 = 'Type'
        trace = go.Histogram(x= df2.Type,marker=dict(color='rgb(0, 0, 100)'))
    else:
        trace = go.Histogram(x=df2[selected_feature1],marker=dict(color='rgb(0, 0, 100)'))

    return {
        'data': [trace],
        'layout': go.Layout(title=f'Metrics considered: 
                            {selected_feature1.title()}',
                            colorway=["#EF963B", "#EF533B"], hovermode="closest",
                            xaxis={'title': "Distribution", 
                                   'titlefont': {'color': 'black', 'size': 14},
                                   'tickfont': {'size': 14, 'color': 'black'}},
                            yaxis={'title': "Frequency", 
                                   'titlefont': {'color': 'black', 'size': 14, },
                                   'tickfont': {'color': 'black'}})}

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
               
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
        
if __name__ == '__main__':
    app.run_server(debug=True)
