import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output

external_scripts = ['/assets/style.css']
app = dash.Dash(__name__,external_scripts=external_scripts)

server = app.server
app.config.suppress_callback_exceptions = True