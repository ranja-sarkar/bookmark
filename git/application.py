import os
import numpy as np
from flask import request
from flask import Flask, jsonify, send_file
from flask_cors import CORS, cross_origin
import logging as logger
from flask_restful import Api
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_table as dt
import pickle
import pandas as pd
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

logger.basicConfig(level="DEBUG")

# Flask App
flask_app = Flask(__name__, static_url_path='')
flask_app.config['JSON_SORT_KEYS'] = False
restServerInstance = Api(flask_app)
cors = CORS(flask_app, resources={r"/*": {"origins": "*"}})
flask_app.config['CORS_HEADERS'] = 'Content-Type'

app = dash.Dash(__name__, server=flask_app, url_base_pathname='/')
app.title = 'Digital Polymer'
# MyApp Design
colors = {'background': '#FFD500', 'text': '#ED1C24'}
app.layout = html.Div([
    html.H1('Search for non-metallics data', style = {'color': colors['text'], 'backgroundColor': colors['background']}),
    dcc.Input(placeholder = '', value = None, id = 'box', type = 'text', style = {'width': '25%'}),
    html.Button(id = 'button', children = 'Search', style = {'width': '5%'}),
    html.Hr(),
    html.Div(id = 'output')
                     ],

    style = {'textAlign': 'center', 'font-family': 'futura medium', 'font-size': '20px'}
                    )


# Dash' attribute callback for output
@app.callback(Output('output', 'children'), [Input('button', 'n_clicks')], [State('box', 'value')])
@flask_app.route("/")
def update_output(n_clicks, keyword):
    """ Search result """
    if n_clicks and n_clicks > 0:
        filepath = 'grams_144_summary.pkl'
        with open(filepath, 'rb') as f:
            search_word = pickle.load(f)

        try:
            rr = search_word[str(keyword)]
            df = pd.DataFrame(rr, columns=('Title', 'Summary', 'Score'))
            df = df.sort_values('Score', ascending=False)
            df2 = pd.DataFrame({'Title': os.listdir('assets')})
            links = [html.A(i, href = app.get_asset_url(i), title = "Link", target = "_blank", rel = "noopener noreferrer") for i in df2.Title]
            df2['Source'] = links
            df = df.merge(df2, how = 'left', on = 'Title')
            df = df.drop(['Title'], axis = 1)
            table = dbc.Table.from_dataframe(df[['Source', 'Summary', 'Score']])
            return html.Div(['{} result(s) found for {}'.format(len(rr), str(keyword)),
                             html.Br(),
                             html.Div(id = 'output_table'), table
                             #html.Div(id='output_table'), dt.DataTable(id='table', data=df.to_dict('records'),
                             #                                          columns=[{"name": i, "id": i} for i in
                             #                                                   df.columns],
                             #                                          style_cell={'textAlign': 'left',
                             #                                                      'overflow': 'hidden',
                             #                                                      'textOverflow': 'ellipsis',
                             #                                                      'font-family': 'futura medium'},
                             #                                          style_data={'whiteSpace': 'normal'},
                             #                                          style_header=
                             #                                          {'backgroundColor': '#FFD500',
                             #                                           'color': '#ED1C24', 'textAlign': 'center',
                             #                                           'font-family': 'futura medium'},
                             #                                          style_as_list_view=False)
                             ])
        except Exception:
            if len(keyword.split()) > 3:
                return html.Div([html.H4('Please restrict keyphrase to a maximum of 3 words')])
            else:
                return html.Div(['No result(s) found for {}'.format(str(keyword))])


if __name__ == '__main__':
    print("Starting Flask Server")
    # app.run_server(host='0.0.0.0', port=5000)
    flask_app.run(host="0.0.0.0", port=5000, debug=True)
