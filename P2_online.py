# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:23:12 2023

@author: Utilizador
"""

#First import the main libraries
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split


#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

#import os
from dash.exceptions import PreventUpdate

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


white_text_style = {'color': 'white'}


#Load data
#Raw Data
df_total = pd.read_csv("df_total.csv", index_col=0, parse_dates=True)
columns = df_total.columns.tolist()
start_date = df_total.index.min()
end_date = df_total.index.max()


#Create a table for 2017 and 2018
# Set a variable with the cut-off date for the 2019 set
test_cutoff_date = '2019-01-01'

# Split the dataset into training and test sets
df_data = df_total.loc[df_total.index < test_cutoff_date] #dataset with values from 2017 and 2018
df_2019 = df_total.loc[df_total.index >= test_cutoff_date] #dataset with values from 2019 (which are going to predict)

df_data = df_data.dropna() #Clean the first dataframe

df_dataFS = df_data.copy()
df_dataFS = df_dataFS.drop("Power (kW)", axis=1)

#Real data 2019
df = pd.read_csv('forecast_data2.csv')
df = df.drop(0)
df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type
df2=df.iloc[:,1:5]
X2=df2.values
fig1 = px.line(df, x="Date", y=df.columns[1:4])# Creates a figure with the raw data

df_real = pd.read_csv('df_real2.csv')
df_real['Date'] = pd.to_datetime(df_real['Date'])
y=df_real['Power (kW)'].values

df_meteo_2019 = df_2019.drop('Power (kW)', axis=1)
dates_to_drop = df_meteo_2019[~df_meteo_2019.index.isin(df_real['Date'])].index
df_meteo_2019 = df_meteo_2019.drop(dates_to_drop)




X = None
Y = None

X_train = None
X_test = None
y_train = None
y_test = None

X_2019 = None



fig2 = px.line(df_real, x='Date', y='Power (kW)')



# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    # Apply some CSS styles to the table
    table_style = {
        'borderCollapse': 'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'border': '1px solid #ddd',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px'
    }
    
    th_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold',
        'color': '#333'
    }
    
    td_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left'
    }
    
    return html.Table(
        # Apply the table style
        style=table_style,
        children=[
            # Add the table header
            html.Thead(
                html.Tr([
                    html.Th('Index', style=th_style),
                    *[html.Th(col, style=th_style) for col in dataframe.columns]
                ])
            ),
            # Add the table body
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.index[i], style=td_style),
                    *[html.Td(dataframe.iloc[i][col], style=td_style) for col in dataframe.columns]
                ])
                for i in range(min(len(dataframe), max_rows))
            ])
        ]
    )




def generate_graph(df, columns, start_date, end_date):
    filtered_df = df.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'title': column, 'overlaying': 'y', 'side': 'right', 'position': i * 0.1})
    
    # Define the data and layout of the figure
    data = [go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column) for column in filtered_df.columns]
    layout = go.Layout(title=', '.join(columns), xaxis_title='Date')
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = go.Figure(data=data, layout=layout)
    
    return fig





app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={'backgroundColor': 'white'},children = [
    html.H1('IST Energy Forecast tool (kWh)'),
    html.Div(id='df_total', children=df_total.to_json(orient='split'), style={'display': 'none'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', children=[
            html.Div([
                html.H2("Raw Data"),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': i, 'value': i} for i in df_total.columns],
                    value=[df_total.columns[0]],
                    multi=True
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=df_total.index.min(),
                    max_date_allowed=df_total.index.max(),
                    start_date=df_total.index.min(),
                    end_date=df_total.index.max()
                ),
                dcc.Graph(id='graph'),
            ])
        ]),
        
        dcc.Tab(label='Exploratory Data Analysis', value='tab-2', children=[
            html.Div([
                html.H2("Exploratory Data Analysis"),
                dcc.Dropdown(
                    id='feature1',
                    options=[{'label': col, 'value': col} for col in df_total.columns],
                    value=df_data.columns[0]
                    ),
                dcc.Dropdown(
                    id='feature2',
                    options=[{'label': col, 'value': col} for col in df_total.columns],
                    value=df_data.columns[1]
                    ),
                dcc.Graph(id='scatter-plot'),
                dcc.Dropdown(
                    id='feature-boxplot',
                    options=[{'label': col, 'value': col} for col in df_total.columns],
                    value=df_total.columns[1]
                    ),
                dcc.Graph(id='box-plot')
                ])
            ]),
        
        
        
        
        
        
        dcc.Tab(label='Feature Selection', value='tab-3', children=[
    html.Div([
        html.H2("Feature Selection"),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': col, 'value': col} for col in df_dataFS.columns],
            value=[df_dataFS.columns[0]],
            multi=True
        ),
        html.Div(id='feature-table-div'),
        html.Button('Generate Train-Test Split', id='split-button'),
        html.Div(id='split-values'),
        html.Div([
            html.H6(""),
            html.Pre(id="x-values", style=white_text_style)
        ]),
        html.Div([
            html.H6(""),
            html.Pre(id="y-values", style=white_text_style)
        ]),
        html.Div([
            html.H6(""),
            html.Pre(id="x-2019-values", style=white_text_style)
            ]),
        
        ])
    ]),
        
        
        
dcc.Tab(label='Regression Models', value='tab-4', children=[
    html.Div([
        html.H2("Regression Models"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Linear Regression', 'value': 'linear'},
                {'label': 'Random Forests', 'value': 'random_forests'},
                {'label': 'Bootstrapping', 'value': 'bootstrapping'},
                {'label': 'Decision Tree Regressor', 'value': 'decision_trees'}
            ],
            value='linear'
        ),
        html.Button('Train Model', id='train-model-button'),
        #html.Div(id='df-model-div')
    ]), 
    html.Div([
        html.H2(""),
        dcc.Loading(
            id="loading-1",
            children=[html.Div([dcc.Graph(id="lr-graph")])]
        )
    ]),
]),
        
        
        dcc.Tab(label='Model Deployment', value='tab-6', children=[
    html.Div([
        html.H2('Model Deployment'),
        dcc.Graph(id='time-series-plot', figure=fig2),
        html.Button('Run Model', id='button_model'),
        #html.Button('Reset Model', id='button_model_reset'),
        html.Div(id='model-performance-table')
    ])
]),
    ]),
    html.Div(id='tabs-content')
])

# Define the callbacks
@app.callback(Output('graph', 'figure'),
              Input('column-dropdown', 'value'),
              Input('date-picker', 'start_date'),
              Input('date-picker', 'end_date')
)
def update_figure(columns, start_date, end_date):
    
    filtered_df = df_total.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1})
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': {'text': ', '.join(columns)}, 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig

@app.callback(Output('scatter-plot', 'figure'),
              Input('feature1', 'value'),
              Input('feature2', 'value'))


def update_scatter_plot(feature1, feature2):
    fig = {
        'data': [{
            'x': df_total[feature1],
            'y': df_total[feature2],
            'mode': 'markers'
        }],
        'layout': {
            'title': f'{feature1} vs {feature2}',
            'xaxis': {'title': feature1},
            'yaxis': {'title': feature2},
        }
    }
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    Input('feature-boxplot', 'value')
)
def update_box_plot(feature_boxplot):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_total[feature_boxplot], name=feature_boxplot))
    fig.update_layout(title=f"Box Plot for {feature_boxplot}")
    return fig


@app.callback(
    Output('feature-table-div', 'children'),
    #Output('df-model-div', 'children'),  # add this line to update the content of the 'df-model-div' component
    Input('feature-dropdown', 'value')
)
def update_feature_table(selected_features):
    if selected_features:
        global df_model
        df_model = df_dataFS[selected_features]
        table = generate_table(df_model)
        return table  # add this line to update the content of the 'df-model-div' component
    else:
        return html.Div()  # return empty divs if no features are selected
    

@app.callback(
    Output('x-values', 'children'),
    Output('y-values', 'children'),
    Output('x-2019-values', 'children'),
    Input('feature-dropdown', 'value')
)
def update_x_y(selected_features):
    global X, Y, X_2019
    if selected_features:
        X = df_model.iloc[:, :].values
        Y = df_data.loc[:, 'Power (kW)'].values
        X_2019 = df_meteo_2019[selected_features].values
        return str(X), str(Y), str(X_2019)
    else:
        return "", ""

@app.callback(
    Output('split-values', 'children'),
    Input('split-button', 'n_clicks')
)
def generate_train_test_split(n_clicks):
    global X_train, X_test, y_train, y_test
    if n_clicks:
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        return 'Done!'
    else:
        return ""
    
# define global variables
y_pred_list = []
y_pred_2019 = []

@app.callback(
    Output('lr-graph', 'figure'),
    Input('train-model-button', 'n_clicks'),
    State('model-dropdown', 'value')
)
def train_and_predict(n_clicks, model_type):
    global y_pred_list, y_pred_2019  # access global variable

    if n_clicks is None:
        return dash.no_update
    else:
        if model_type == 'linear':
            from sklearn import linear_model
            
            # Create linear regression object
            model = linear_model.LinearRegression()

            # Train the model using the training sets
            model.fit(X_train, y_train)

            #Save the trained model
            with open('model.pkl', 'wb') as file:
              pickle.dump(model, file)
              file.close()

            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
            
            y_pred2019 = model.predict(X_2019)
            y_pred_2019 = y_pred2019

            # Generate scatter plot of predicted vs actual values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title='Linear Regression Predictions')
            return fig
        elif model_type == 'random_forests':            
            parameters = {'bootstrap': True,
                          'min_samples_leaf': 3,
                          'n_estimators': 200, 
                          'min_samples_split': 15,
                          'max_features': 'sqrt',
                          'max_depth': 20,
                          'max_leaf_nodes': None}
            
            # Create random forests model object
            model = RandomForestRegressor(**parameters)

            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Save the trained model
            with open('model.pkl', 'wb') as file:
                pickle.dump(model, file)
                file.close()

            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
            
            y_pred2019 = model.predict(X_2019)
            y_pred_2019 = y_pred2019

            # Generate scatter plot of predicted vs actual values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title='Random Forests Predictions')
            return fig
        
        elif model_type == 'bootstrapping':
            
            model = BaggingRegressor()
            model.fit(X_train, y_train)

            # Save the trained model
            with open('model.pkl', 'wb') as file:
                pickle.dump(model, file)
                file.close()

            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
            
            y_pred2019 = model.predict(X_2019)
            y_pred_2019 = y_pred2019
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title='Bootstrapping Predictions')
            
            return fig
        


        elif model_type == 'decision_trees':
            model = DecisionTreeRegressor() 
            model.fit(X_train, y_train)

            # Save the trained model
            with open('model.pkl', 'wb') as file:
                pickle.dump(model, file)
                file.close()

            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
            
            y_pred2019 = model.predict(X_2019)
            y_pred_2019 = y_pred2019
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title='Decision Tree Regressor Predictions')
            
            return fig

@app.callback(
    Output('time-series-plot', 'figure'),
    Output('model-performance-table', 'children'),
    Input('button_model', 'n_clicks')
)
def run_model(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        if 'Date' in df_real.columns:
            df_real['Date'] = pd.to_datetime(df_real['Date'])
            df_real.set_index('Date', inplace=True)

        
        #df_real.set_index('Date', inplace=True)
        fig = go.Figure(layout=go.Layout(title='Real vs Predicted Power Consumption'))
        fig.add_scatter(x=df_real.index, y=df_real['Power (kW)'], name='Real Power (kW)')
        fig.add_scatter(x=df_real.index, y=y_pred_2019, name='Predicted Power (kW)')
        
        # Calculate model performance metrics
        MAE = metrics.mean_absolute_error(df_real['Power (kW)'], y_pred_2019)
        MBE = np.mean(df_real['Power (kW)'] - y_pred_2019)
        MSE = metrics.mean_squared_error(df_real['Power (kW)'], y_pred_2019)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(df_real['Power (kW)'])
        nmbe = MBE / np.mean(df_real['Power (kW)'])

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'Model':['Model Selected'],'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table = generate_table(df_metrics)
        
    return fig, table

    
if __name__ == '__main__':
    app.run_server()