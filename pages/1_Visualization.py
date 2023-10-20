from pages.Visualization.Stops import plot_stop_activity
from pages.Visualization.Heatmaps import heat_map
from Setup import *
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import time as tm
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from streamlit_extras.mandatory_date_range import date_range_picker
import os
import pickle
import copy
from datetime import datetime, timedelta
from datetime import time as dt_time
import itertools
from calendar import month_name
from dateutil.relativedelta import relativedelta
from streamlit_extras.dataframe_explorer import dataframe_explorer
#from streamlit_profiler import Profiler
import cProfile
from pyinstrument import Profiler
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


class Device:
    def __init__(self, device_name):
        device = st.session_state.devices_list[device_name]
        self.name = device['name']
        self.data = device['data']
        self.settings = device['settings']
        if "state_info" in device:
            self.state_info = device['state_info']
        self.stop_info = device['stop_info']
        self.date_time_info = device['date_time_info']
        if "plot_variables" not in device:
            self.plot_variables = {'Time': {'Name': None, 'Column': None, 'Unit': None},
                                        'Numerical': {'Name': None, 'Column': None, 'Unit': None},
                                        'Categorical': {'Name': None, 'Column': None, 'Unit': None}}
        else:
            self.plot_variables = device['plot_variables']
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def get_var_columns(self, column):
        columns = []
        if column in self['settings']:
            for var_type, var in self['settings'][column].items():
                columns.append(var['column'])
        return columns
    
    def get_var_names(self, column):
        columns = []
        if column in self['settings']:
            for var_type, var in self['settings'][column].items():
                columns.append(var['variable_name'])
        return columns
    
    def add_plot_var(self, var_type, var_name):
        self.plot_variables[var_type]['Name'] = var_name
        self.plot_variables[var_type]['Column'] = self['settings'][var_type][var_name]['column']
        if 'unit' in self['settings'][var_type][var_name]:
            self.plot_variables[var_type]['Unit'] = self['settings'][var_type][var_name]['unit']

    def date_filter(self, date):
        time = self.get_var_columns("Time")[0]

        if not isinstance(date[0], np.datetime64):
            date_start = np.datetime64(date[0])
            date_end = np.datetime64(date[1]) + np.timedelta64(1, 'D') - np.timedelta64(1, 's')
        self['data'] = self['data'][(self['data'][time] >= date_start) & (self['data'][time] <= date_end)]

    def update_session_state(self):
        st.session_state.devices_list[self.name] = self.__dict__
    
    def time_delta(self):
        time = self.plot_variables['Time']['Column']
        return self.data[time].max() - self.data[time].min()



@st.cache_data
def plot_time_series_help(df, plot_variables):
    
    selection = alt.selection_multi(fields=[plot_variables['Categorical']['Column']], bind='legend')
    
    # Plot Time Series
    c = alt.Chart(df).mark_circle().encode(
                x=alt.X(plot_variables['Time']['Column']+':T', title=plot_variables['Time']['Name'], axis=alt.Axis(grid=True), scale=alt.Scale(type='utc')),
                y=alt.Y(plot_variables['Numerical']['Column']+':Q', title=plot_variables['Numerical']['Name'] + f"  [{plot_variables['Numerical']['Unit']}]" ),
                color = alt.Color(plot_variables['Categorical']['Column']+':N', title=plot_variables['Categorical']['Name']),
                opacity=alt.condition(selection, alt.value(1), alt.value(0.005)),
                tooltip=[
                    {'field': plot_variables['Time']['Column'], 'type': 'temporal', 'title': 'Date', 'format': '%a, %e %b %Y'},
                    {'field': plot_variables['Time']['Column'], 'type': 'temporal', 'title': 'Time', "timeUnit": "utchoursminutes", 'format': '%H:%M'},
                    {'field': plot_variables['Numerical']['Column'], 'type': 'quantitative', 'title': plot_variables['Numerical']['Name'], 'format': '.0f'},
                    {'field': plot_variables['Categorical']['Column'], 'type': 'nominal', 'title': plot_variables['Categorical']['Name']},
                ]
                ).add_selection(
                    selection
                )
    
    # c = alt.Chart(df).mark_area(
    #         #color="lightblue",
    #         #interpolate='step-after',
    #         #line=True
    #     ).encode(
    #         x=alt.X(plot_variables['Time']['Column']+':T', title=plot_variables['Time']['Name'], axis=alt.Axis(grid=True), scale=alt.Scale(type='utc')),
    #         y=alt.Y(plot_variables['Numerical']['Column']+':Q', title=plot_variables['Numerical']['Name'] + f"  [{plot_variables['Numerical']['Unit']}]" ),
    #         color = alt.Color(plot_variables['Categorical']['Column']+':N', title=plot_variables['Categorical']['Name']),
    #     )
    
    return c

def resampler_help(x):
    """
    Compute the mode but avoid errors for empty slices or all-NaN slices by checking them.
    If slice is empty or all NaN, returns NaN.
    """
    mode = x.mode()
    return mode.iloc[0] if not mode.empty else None

@st.cache_data
def resampler(df, plot_variables):
    
    original_df = df.copy()
    
    time = plot_variables['Time']['Column']
    
    # Convert 'time' column to datetime and set it as the index
    df[time] = pd.to_datetime(df[time])

    # Calculate the total time period of the dataframe
    time_period = df[time].max() - df[time].min()
    
    df.set_index(time, inplace=True)
    
    # Identifying numerical and non-numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    non_numerical_cols = df.select_dtypes(include=['object']).columns
    
    # Calculate the difference in months
    months_diff = relativedelta(df.index.max(), df.index.min()).months + \
                  12 * relativedelta(df.index.max(), df.index.min()).years
    
    # Apply functions based on the time period
    if months_diff >= 1:  # More than 1 month
        resample_str = f'{months_diff}H'
        # Resampling non-numerical columns using the updated robust mode calculation method
        resampled_non_numerical_df = df[non_numerical_cols].resample(resample_str).apply(resampler_help)
        # Resampling numerical columns using mean
        resampled_numerical_df = df[numerical_cols].resample(resample_str).mean()
    else:
        return original_df
    
    # Merging the resampled data
    resampled_df = pd.concat([resampled_numerical_df, resampled_non_numerical_df], axis=1)
    resampled_df = resampled_df.reset_index().rename(columns={'index': time})
    
    #st.dataframe(resampled_df)  # Streamlit function to display dataframe
    
    return resampled_df

# Cannot use st.cache_data
def plot_time_series(device): 
    df = device.data.copy()
    tab1, tab2, tab3 = st.tabs(["Chart 📈", "Table 📄", "Export 📁"])
    
    resample_dataframe = resampler(df, device.plot_variables)
    
    c = plot_time_series_help(resample_dataframe, device.plot_variables)
    
    tab1.altair_chart(c, use_container_width=True)
    
    with tab2:
        columns = []
        columns_categorical = []
        columns_numerical = []
        columns_time = []
        for var in device.settings['Time'].values():
            columns.append(var)
            columns_time.append(var['column'])
                
        for var in device.settings['Numerical'].values():
            columns.append(var)
            columns_numerical.append(var['column'])
                
        for var in device.settings['Categorical'].values():
            columns.append(var)
            columns_categorical.append(var['column'])
                
        column_config = {}
        column_order = []
        
        for var in columns:
            column_config[var['column']] = var['variable_name']
            column_order.append(var['column'])
        
        st.dataframe(df, use_container_width=True, column_order=column_order, column_config=column_config, height=300)

    csv = df.to_csv(index=True).encode('utf-8')
    tab3.download_button("Press to Download all data", csv, "data.csv", "text/csv", key='download-csv')

@st.cache_data
def percentage(df):
    value_counts = df['state'].value_counts()
    percentages = (value_counts / len(df)) * 100
    percentages = percentages.round(1)
    
    return percentages

@st.cache_data
def shift_percentage(df):
    
    df['date'] = pd.to_datetime(df['ts'].dt.date)
    df['time'] = pd.to_datetime(df['ts'].dt.time, format='%H:%M:%S')
    
    # Filter the DataFrame based on weekday range
    start_weekday = time.strptime(st.session_state['shift']['work_days'][0], "%A").tm_wday
    end_weekday = time.strptime(st.session_state['shift']['work_days'][1], "%A").tm_wday
    in_df = df[(df['date'].dt.dayofweek >= start_weekday) & (df['date'].dt.dayofweek <= end_weekday)]
    
    # Filter the DataFrame based on time range
    start_time = st.session_state['shift']['start_time']
    end_time = st.session_state['shift']['end_time']
    in_df = in_df[(in_df['time'].dt.time >= start_time) & (in_df['time'].dt.time <= end_time)]
    
    out_df = df[~df[['ts']].isin(in_df[['ts']]).all(axis=1)]
    
    return percentage(in_df), percentage(out_df)
    
@st.cache_data
def plot_horizontal_bar_chart(device):
    top_labels = device["state_name"]
    colors =  ['rgba(38, 24, 74, 0.8)', 'rgba(122, 120, 168, 0.8)', 'rgba(190, 192, 213, 1)']
    
    index_order = []
    for i in range(len(device["state_name"])):
        index_order.append(device["state_name"][i])
    
    # Calculate the percentage for total time
    total = percentage(device["data"])
    total = total.reindex(index=index_order)
    
    # Calculate the percentage for inshift time
    in_shift, out_shift = shift_percentage(device["data"]) 
    in_shift = in_shift.reindex(index=index_order)
    out_shift = out_shift.reindex(index=index_order) 

    x_data = [total.tolist(), in_shift.tolist(), out_shift.tolist()]
    
    y_data = ['Total', 'In Shift', 'Out of Shift']
    
    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode='stack',
        #paper_bgcolor='rgb(248, 248, 255)',
        #plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=0, r=10, t=90, b=10),
        showlegend=False,
        title='Machine State',
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(dict(xref='paper', yref='y',
                                x=0.14, y=yd,
                                xanchor='right',
                                text=str(yd),
                                font=dict(family='Arial', size=14,
                                        color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=str(xd[0]) + '%',
                                font=dict(family='Arial', size=14,
                                        color='rgb(248, 248, 255)'),
                                showarrow=False))
        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.1,
                                    text=top_labels[0],
                                    font=dict(family='Arial', size=14,
                                            color='rgb(67, 67, 67)'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
                # labeling the rest of percentages for each bar (x_axis)
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i]/2), y=yd,
                                        text=str(xd[i]) + '%',
                                        font=dict(family='Arial', size=14,
                                                color='rgb(248, 248, 255)'),
                                        showarrow=False))
                # labeling the Likert scale
                if yd == y_data[-1]:
                    annotations.append(dict(xref='x', yref='paper',
                                            x=space + (xd[i]/2), y=1.1,
                                            text=top_labels[i],
                                            font=dict(family='Arial', size=14,
                                                    color='rgb(67, 67, 67)'),
                                            showarrow=False))
                space += xd[i]

    fig.update_layout(annotations=annotations)
    
    # Plot!
    st.plotly_chart(fig, use_container_width=True)  
 
@st.cache_data 
def help2(df):
    ##### Week days Selector #####

    temp_df = df.copy()
    
    #temp_df = temp_df.dropna(subset=['value'])
    temp_df['day_of_week'] = temp_df['ts'].dt.day_name()
    temp_df['hour'] = temp_df['ts'].dt.hour
    
    temp_df = temp_df.groupby(['day_of_week','hour'],sort=False,as_index=False).agg(lambda x:x.value_counts().index[0])

    week_hours = {day: {hour: None for hour in range(24)} for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

    week_hours = {}

    # Iterate through the rows of the DataFrame
    for _, row in temp_df.iterrows():
        # Get the values for the week and hour columns
        week = row['day_of_week']
        hour = row['hour']
        value = row['state_id']
        
        # If the week is not already in the dictionary, add it
        if week not in week_hours:
            week_hours[week] = {}
        
        # Set the value for the specified week and hour
        week_hours[week][hour] = value

    # Convert the dictionary to a list of lists
    data = [[v for v in row.values()] for row in week_hours.values()]
        
    days_of_week = list(week_hours.keys())  
    
    # Create a DataFrame
    df1 = pd.DataFrame({'Weekday': days_of_week, 'Value': data})    
    
    # Define the correct order of weekdays
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Convert the 'Weekday' column to categorical with the specified order
    df1['Weekday'] = pd.Categorical(df1['Weekday'], categories=weekday_order, ordered=True)

    # Sort the DataFrame by the 'Weekday' column
    df_sorted = df1.sort_values('Weekday')
    
    return df_sorted['Weekday'].to_list(), df_sorted['Value'].to_list()   
    
@st.cache_data 
def help1(device):
    ##### Week days Selector #####

    temp_df = device.data.copy()

    time = device.plot_variables['Time']['Column']
    categorical = device.plot_variables['Categorical']['Column']
    

    #temp_df = temp_df.dropna(subset=['value'])
    temp_df['day_of_week'] = temp_df[time].dt.day_name()
    temp_df['hour'] = temp_df[time].dt.hour

    temp_df = temp_df[[categorical, 'day_of_week', 'hour']]

    temp_df = temp_df.groupby(['day_of_week','hour'],sort=False,as_index=False, dropna=False).mean(numeric_only=False)

    temp_df['value'] = temp_df['value'].astype('int')

    week_hours = {day: {hour: None for hour in range(24)} for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

    week_hours = {}

    # Iterate through the rows of the DataFrame
    for _, row in temp_df.iterrows():
        # Get the values for the week and hour columns
        week = row['day_of_week']
        hour = row['hour']
        value = row['value']
        
        # If the week is not already in the dictionary, add it
        if week not in week_hours:
            week_hours[week] = {}
        
        # Set the value for the specified week and hour
        week_hours[week][hour] = value

    # Convert the dictionary to a list of lists
    data = [[v for v in row.values()] for row in week_hours.values()] 
        
    days_of_week = list(week_hours.keys())  
    
    # Create a DataFrame
    df1 = pd.DataFrame({'Weekday': days_of_week, 'Value': data})    
    
    # Define the correct order of weekdays
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Convert the 'Weekday' column to categorical with the specified order
    df1['Weekday'] = pd.Categorical(df1['Weekday'], categories=weekday_order, ordered=True)

    # Sort the DataFrame by the 'Weekday' column
    df_sorted = df1.sort_values('Weekday')
    
    
    return df_sorted['Weekday'].to_list(), df_sorted['Value'].to_list()  
    
@st.cache_data 
def help3(device):
    ##### Week days Selector #####

    temp_df = device["data"].copy()

    #temp_df = temp_df.dropna(subset=['value'])
    temp_df['day_of_week'] = temp_df['ts'].dt.day_name()
    temp_df['hour'] = temp_df['ts'].dt.hour
    temp_df['day'] = temp_df['ts'].dt.day

    temp_df = temp_df[['value', 'day_of_week', 'hour', 'day']]
    
    temp_df = temp_df.groupby(['day_of_week','hour', 'day'],sort=False,as_index=False, dropna=False).mean(numeric_only=False)

    temp_df['value'] = pd.to_numeric(temp_df['value'], errors='coerce').astype('Int64')
    temp_df['state_id'] = temp_df['value'].apply(value_dependent_func, args=(device,))
    
    temp_df = temp_df.groupby(['day_of_week','hour'],sort=False,as_index=False, dropna=False).mean(numeric_only=False)
    temp_df = temp_df[['day_of_week', 'hour', 'state_id']]

    week_hours = {day: {hour: None for hour in range(24)} for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

    week_hours = {}

    # Iterate through the rows of the DataFrame
    for _, row in temp_df.iterrows():
        # Get the values for the week and hour columns
        week = row['day_of_week']
        hour = row['hour']
        value = row['state_id']
        
        # If the week is not already in the dictionary, add it
        if week not in week_hours:
            week_hours[week] = {}
        
        # Set the value for the specified week and hour
        week_hours[week][hour] = value

    # Convert the dictionary to a list of lists
    data = [[v for v in row.values()] for row in week_hours.values()] 
        
    days_of_week = list(week_hours.keys())  
    
    # Create a DataFrame
    df1 = pd.DataFrame({'Weekday': days_of_week, 'Value': data})    
    
    # Define the correct order of weekdays
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Convert the 'Weekday' column to categorical with the specified order
    df1['Weekday'] = pd.Categorical(df1['Weekday'], categories=weekday_order, ordered=True)

    # Sort the DataFrame by the 'Weekday' column
    df_sorted = df1.sort_values('Weekday')
    
    return df_sorted['Weekday'].to_list(), df_sorted['Value'].to_list()      
    
@st.cache_data    
def value_dependent_func(x, device):
    
    if pd.isna(x):
        return pd.NA
    
    for i in range(len(device["state_name"])):
        if x >= device["state_min"][i] and x<=device["state_max"][i]:  
            return i   
 
@st.cache_data      
def plot_pie_chart(device):    
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate the percentage for total time
    total  = percentage(device["data"])
    
    # Calculate the percentage for inshift time
    in_shift, out_shift = shift_percentage(device["data"])  
    
    index_order = []
    for i in range(len(device["state_name"])):
        index_order.append(device["state_name"][i])
    
    total = total.reindex(index=index_order)
    in_shift = in_shift.reindex(index=index_order)
    out_shift = out_shift.reindex(index=index_order)
    
    fig = px.pie(device["data"], names=index_order, title='Total time')
    col1.plotly_chart(fig, use_container_width=True)
    fig = px.pie(in_shift, values='state', names=index_order, title='In shift time')
    col2.plotly_chart(fig, use_container_width=True)
    fig = px.pie(out_shift, values='state', names=index_order, title='Outside shift time')
    col3.plotly_chart(fig, use_container_width=True)
    
    return

########## Sidebar ##########

def sidebar_memory(device_name):
    key = "date_range_picker_"+device_name
    
    # Create a temporary file to store the session state
    if key not in st.session_state:  
        if not os.path.exists('temp'): # 
            os.makedirs('temp')
        with open("temp/"+device_name+'.pkl', 'wb') as f:
            pickle.dump(st.session_state.devices_list[device_name], f)
        
    else:
        with open("temp/"+device_name+'.pkl', 'rb') as f:
            st.session_state.devices_list[device_name] = pickle.load(f)

def sidebar():
    # Device Selector
    device_name = st.sidebar.selectbox("Select a device", st.session_state.devices_list.keys(), key="select_device")
    
    sidebar_memory(device_name) # Creates a temporary file to store the session state

    device = Device(device_name) # Create a Device object

    time_column = device.get_var_columns("Time")[0] # Get the time column name
    time_name = device.get_var_names("Time")[0] # Get the time variable name
    device.add_plot_var("Time", time_name) # Add the time column to the variables dictionary

    with st.sidebar:
        default_end = device["data"][time_column].min() + timedelta(days=30)
        if default_end > device["data"][time_column].max():
            default_end = device["data"][time_column].max()
        date = date_range_picker("Select a date range",
                                default_start=device["data"][time_column].min(),
                                default_end=default_end,
                                min_date=device['date_time_info']['start'].date(),
                                max_date=device['date_time_info']['end'].date(),
                                key="date_range_picker_"+device_name)
        
        device.date_filter(date) # Filter the DataFrame based on the selected date range
    
        numerical_name = st.selectbox("Select a numerical variable", device.get_var_names("Numerical"), key="select_numerical")        
        device.add_plot_var("Numerical", numerical_name)
        
        categorical_name = st.selectbox("Select a categorical variable", device.get_var_names("Categorical"), key="select_categorical")
        device.add_plot_var("Categorical", categorical_name)
        
        # filter_data = st.toggle("Filter data", key="filter_data")
        # if filter_data==True:
        #     df = dataframe_explorer(device.data)
        #     st.dataframe(df)
            
        #     # check id dataframe is empty
        #     if df.empty:
        #         st.error("No data available for the selected date range")
        #         st.stop()
        #     else:
        #         device.data = df
    
    device.update_session_state() # Update the session state with the new data
    
    return device

########## Sidebar ##########

########## Shift Activity ##########

@st.cache_data
def aggregate(df, plot_variables):
    
    # Count the occurrences of each categorical value
    categorical = df[plot_variables['Categorical']['Column']].value_counts().reset_index()
    categorical.columns = ['Categorical', 'Count']
    # Calculate the percentage for each material
    total_count = categorical['Count'].sum()
    categorical['Percentage'] = round((categorical['Count'] / total_count) * 100, 1)/100

    return categorical

@st.cache_data
def filter_time(df, plot_variables, key):

    start_time = st.session_state['shift']['start_time']
    end_time = st.session_state['shift']['end_time']
    time = plot_variables['Time']['Column']
    
    if key == "in_shift":
        df = df[(df[time].dt.time >= start_time) & (df[time].dt.time < end_time)]
    elif key == "out_shift":
        df = df[(df[time].dt.time < start_time) | (df[time].dt.time >= end_time)]
    
    return df 

@st.cache_data
def plot_shift_activity_pie_chart(df_1, df_2, df_3, plot_variables):
    col1, col2, col3 = st.columns(3)
    
    c = alt.Chart(df_1, title="Overall Distribution").mark_arc().encode(
        theta=alt.Theta("Count", type="quantitative", stack=True),
        color=alt.Color(field="Categorical", type="nominal", title=plot_variables['Categorical']['Name']),
        tooltip=[
            {'field': 'Categorical', 'type': 'nominal', 'title': plot_variables['Categorical']['Name']},
            {'field': 'Percentage', 'type': 'quantitative', 'format': '.1%'}
        ]
    )
    col1.altair_chart(c, use_container_width=True)
    
    c = alt.Chart(df_2, title="Active Shift Distribution").mark_arc().encode(
        theta=alt.Theta("Count", type="quantitative"),
        color=alt.Color(field="Categorical", type="nominal", title=plot_variables['Categorical']['Name']),
        tooltip=[
            {'field': 'Categorical', 'type': 'nominal'},
            {'field': 'Percentage', 'type': 'quantitative', 'format': '.1%'}
        ]
    )
    col2.altair_chart(c, use_container_width=True)
    
    
    c = alt.Chart(df_3, title="Off-Duty Distribution").mark_arc().encode(
        theta=alt.Theta("Count", type="quantitative"),
        color=alt.Color(field="Categorical", type="nominal", title=plot_variables['Categorical']['Name']),
        tooltip=[
            {'field': 'Categorical', 'type': 'nominal'},
            {'field': 'Percentage', 'type': 'quantitative', 'format': '.1%'}
        ]
    )
    col3.altair_chart(c, use_container_width=True)
    
    return    
    
@st.cache_data    
def plot_shift_activity_normalized_bar_chart(df_1, df_2, df_3, plot_variables):
    
    df_1['Time Period'] = "Overall"
    df_2['Time Period'] = "Active Shift"
    df_3['Time Period'] = "Off-Duty"  
    stacked_df = pd.concat([df_1, df_2, df_3], ignore_index=True)

    c = alt.Chart(stacked_df, title="Shift Activity").mark_bar(cornerRadius=15).encode(
        x=alt.X('Percentage:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
        y=alt.Y('Time Period', sort=['All time', 'In shift time', 'Outside of shift time'], ),
        color=alt.Color(field="Categorical", type="nominal", title=plot_variables['Categorical']['Name']),
        tooltip=[
            {'field': 'Categorical', 'type': 'nominal', 'title': plot_variables['Categorical']['Name']},
            {'field': 'Percentage', 'type': 'quantitative', 'format': '.1%'}
        ]
    ).properties(
        height=300
    ).configure_legend(
        symbolType='circle'
    )
    
    st.altair_chart(c, use_container_width=True)
    
    return
    
@st.cache_data    
def plot_shift_activity_horizontal_grouped_bar_chart(df_1, df_2, df_3, plot_variables):
    
    df_1['Time Period'] = "Overall"
    df_2['Time Period'] = "Active Shift"
    df_3['Time Period'] = "Off-Duty"     
    stacked_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
    
    col1, col2 = st.columns([5,1.5])
    
    c = alt.Chart(stacked_df, title="Shift Activity").mark_bar(cornerRadius=15).encode(
        x=alt.X('Percentage:Q', axis=alt.Axis(format='%')),
        y = alt.Y('Categorical:N', title=''),
        color= alt.Color('Categorical:N'),
        row = alt.Row('Time Period:N', title=''),
        tooltip=[
            {'field': 'Categorical', 'type': 'nominal', 'title': plot_variables['Categorical']['Name']},
            {'field': 'Percentage', 'type': 'quantitative', 'format': '.1%'}
        ]
    ).properties(
        height=100,
        width='container',
    ).configure_legend(
        symbolType='circle'
    )
    
    col1.altair_chart(c, use_container_width=True)
    
    
    return    
   
@st.cache_data   
def filter_weekday(df, plot_variables, work_days, key):
    """
    Filters the dataframe based on the provided start and end weekdays.
    """
    # Define a mapping of weekday strings to corresponding integers as per datetime module
    weekday_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    time = plot_variables['Time']['Column']
    
    # Convert input weekdays to corresponding integer values
    start_weekday, end_weekday = work_days[0], work_days[1]
    start_weekday, end_weekday = weekday_mapping[start_weekday], weekday_mapping[end_weekday]
    
    # Extract the weekday from the 'time' column
    df['weekday'] = df[time].dt.weekday
    
    if key == "in_shift":
        # Filter the dataframe based on the specified weekday range
        filtered_df = df[(df['weekday'] >= start_weekday) & (df['weekday'] <= end_weekday)].copy()
    elif key == "out_shift":
        filtered_df = df[(df['weekday'] < start_weekday) | (df['weekday'] > end_weekday)].copy()
    
    # Drop the 'weekday' column used for filtering, to return a dataframe similar to input
    filtered_df.drop(columns=['weekday'], inplace=True)
    
    return filtered_df
   
@st.cache_data    
def data_plot_shift_activity(df, plot_variables): 
    
    time = plot_variables['Time']['Column']
    
    df_1 = aggregate(df.copy(), plot_variables)
    
    df_2 = filter_weekday(df.copy(), plot_variables, st.session_state['shift']['work_days'], "in_shift")
    df_2 = filter_time(df_2, plot_variables, "in_shift")
    df_2 = aggregate(df_2, plot_variables)
    
    df_3_days = filter_weekday(df.copy(), plot_variables, st.session_state['shift']['work_days'], "out_shift")
    df_3_time = filter_time(df.copy(), plot_variables, "out_shift")
    combined_df = pd.concat([df_3_days, df_3_time])
    combined_df = combined_df.drop_duplicates(keep='first')
    df_3 = aggregate(combined_df, plot_variables)

    
    return df_1, df_2, df_3
           
# Cannot use st.cache_data            
def plot_shift_activity(df, plot_variables):
    
    tab1, tab2, tab3 = st.tabs(["Pie Chart ", "Normalized Bar Chart ", "Grouped Bar Chart"])
    
    df_1, df_2, df_3 = data_plot_shift_activity(df.copy(), plot_variables.copy())
    
    with tab1:
        plot_shift_activity_pie_chart(df_1, df_2, df_3, plot_variables.copy())
    with tab2:
        plot_shift_activity_normalized_bar_chart(df_1, df_2, df_3, plot_variables.copy())
    with tab3:
        plot_shift_activity_horizontal_grouped_bar_chart(df_1, df_2, df_3, plot_variables.copy())
    # with tab4:
    #     plot_shift_activity_vertical_grouped_bar_chart(copy.deepcopy(device))
    
    return
          
########## Shift Activity ##########     
  
def check_setup():   
    
    if "devices_list" not in st.session_state:
        st.error("Please go to Setup page and select a device.")
        want_to_contribute = st.button("Take me to Setup page")
        if want_to_contribute:
            switch_page("Setup")
        st.stop()      
    elif "setup_completed" not in st.session_state:
        st.error("Please go to Setup page and select a device.")
        want_to_contribute = st.button("Take me to Setup page")
        if want_to_contribute:
            switch_page("Setup")
        st.stop()
    else:
        if st.session_state.setup_completed == False:
            st.error("Please go to Setup page and select a device.")
            want_to_contribute = st.button("Take me to Setup page")
            if want_to_contribute:
                switch_page("Setup")
            st.stop()
            
    return 
    
def the_form(df1, settings):
    
    df = df1.copy()
    
    #with st.form(key='my_form'):    
    with st.form('true'):
        #df = filter_dataframe(device.data)
        

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)


        columns = []
        columns_categorical = []
        columns_numerical = []
        columns_time = []
        for var in settings['Time'].values():
            columns.append(var)
            columns_time.append(var['column'])
                
        for var in settings['Numerical'].values():
            columns.append(var)
            columns_numerical.append(var['column'])
                
        for var in settings['Categorical'].values():
            columns.append(var)
            columns_categorical.append(var['column'])
                
        column_config = {}
        column_order = []
        var_names = [item['variable_name'] for item in columns]

        to_filter_columns = st.multiselect("Filter data on:", var_names)
        for var_name in to_filter_columns:
            column = columns[var_names.index(var_name)]['column']
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
                
            # Categorical
            if column in columns_categorical:
                user_cat_input = right.multiselect(
                    f"Values for {column}:",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
                    
            # Numerical
            elif column in columns_numerical:    
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}:",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
                
            # Datetime
            elif column in columns_time:
                user_date_input = right.date_input(
                    f"Values for {column}:",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}:",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
            
        st.form_submit_button("Filter")
        
            
        for var in columns:
            column_config[var['column']] = var['variable_name']
            column_order.append(var['column'])
        
        view_data = st.checkbox("View data")
        if view_data:    
            st.dataframe(df, use_container_width=True, column_order=column_order, column_config=column_config, height=300)
        # check id dataframe is empty
        if df.empty:
            st.error("No data available for the selected date range")
            st.stop()
        else:
            df1=df
    
    return df1
    
def visualization():
    st.set_page_config(page_title="Visualization", page_icon="📊", layout="wide")
    start_time = tm.time()
    start_time_all = tm.time()
    add_logo("images/IST-1 - 01.png")
    st.write("# Visualization")
    
    st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>', unsafe_allow_html=True)
    
    check_setup()
    end_time = tm.time()
    execution_time = {}
    execution_time['Setup'] = end_time-start_time
    
    start_time = tm.time()
    device = sidebar()
    end_time = tm.time()
    execution_time['Sidebar'] = end_time-start_time
    
    start_time = tm.time()
    plots = st.multiselect("Visualization mode:", ["Time Series", "Shift Analysis", "Temporal Analysis", "Interruption Analysis"], 
                               default=[], key=f"plots_{device['name']}")
    
    filter_data = st.toggle("Filter by value", key="filter_data", value=False)    
    if filter_data==True:
        device.data=the_form(device.data.copy(), device.settings.copy())
        
    for plot in plots:
        if plot == "Time Series":
            with st.spinner("Loading " + plot + " ..."):
                st.write("#### Time Series")
                plot_time_series(device)
                
        if plot == "Shift Analysis":
            with st.spinner("Loading " + plot + " ..."):
                st.write("#### Shift Activity")
                plot_shift_activity(device.data.copy(), device.plot_variables.copy())
        
        if plot == "Temporal Analysis":
            with st.spinner("Loading " + plot + " ..."):
                st.write("#### Temporal Analysis")
                heat_map(copy.deepcopy(device))
                
        if plot == "Interruption Analysis":
            #st.write("#### Stop Activity")
            with st.spinner("Loading " + plot + " ..."):
                plot_stop_activity(copy.deepcopy(device))
    
    end_time = tm.time()
    execution_time['Plots'] = end_time-start_time
    end_time_all = tm.time()
    execution_time['Overall'] = end_time_all-start_time_all
    
    st.subheader("Execution time (in miliseconds)")
    for execution in execution_time:
        st.write(execution, ":", execution_time[execution]*1000)         


    
# Run the Streamlit app
if __name__ == "__main__":
    # # Create a GraphvizOutput object that specifies the output format and file
    # graphviz = GraphvizOutput(output_file='visualization_graph.pdf', output_type='pdf', grouping=True)
    
    # # Use PyCallGraph context manager to capture the function calls
    # with PyCallGraph(output=graphviz):

    #visualization()
    
    
    # profiler = Profiler()
    # profiler.start()
    
    visualization()
    
    # profiler.stop()

    # profiler.print()
    
    
