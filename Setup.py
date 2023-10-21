import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from datetime import *
import pandas as pd
import datetime
import numpy as np
from streamlit_extras.app_logo import add_logo
import matplotlib.pyplot as plt
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
import pickle
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from scipy.stats import mode
import sys
#import pandas_profiling
import ydata_profiling
from st_pages import Page, add_page_title, show_pages
from collections import Counter


# DATA = sys.argv[1]

# if DATA == "pedra":
#     FILE_NAME = "data_pedreira.xlsx"
#     SETUP_CONFIG_FORM = True
# elif DATA == "compal":
#     SETUP_CONFIG_FORM = True
#     FILE_NAME = "data_compal.csv"



class Device:
    def __init__(self, name):
        self = st.session_state[name]
    
    def __str__(self):
        return self.name
    
    def numerical_columns(self):
        return self.settings["Numerical"]
    
    def categorical_columns(self):
        return self.settings["Categorical"]
    
    def time_columns(self):
        return self.settings["Time"] 

def filter_df(device_settings, df, name):
    
    column_names = []

    for outer_key, outer_value in device_settings.items():
        for inner_key, inner_value in outer_value.items():
            if "column" in inner_value:
                column_names.append(inner_value["column"])
    
    df = df[column_names]
    
    return df

#@st.cache_data
def get_data(dict, settings, resampling):
    
    devices_list = {}
    for name, df in dict.items():
        df = filter_df(settings[name], df, name)
        df = format_data(df, settings[name])
        df = clean_data(df, settings[name], resampling)
        devices_list[name] = {'name':name, 'data': df, 'settings': settings[name], 'resampling': resampling}
        
        
    return devices_list
  
def unit_factor_converter(var):
        
    unit_factors = {
            'Nano': 1e-9,
            'Micro': 1e-6,
            'Milli': 1e-3,
            'Base': 1,
            'Kilo': 1e3,
            'Mega': 1e6,
            'Giga': 1e9
        }
    
    unit_factor = unit_factors[var["unit_order"]]
    
    return unit_factor   
    
@st.cache_data()
def format_data(df1, device_settings):

    df = df1.copy()

    if "Time" in device_settings:
        for var_type, var in device_settings["Time"].items():
            column_name = var["column"]
            unit_factor = unit_factor_converter(var)
            df[column_name] = pd.to_datetime(df[column_name]*unit_factor, unit='s')
            df[column_name] = df[column_name].astype('datetime64[s]')
    
    if "Categorical" in device_settings:
        for var_type, var in device_settings["Categorical"].items():
            column_name = var["column"]
            if df[column_name].dtype != 'object':
                # Convert numeric values to string, and NaN to an empty string
                df[column_name] = df[column_name].apply(lambda x: None if pd.isna(x) else str(int(x)))
            #df[column_name] = df[column_name].astype(str)
            df[column_name] = df[column_name].apply(lambda x: str(x) if x == x else 'None')
            
            
    if "Numerical" in device_settings:
        for var_type, var in device_settings["Numerical"].items():
            column_name = var["column"]
            unit_factor = unit_factor_converter(var)
            df[column_name] = df[column_name]*unit_factor
            df[column_name] = df[column_name].astype(float)
            

    return df

def mode_function(x):
    # Use scipy's mode function which returns mode and count
    m = mode(x)
    # Return the mode (m[0]) - if there are multiple modes, this returns the first one
    return m[0][0]

# Fast mode function for use with .agg()
def fast_mode(series):
    # Use Counter to find most common item in series
    mode_count = Counter(series)
    # Find the most common item
    most_common = mode_count.most_common(1)
    # Return the most common item's value
    return most_common[0][0] if most_common else None

@st.cache_data(show_spinner=False)
def resample_dataframe(df, frequency_unit, frequency_value, resampling_method, device_settings):
    # Mapping between resampling methods and corresponding functions
    method_mapping = {
        'Mean': 'mean',
        'Median': 'median',
        'Max': 'max',
        'Min': 'min',
        'Sum': 'sum',
        'First': 'first',
        'Last': 'last'
    }

    # Mapping between frequency units and corresponding pandas time offsets
    unit_mapping = {
        'Seconds': 'S',
        'Minutes': 'T',
        'Hours': 'H',
        'Days': 'D',
        'Weeks': 'W',
        'Months': 'M',
        'Years': 'Y'
    }

    # Convert frequency unit and value to corresponding time offset
    time_offset = str(frequency_value) + unit_mapping[frequency_unit]
    
    time = None
    # Initialize empty aggregation dictionary
    agg_dict = {}
    
    if "Time" in device_settings:
        for var_type, var in device_settings["Time"].items():
            time = var["column"]
    
    if "Categorical" in device_settings:
        for var_type, var in device_settings["Categorical"].items():
            categorical = var["column"]
            df[categorical] = df[categorical].astype('category')
            agg_dict[categorical] = fast_mode
            #agg_dict[categorical] = 'first'
            #agg_dict[categorical] = mode_function
    
    if "Numerical" in device_settings:
        for var_type, var in device_settings["Numerical"].items():
            numerical = var["column"]
            agg_dict[numerical] = method_mapping[resampling_method]
    
    
    # Perform the resampling
    if time:
        resampled_df = df.resample(time_offset, on=time).agg(agg_dict)
    else:
        st.error("Error: No time column found!")
        st.stop()

    return resampled_df

def clean_data(df, device_settings, resampling):     
    
    frequency_unit = resampling["frequency_unit"]
    frequency_value = resampling["frequency_value"]
    resampling_method = resampling["resampling_method"]
    interpolation_limit = resampling["interpolation_limit"]
    
    with st.spinner("Resampling data..."):
        df = resample_dataframe(df, frequency_unit, frequency_value, resampling_method, device_settings)
    
    if "Time" in device_settings:
        for var_type, var in device_settings["Time"].items():
            time = var["column"]
        
    if time and time not in df.index.names:
        df.set_index(time, inplace=True)
    
    df = df.interpolate(method='time', limit=interpolation_limit)
    
    # For numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = df[numerical_cols].interpolate(method='time', limit=interpolation_limit)

    # For string/categorical columns
    string_cols = df.select_dtypes(include=['object']).columns
    df[string_cols] = df[string_cols].fillna(method='ffill', limit=interpolation_limit)
    #df[string_cols] = df[string_cols].fillna(method='bfill', limit=interpolation_limit)
    
    df[time] = df.index
    df=df.reset_index(drop=True)

    return df

def shift_details():
    
    col1, col2 = st.columns(2)
    
    col1_1, col1_2 = col1.columns(2)
    
    
    if st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_compal.csv":
        step = datetime.timedelta(minutes=1)
        default_start = datetime.time(0, 0)
        default_end = datetime.time(23, 59)
    else:
        step = datetime.timedelta(minutes=1)
        default_start = datetime.time(9, 0)
        default_end = datetime.time(18, 0)

    
    shift_start = col1_1.time_input("Shift Start", default_start, step=datetime.timedelta(minutes=1))
    shift_end = col1_2.time_input("Shift End", default_end, step=datetime.timedelta(minutes=1))
    
    work_days = col2.select_slider('Select work days',
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                value=['Monday', 'Friday'])
    
    return {
    "start_time": shift_start,
    "end_time": shift_end,
    "work_days": work_days
    }
    
@st.cache_data
def date_time_combiner(start_date, start_time, end_date, end_time):
    start_date_time = datetime.datetime.combine(start_date, start_time)
    end_date_time = datetime.datetime.combine(end_date, end_time)
  
    return start_date_time, end_date_time

def date_time_selector(device_name, device):
    col1, col2 = st.columns(2)
    col1_1, col1_2 = col1.columns(2)
    col2_1, col2_2 = col2.columns(2)
    
    if "Time" in device["settings"]:
        for var_type, var in device["settings"]["Time"].items():
            time = var["column"]
    
    start_date = col1_1.date_input("Start Date:", device['data'][time].min().date(), min_value=device['data'][time].min().date(), max_value=device['data'][time].max().date(), key='start_date_'+device_name)
    start_time = col1_2.time_input("Start Time:", datetime.time(0,0,0), key='start_time_'+device_name)
    end_date = col2_1.date_input("End Date:", device['data'][time].max().date(), min_value=device['data'][time].min().date(), max_value=device['data'][time].max().date(), key='end_date_'+device_name)
    end_time = col2_2.time_input("End Time:", datetime.time(23,59,59), key='end_time_'+device_name)
    
    start_date_time, end_date_time = date_time_combiner(start_date, start_time, end_date, end_time)
    
    device['date_time_info'] = {'start': start_date_time, 'end': end_date_time}
    
    # Ensure that the end date is after the start date
    if start_date_time > end_date_time:
        st.error("Error: End date must fall after start date.")
        
    # Filter the data based on the start and end date and time
    device["data"] = device["data"][(device["data"][time] >= start_date_time) & (device["data"][time] <= end_date_time)]
    
    return device

def state_personalized(device_name, device, col1, col_extra):
    col2, col3, col4 = col_extra.columns([1, 1, 1])     
    n = col1.number_input("Number of States", min_value=1, max_value=10, value=3, step=1, key='n_states_'+device_name)
    state_info = {'min': [], 'max': [], 'name': [], 'n_states': n, 'from_data': False}
    
    if "Numerical" in device["settings"]:
        for var_type, var in device["settings"]["Numerical"].items():
            numerical = var["column"]
    
    for i in range(n):
        if n ==3:
            state_info["min"].append(col2.number_input('Interval 1 minimum:', value=0, step=1, key='min_1_'+device_name))
            state_info["max"].append(col3.number_input('Interval 1 maximum:', value=49, step=1, key='max_1_'+device_name))
            state_info["name"].append(col4.text_input('Interval 1 name:', value='Off', key='name_1_'+device_name))
            
            state_info["min"].append(col2.number_input('Interval 2 minimum:', value=50, step=1, key='min_2_'+device_name))
            state_info["max"].append(col3.number_input('Interval 2 maximum:', value=600, step=1, key='max_2_'+device_name))
            state_info["name"].append(col4.text_input('Interval 2 name:', value='Idle', key='name_2_'+device_name))
            
            state_info["min"].append(col2.number_input('Interval 3 minimum:', value=601, step=1, key='min_3_'+device_name))
            state_info["max"].append(col3.number_input('Interval 3 maximum:', value=int(device["data"][numerical].max()), step=1, key='max_3_'+device_name))
            state_info["name"].append(col4.text_input('Interval 3 name:', value='Operating', key='name_3_'+device_name))
            break
            
        else:     
            state_info["min"].append(col2.number_input('Interval ' + str(i+1) + ' minimum:', min_value=int(device["data"][numerical].min()), max_value=int(device["data"][numerical].max()), value=1, step=1, key='min_'+str(i+1)+'_'+device_name))
            state_info["max"].append(col3.number_input('Interval ' + str(i+1) + ' maximum:', min_value=int(device["data"][numerical].min()), max_value=int(device["data"][numerical].max()), value=1, step=1, key='max_'+str(i+1)+'_'+device_name))
            state_info["name"].append(col4.text_input('Interval ' + str(i+1) + ' name:', value='', key='name_'+str(i+1)+'_'+device_name))
    
    if state_info["name"][n-1] != "":
        state_info['working_states'] = col1.multiselect("Select working states", options=state_info["name"], default=state_info["name"][n-1], key='working_states_'+device_name)
    
    device["state_info"] = state_info

    return device
          
def clustering(device):
    df = device["data"].copy()
    
    if "Numerical" in device["settings"]:
        for var_type, var in device["settings"]["Numerical"].items():
            numerical = var["column"]

    # Round the values
    df['rounded_value'] = df[numerical].round(0)
    
    # Initialize new columns
    df['state'] = pd.NA
    df['state_id'] = pd.NA
    
    # Create NumPy arrays for faster computation
    state_info_name = np.array(device['state_info']["name"])
    state_info_min = np.array(device['state_info']["min"])
    state_info_max = np.array(device['state_info']["max"])
    
    for i in range(len(state_info_name)):
        mask = (df['rounded_value'] >= state_info_min[i]) & (df['rounded_value'] <= state_info_max[i])
        df.loc[mask, 'state'] = state_info_name[i]
        df.loc[mask, 'state_id'] = i
    
    # Drop the temporary column
    df.drop('rounded_value', axis=1, inplace=True)
    
    if "Categorical" in device["settings"]:
        device["settings"]["Categorical"]["State"] = {"column": "state", "variable_type": "Categorical", "variable_name": "State"}
        #device["settings"]["Categorical"]["State ID"] = {"column": "state_id", "variable_type": "Categorical", "variable_name": "State ID"}


    return df

@st.cache_data
def outlier_remover_help(df, state, window_size, threshold_duration):

    # Create a mask to identify outlier 'Operating' states
    outlier_mask = (
        (df['state_id'] == state) &
        (df['state_id'].rolling(2 * window_size + 1).apply(lambda x: x[:window_size].ne(state).all() and x[-window_size:].ne(state).all())) &
        ((df['timestamp'].shift(-window_size) - df['timestamp']) < pd.Timedelta(seconds=threshold_duration)) &
        ((df['timestamp'] - df['timestamp'].shift(window_size)) < pd.Timedelta(seconds=threshold_duration))
    )

    # Replace the outlier 'Operating' states with NaN
    df.loc[outlier_mask, 'state'] = np.nan
    
    return df
    
def outlier_remover(device_name, device):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    outliers = col1.checkbox("Remove outliers", value=False, key='outliers_'+device_name)
    if outliers:
        states = col2.multiselect("Select states to remove", options=device["name"], default=device["name"][2])
        window = col3.number_input("Window size", min_value=1, max_value=10, value=5, step=1, help="Define the number of seconds before and after the outlier to be removed")
        threshold = col4.number_input("Threshold", min_value=1, max_value=10, value=2, step=1, help="Define the threshold for the outlier detection")
        for state in states:
            st.write(device["name"].index(state))
            device["data"] = outlier_remover_help(device["data"], device["name"].index(state), window, threshold)
            
    return device["data"]
    
def grouping_from_data(device_name, device, col1_original, col_extra):
    
    col1, col2 = col_extra.columns([4,2])
    
    categorical_var = list(device['settings']['Categorical'].keys())
    
    state = col1.selectbox("Select column with the state value:", categorical_var, index=0, key='categorical_var_'+device_name, help="This column will be renamed to 'State'")
    
    
    col2.write("")
    col2.write("")
    reason_toggle = col2.toggle("Reason column", value=True, key='reason_'+device_name)
    
    if reason_toggle ==True:
        reason_name = col1.selectbox("Select column with the reason value:", categorical_var, index=2, key='reason_var_'+device_name)
        
    state_column = device['settings']['Categorical'][state]['column']
    #reason_column = device['settings']['Categorical'][reason]['column']
    
    device['data'].rename(columns={state_column: 'state'}, inplace=True)
    
    if "Categorical" in device["settings"]:
        new = {"column": "state", "variable_type": "Categorical", "variable_name": "State"}
        device["settings"]["Categorical"] = {'State': new, **device["settings"]["Categorical"]}
        #device["settings"]["Categorical"]["State"] = 
    
    del device["settings"]["Categorical"][state]
    
    working_states = col1_original.multiselect("Select working states", options=device['data']['state'].unique(), default=device['data']['state'].unique()[2], key='working_states_'+device_name)
    
    state_info = {'from_data': True, 'working_states': working_states, 'reason': reason_name}
    device["state_info"] = state_info
    
    return device    
    
def grouping(device_name, device):
    st.subheader("State Analysis")
    col1, col_extra = st.columns([1, 3])
    index = 0
    
    if st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_compal.csv":
        index=1
    
    mode = col1.radio("State Analysis Mode:", ("Personalized", "From data", 'Off'), index=index, key='mode_'+device_name)
    
    if mode == 'Personalized':
        device = state_personalized(device_name, device, col1, col_extra)
        device["data"] = clustering(device)
        
        
    elif mode == 'Automatic':
        a=1
        #device["data"] = automatic(device["data"], n)
        
    elif mode == 'From data':
        device = grouping_from_data(device_name, device, col1, col_extra)
    
    
    #device["data"] = outlier_remover(device_name, device)
    
    
    return device

def stop_personalized(device_name, device, col1, col_extra):
    col2, col3, col4 = col_extra.columns([1, 1, 1])       
    n_stop = col1.number_input("Number of Stop states", min_value=1, max_value=10, value=2, step=1, key='n_stop_'+device_name)
    stop_info = {'min': [], 'max': [], 'name': [], 'n_stop': n_stop, 'from_data': False}
    
    for i in range(n_stop):
        if n_stop == 2:
            stop_info["min"].append(col2.number_input('Interval 1 minimum:', min_value=0, value=0, step=1, key='stop_min_1_'+device_name))
            stop_info["max"].append(col3.number_input('Interval 1 maximum:', min_value=0, value=10, step=1, key='stop_max_1_'+device_name))
            stop_info["name"].append(col4.text_input('Interval 1 name:', value='Micro Stop', key='stop_name_1_'+device_name))
            
            stop_info["min"].append(col2.number_input('Interval 2 minimum:', min_value=0, value=11, step=1, key='stop_min_2_'+device_name))
            stop_info["max"].append(col3.number_input('Interval 2 maximum:', min_value=0, value=20, step=1, key='stop_max_2_'+device_name))
            stop_info["name"].append(col4.text_input('Interval 2 name:', value='Setup', key='stop_name_2_'+device_name))
            break
            
        else:     
            stop_info["min"].append(col2.number_input('Interval ' + str(i+1) + ' minimum:', min_value=0, value=1, step=1, key='stop_min_'+str(i+1)+'_'+device_name))
            stop_info["max"].append(col3.number_input('Interval ' + str(i+1) + ' maximum:', min_value=0, value=1, step=1, key='stop_max_'+str(i+1)+'_'+device_name))
            stop_info["name"].append(col4.text_input('Interval ' + str(i+1) + ' name:', value='', key='stop_name_'+str(i+1)+'_'+device_name))
    
    device["stop_info"] = stop_info
    
    return device

def stop_from_data(device_name, device, col1, col_extra):
    
    col1, col2 = col_extra.columns([4,2])
    notes_toggle = False
    
    categorical_var = list(device['settings']['Categorical'].keys())
    
    stop = col1.selectbox("Select column with the stop value:", categorical_var, index=2, key='categorical_var_stop_'+device_name)
    
    stop_info = {'from_data': True, 'stop': stop}
    
    col2.write("")
    col2.write("")
    cause_toggle = col2.toggle("Cause column", value=True, key='cause_'+device_name)
    
    if cause_toggle ==True:
        cause_name = col1.selectbox("Select column with the cause value:", categorical_var, index=3, key='cause_var_'+device_name)
        stop_info['cause'] = cause_name
        
        col2.subheader("")
        col2.write("")
        notes_toggle = col2.toggle("Notes column", value=True, key='notes_'+device_name)
    
    if notes_toggle ==True:
        notes_name = col1.selectbox("Select column with the notes value:", categorical_var, index=4, key='notes_var_'+device_name)
        stop_info['notes'] = notes_name
    
    device["stop_info"] = stop_info
    
    return device

def stops(device_name, device):
    st.subheader("Stop Analysis")
    #col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col1, col_extra = st.columns([1, 3])
    
    if st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_compal.csv":
        index=1
    elif st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_pedreira.xlsx":
        index=0
    else:
        index=0
    
    stop_mode = col1.radio("Stop Analysis Mode:", ("Personalized", "From data", 'Off'), index=index, key='stop_mode_'+device_name)
    
    if stop_mode == 'Personalized':
        device = stop_personalized(device_name, device, col1, col_extra)
        
    elif stop_mode == 'From data':
        device = stop_from_data(device_name, device, col1, col_extra)
    
    
    return device

def identify_column_type(column, unique_ratio_threshold=0.001):
    """
    Identify the type of a single column in the DataFrame.
    
    Parameters:
    - column: Pandas DataFrame with a single column
    - unique_ratio_threshold: The threshold for the unique-to-total ratio for categorizing as 'Categorical'
    
    Returns:
    A string representing the identified type ('Time', 'Categorical', 'Numerical', 'Empty', or 'Constant (Value)').
    """
    
    # Check for empty column
    if column.isna().all():
        return 'Empty'
    
    # Check for constant value
    unique_values = column.unique()
    if len(unique_values) == 1:
        return f'Constant ({unique_values[0]})'
    
    # Check for time type
    if is_datetime64_any_dtype(column):
        return 'Time'
    
    # Check for categorical type
    unique_count = len(unique_values)
    total_count = len(column)
    unique_ratio = unique_count / total_count
    if unique_ratio < unique_ratio_threshold:
        return 'Categorical'
    
    if unique_ratio >= 0.99:
        return 'Time'
    
    # Check for continuous type
    if is_numeric_dtype(column):
        if (column < 0).any():
            return 'Negatives'
        else:
            return 'Numerical'
    
    # Default to 'Unknown'
    return 'Unknown'

def data_editor(device, key):
    
    settings= {}
    list_columns = list(device.columns)
    list_time_units = ["Unix Timestamp", "Seconds", "Minutes", "Hours", "Days", "Weeks", "Months", "Years"]
    list_value_units = ["Watts", "Units", ]
    list_unit_order = ["Nano", "Micro", "Milli", "Base", "Kilo", "Mega", "Giga"]
    
    if st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_pedreira.xlsx":
        if key[-1]=="0":
            column_index = 0
            variable_name = "Time"
            unit_order_index = 2
        elif key[-1]=="1":
            column_index = 18
            variable_name = "Power"
            unit_order_index = 2
        elif key[-1]=="2":
            column_index = 25
            variable_name = "Material"
        
            
    elif st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_compal.csv":
        if key[-1]=="0":
            column_index = 7
            variable_name = "Time"
            unit_order_index = 2
        elif key[-1]=="1":
            column_index = 6
            variable_name = "Power"
            unit_order_index = 3
        elif key[-1]=="2":
            column_index = 1
            variable_name = "Status"
        elif key[-1]=="3":  
            column_index = 2
            variable_name = "Production"
        elif key[-1]=="4":  
            column_index = 3
            variable_name = "Classification"
        elif key[-1]=="5":  
            column_index = 4
            variable_name = "Cause"
        elif key[-1]=="6":  
            column_index = 5
            variable_name = "Notes"
        
        
            
    else:
        column_index = 0
        variable_name = ""
        unit_order_index = 0
            
    
    col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 0.8])
    
    settings['column'] = col2.selectbox("Column from data:", list_columns, key='column_'+key,
                                        help="The column should contain the values of the variable to be simulated.",
                                        index = column_index)
    settings['variable_type'] = identify_column_type(device[settings['column']]) 
    types_of_data = ["Time", "Categorical", "Numerical"]
    settings['variable_type'] = col3.selectbox("Variable Type:", types_of_data, index=types_of_data.index(settings['variable_type']), key='variable_type_'+key)
    
    if settings['variable_type'] == "Empty":
        st.warning(":arrow_up: The selected column is empty!")
        st.stop()
    elif settings['variable_type'][:8] == "Constant":
        # Find the positions of the opening and closing parentheses
        start_pos = settings['variable_type'].find("(") + 1  # Add 1 to exclude the opening parenthesis itself
        end_pos = settings['variable_type'].find(")")
        st.warning(":arrow_up: The selected column has always the same value: " + settings['variable_type'][start_pos:end_pos])
        st.stop()
    elif settings['variable_type'] == "Negatives":
        st.warning(":arrow_up: The selected column has negative values. Negative values will be ignored.")
        settings['variable_type'] = "Numerical"
    settings['variable_name'] = col4.text_input("Name of variable:", key='variable_name_'+key, value=variable_name if variable_name != "" else settings['column'])
    if settings['variable_type'] != "Categorical":
        settings['unit'] = col5.selectbox(
                            "Unit of variable:",
                            list_time_units if settings['variable_type'] == "Time" else list_value_units,
                            key='unit_'+key)
    
        settings['unit_order'] = col6.selectbox("Order of magnitude:", list_unit_order, index=unit_order_index, key='unit_order_'+key)
        col7.write("")
        col7.write("")
        settings['change'] = col7.toggle("Change Unit", value=False, key='change_'+key)
        if settings['change']:
            new_list_time_units = [x for x in list_time_units if x != settings['unit']]
            new_list_value_units = [x for x in list_value_units if x != settings['unit']]
            # settings['new_unit'] = col4.selectbox(
            #                     "Select the new unit of the variable",
            #                     new_list_time_units if settings['variable_type'] == "Time" else new_list_value_units,
            #                     key='new_unit_'+key)
            new_list_unit_order = [x for x in list_unit_order if x != settings['unit_order']]
            settings['new_unit_order'] = col6.selectbox("New order of magnitude:", new_list_unit_order, key='new_unit_order_'+key)
    else:
        settings['change'] = False    
     
    
    return settings  

def change_unit(temp):
    
    return temp

def upload_file():
    
    dict = {}
    st.session_state['SETUP_CONFIG_FORM'] = False
    st.session_state['FILE_NAME'] = ""
    error = False
    index = 2
    
    with st.sidebar:
        if st.experimental_get_query_params() != {}:
            params = st.experimental_get_query_params()
            if "dataset" in params:
                if params["dataset"][0] == "pedreira":
                    index = 0
                elif params["dataset"][0] == "compal":
                    index = 1
        
        if len(sys.argv) >= 2:
            if sys.argv[1] == "pedra":
                index = 0
                st.experimental_set_query_params(dataset=["pedreira"])
            elif sys.argv[1] == "compal":
                index = 1

            
        upload_mode = st.radio("Choose Dataset", ("Example Pedreira", "Example Compal", "Upload your data"), index=index, key='upload_file_radio')
    
    if upload_mode == "Upload your data":
        st.experimental_set_query_params(dataset=["upload"])
    
    elif upload_mode == "Example Pedreira":
        st.experimental_set_query_params(dataset=["pedreira"])
        
    elif upload_mode == "Example Compal":
        st.experimental_set_query_params(dataset=["compal"])


    if upload_mode == "Upload your data":
        list_files = st.file_uploader("Upload your data", 
                                      type=["csv", "xlsx"], 
                                      key='upload_file', 
                                      accept_multiple_files=True,
                                      help="HELP")
        
        
        if list_files == []:
            st.info("Please upload your data.")
            st.stop()
            
        if list_files != []:
            with st.status("Uploading file...", expanded=True) as status:    
                status.update(label="Upload complete!", state="running", expanded=True)
                
                for file in list_files:
                    if file.name.split(".")[1] == "csv":
                        df = pd.read_csv(file)
                        dict[file.name.split(".")[0]] = df
                    elif file.name.split(".")[1] == "xlsx":
                        temp_dict = pd.read_excel(file, sheet_name=None, engine='openpyxl')
                        dict.update(temp_dict)
                        
    if upload_mode == "Example Pedreira" or upload_mode == "Example Compal":    
        with st.status("Loading data...", expanded=True) as status:
            if upload_mode == "Example Pedreira":
                st.session_state['FILE_NAME'] = "data_pedreira.xlsx"
                st.session_state['SETUP_CONFIG_FORM'] = True
                file_name = "data/" + st.session_state['FILE_NAME'].split(".")[0] + ".pkl"
                with open(file_name, "rb") as f:
                    temp_dict = pickle.load(f)
                    dict.update(temp_dict)

            elif upload_mode == "Example Compal":
                st.session_state['FILE_NAME'] = "data_compal.csv"
                st.session_state['SETUP_CONFIG_FORM'] = True
                file_name = "data/" + st.session_state['FILE_NAME'].split(".")[0] + ".pkl"
                with open(file_name, "rb") as f:
                    df = pickle.load(f)
                    temp_dict = {file_name.split(".")[0]: df}
                    dict.update(temp_dict)

    
    
    
    
            
    
    # file_name = st.text_input('Enter the name of the CSV file to be used for the simulation:', FILE_NAME, help='The file should be located is the same directory as the main python file.')
    # dict = {}
    # error = False
    
    # if file_name != "":
    #     if file_name=="data_pedreira.xlsx":
    #         file_path = "data/merged_data_material.pkl"
    #         with open(file_path, "rb") as f:
    #             with st.status("Reading file...") as status:
    #                 st.write("Loading file...")
    #                 dict = pickle.load(f)
    #                 st.write("Cleaning data...")
                    
    #     else:
    #         with st.status("Reading file...") as status:
    #             try:
    #                 st.write("Loading file...")
    #                 if file_name.split(".")[1] == "csv":
    #                     df = pd.read_csv("data/" + file_name)
    #                     dict = {file_name.split(".")[0]: df}
    #                 elif file_name.split(".")[1] == "xlsx":
    #                     dict = pd.read_excel("data/" + file_name, sheet_name=None, engine='openpyxl')
    #                 st.write("Cleaning data...")
    #             except:
    #                 st.error("File not found! Go to Help tab for more information.")
    #                 error=True
    
    return dict, error, status                      

def resample_function(key):
    
    # col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 0.8])
    # resampling = col1.checkbox('Resampling', value=True, key='resampling_'+key)
    # col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 0.8])
    # if resampling:
    #     frequency_unit = col1.selectbox('Resampling frequency unit:', options=['Seconds', 'Minutes', 'Hours', 'Days', 'Weeks', 'Months', 'Years'], index=1, key='frequency_unit_'+key)
    #     frequency_value = col2.number_input('Resampling frequency value:', min_value=1, max_value=100, value=1, key='frequency_value_'+key)
    #     resampling_method = col3.selectbox('Resampling method', options=['Mean', 'Median', 'Max', 'Min', 'Sum', 'First', 'Last'], index=0, key='resampling_method_'+key)
    #     interpolation_limit = col4.number_input(f'Interpolation limit [{frequency_unit}]', min_value=0, value=25, key='interpolation_limit_'+key)


    st.write("")
    st.markdown("**Resampling Options:**")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 0.8])
    frequency_unit = col1.selectbox('Resampling frequency unit:', options=['Seconds', 'Minutes', 'Hours', 'Days', 'Weeks', 'Months', 'Years'], index=1, key='frequency_unit_'+key)
    frequency_value = col2.number_input('Resampling frequency value:', min_value=1, max_value=100, value=1, key='frequency_value_'+key)
    resampling_method = col3.selectbox('Resampling method', options=['Mean', 'Median', 'Max', 'Min', 'Sum', 'First', 'Last'], index=0, key='resampling_method_'+key, help="The resampling method is used to aggregate the data within the resampling frequency. This only applies to the numerical variables. The categorical variables are aggregated using the mode.")
    interpolation_limit = col4.number_input(f'Interpolation limit [{frequency_unit}]', min_value=0, value=25, key='interpolation_limit_'+key)

    resampling = {'frequency_unit': frequency_unit, 'frequency_value': frequency_value, 'resampling_method': resampling_method, 'interpolation_limit': interpolation_limit}

    return resampling

def options_file(dict, error):
    
    settings = {}
    if dict != {} and error == False:
        keys_to_change = []
        settings = {} # Initialize settings dictionary
        list_device_names = list(dict.keys())
        list_device_names.insert(0, "None")
        for device_name, device in dict.items():
            if device_name == list_device_names[1]:
                expanded = True
            else:
                expanded = False
            with st.expander(device_name, expanded=expanded):
                st.write()
                #device = data_editor(device)
                col1, col2 = st.columns([2, 1])
                if st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_compal.csv":
                    name = "Enchedora"      
                else:
                    name = device_name           
                                    
                # Rename device
                new_device_name = col1.text_input('Rename device:', name, key='new_device_name_'+device_name)
                if new_device_name == "":
                    st.warning("Device name cannot be empty!")
                    error=True
                keys_to_change.append((new_device_name, device_name))
                    
                # Copy settings from another device
                list_device = [x for x in list_device_names if x != device_name]
                settings_from = col2.selectbox("Copy settings from:", list_device, key='settings_from_'+device_name)
                col1, col2 = st.columns([3, 10])
                    
                # Select data type
                if settings_from == "None":
                    if st.session_state['SETUP_CONFIG_FORM'] == True and st.session_state['FILE_NAME'] == "data_compal.csv":
                        default = 7
                    else:
                        default = 3
                    n_variables = col1.number_input("Number of variables", min_value=2, max_value=10, value= default, step=1, key='n_variables_'+device_name)
                    for i in range(n_variables):
                        temp = data_editor(device, device_name+str(i))
                        if device_name not in settings: # Create new device key in settings
                            settings[device_name] = {}
                        if temp['variable_type'] not in settings[device_name]: # Create new variable_type key in settings
                            settings[device_name][temp['variable_type']] = {}
                        if temp['variable_type'] == "Time" and len(settings[device_name]['Time']) == 1: # Check if there is already a time variable
                            st.warning("Only one time variable is allowed! You already have a time variable named " + list(settings['Time'].keys())[0] + ".")    
                        else:
                            settings[device_name][temp['variable_type']][temp['variable_name']] = temp
                        if temp['change'] == True:
                            change_unit(temp)
                    resampling = resample_function(device_name)
                else:
                    settings[device_name] = {}
                    settings[device_name]["Settings from"] = settings_from

                    
        # Rename new keys and delete old keys
        for key in keys_to_change:
            dict[key[0]] = dict.pop(key[1], None)
            settings[key[0]] = settings.pop(key[1], None)
            
        # Copy settings from another device
        for key, value in settings.items():
            if "Settings from" in settings[key]:
                settings[key] = settings[value["Settings from"]]

                   
    else:
        st.write('No data found. Please load data in the Upload tab.')
        error=True
        
        #data_type = st.radio("Select the data type", ("Instantaneous power", "Average Power Consumption"), index=0)
    
    return dict, settings, resampling

def process_file():
    
    tab1, tab2, tab3, tab4 = st.tabs(["Upload :arrow_up:", "Options :gear:", "Data Preview :chart_with_downwards_trend:", "Help :question:"])
    error = False
    dict = {}
    with tab1:
        dict, error, status = upload_file()
    
    with tab2:
        dict, settings, resampling = options_file(dict, error)
        
    with tab3:
        if bool(dict)==True:
            list_device_names = list(dict.keys())
            data = st.multiselect("Select data to visualize", list_device_names, default=list_device_names[0])
            for device_name in data:
                with st.expander(device_name, expanded=True):
                    st.dataframe(dict[device_name])
                    more_info = st.toggle("Show more info", key='more_info'+device_name)
                    if more_info == True:
                        df_temp = dict[device_name]
                        pr = df_temp.profile_report()
                        st_profile_report(pr)
            
        else:
            st.write('No data found. Please load data in the Upload tab.')
            error=True
    
    with tab4:
        st.write("This is the third tab")

    if error:
        status.update(label="Error!", state="error", expanded=True)
        st.stop()
    
    if dict != {} and error == False:
        devices_list = get_data(dict, settings, resampling)
        status.update(label="File processed!", state="complete", expanded=False)
    
    return devices_list

def setup():
    st.set_page_config(page_title="Setup", page_icon=":gear:", layout="wide")
    add_logo("images/IST-1 - 01.png")
    st.write("# Setup") 
    
    st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>', unsafe_allow_html=True)
    
    show_pages(
    [
        Page("Setup.py", "Setup", ":gear:"),
        Page("pages/1_Visualization.py", "Visualization", ":bar_chart:"),
        Page("pages/2_Debug.py", "Debug", "ℹ️"),
        #Page("pages/3_Experiments.py", "Experiments", "🧑‍🔬️"),
    ]
    )
    
    
    if 'setup_completed' not in st.session_state:
        st.session_state.setup_completed = False
    else:
        st.session_state.setup_completed = False
    
    st.session_state.devices_list = process_file()
    st.session_state.shift = shift_details()
    
    for device_name, device in st.session_state.devices_list.items():
        if device_name == list(st.session_state.devices_list.keys())[0]:
            expanded = True
        else:
            expanded = False
        with st.expander(device_name, expanded=expanded):
            device = date_time_selector(device_name, device)
            device = grouping(device_name, device)
            device = stops(device_name, device)
            
    st.toast("Setup completed!", icon='🎉')
    st.session_state.setup_completed = True
         
# Run the Streamlit app
if __name__ == "__main__":
    # # Create a GraphvizOutput object that specifies the output format and file
    # graphviz = GraphvizOutput(output_file='setup_graph.pdf', output_type='pdf')
    
    # # Use PyCallGraph context manager to capture the function calls
    # with PyCallGraph(output=graphviz):
    setup()