import streamlit as st
import datetime
import pandas as pd
from datetime import timedelta
import altair as alt
import concurrent.futures
import time
import numpy as np
import json
import pickle

@st.cache_data
def plot_stop_activity_helper(df, stop_info, i, time, state, working_states):
    df = df.reset_index(drop=True) # Reset the index to avoid problems with the indices
    threshold_min = stop_info["min"][i]
    threshold_max = stop_info["max"][i]
    stop_name = stop_info["name"][i]

    # Initialize variables to count interruptions and store indices
    total_interruptions = 0
    interruption_indices = []
    list = []

    # Initialize variables to track current interruption
    interruption_start = None
    interruption_start_index = None
    interruption_end_index = None
    interruption_start_ts = None
    interruption_end_ts = None
    

    # Iterate over the column values with indices
    #for ts, index, value in enumerate(df[column_to_check]):
    for index, row in df.iterrows():
        value = row[state]
        if pd.isna(value):
            continue

        if value in working_states:
            if interruption_start is not None:
                #interruption_lengths.append(interruption_end_index - interruption_start_index + 1)
                interval = interruption_end_index - interruption_start_index
                if interval <= threshold_max and interval >= threshold_min:
                    total_interruptions += 1
                    interruption_indices.append((interval+1, interruption_start_ts, interruption_end_ts))
                    for i in range(interval+1): 
                        list.append((interruption_start_ts + timedelta(minutes=i), stop_name))
                interruption_start = None
                interruption_start_index = None
                interruption_end_index = None
        else:
            if interruption_start is None:
                interruption_start = value
                interruption_start_index = index
                interruption_start_ts = row[time]
            interruption_end_index = index
            interruption_end_ts = row[time]

    # Print the total number of interruptions and their lengths
    #total_interruptions
    #list
    #interruption_indices
    
    return list, interruption_indices

@st.cache_data
def interruption_converter(list_interruptions):
    
    df1 = pd.DataFrame(columns=['Datetime', 'Interruptions', 'Total'])
    # Iterate over each interruption
    for interruption in list_interruptions:
        duration = interruption[0]
        dt= interruption[1]
        
        # Round the datetime to the nearest hour
        hour_dt = dt.replace(minute=0, second=0)

        # Create a new DataFrame with the data to be appended
        new_data = pd.DataFrame({'Datetime': [hour_dt], 'Interruptions': [1], 'Total': [duration]})

        # Concatenate the new DataFrame with the existing DataFrame
        df1 = pd.concat([df1, new_data], ignore_index=True)
            
    if not df1.empty:
        # Group values by hour and sum the 'Value' column
        df1 = df1.groupby(pd.Grouper(key='Datetime', freq='H')).agg({'Interruptions': 'sum', 'Total': 'sum'})

    # Reset the index of the resulting DataFrame
    df1 = df1.reset_index()
    
    return df1
   
@st.cache_data    
def plot_ticks(df, settings, plot_variables):
    
    state_column = settings['Categorical']['State']['column']
    state_name = settings['Categorical']['State']['variable_name']
    time_column = plot_variables['Time']['Column']
    time_name = plot_variables['Time']['Name']
    numerical_column = plot_variables['Numerical']['Column']
    numerical_name = plot_variables['Numerical']['Name']
    
    df[state_column] = df[state_column].fillna("Unknown")
    dict = df.to_dict(orient='records')
    
    st.vega_lite_chart(dict,{
        "mark": "tick",
        "encoding": {
            "x": {"field": time_column, 
                  #"timeUnit": "utcdayshoursminutes",
                  "type": "temporal", 
                  "axis": {"title": time_name}},
            "y": {"field": state_column, 
                  "type": "nominal", 
                  "axis": {"title": state_name},
                  "sort": ["Unknown", "Operating", "Idle", "Off", "Interruption"]},
            "color": {
                "condition": [
                    {"test": "datum.state === 'Unknown'", "value": "black"},
                    {"test": "datum.state === 'Operating'", "value": "green"},
                    {"test": "datum.state === 'Micro Stop'", "value": "red"},
                    {"test": "datum.state === 'Setup'", "value": "red"}
                ]},
            "tooltip": [
                {"field": time_column, "type": "temporal", "title": "Date", "format": "%a, %e %b %Y"},
                {'field': time_column, 'type': 'temporal', 'timeUnit': 'hoursminutes' ,'title': 'Time'},
                {"field": state_column, "type": "nominal", "title": state_name},
                {"field": numerical_column, "type": "quantitative", "title": numerical_name}
            ]
        },
    }, use_container_width=True)
    
    return    

@st.cache_data 
def plot_bars_interruptions_time(df, device_stop_info):
    
    # Filter the 'state' column based on the values in vector 'x'
    interruptions_df = df[df['state'].isin(device_stop_info["name"])]
    
    interruptions_df['time'] = pd.to_datetime(interruptions_df['time'])
    interruptions_df['day'] = interruptions_df['time'].dt.date
    interruptions_df = interruptions_df.groupby(['state', 'day']).size().reset_index(name='time_units')
    
    
    return interruptions_df

@st.cache_data
def plot_line_total_interruptions(device_stop_info, list_interruptions):
    
    for i, stop in enumerate(device_stop_info["name"]):
        
        list_interruptions[i]["state"] = device_stop_info["name"][i]
        
        # Concatenate df1 and df2
        merged_df = pd.concat(list_interruptions, ignore_index=True) 
        
    merged_df = merged_df[merged_df['Interruptions'] != 0]    
    merged_df = merged_df.sort_values(by='Datetime')
    merged_df = merged_df.reset_index(drop=True)

    df = merged_df.copy()
    df['day'] = df['Datetime'].dt.date
    
    # Group data by State and Day and calculate sum of values
    grouped_df = df.groupby(['state', 'day']).agg({'Interruptions': 'sum'}).reset_index()
 
    
    return grouped_df

@st.cache_data
def plot_interruptions(data, stop_info, list_interruptions):
    
    time = plot_bars_interruptions_time(data.copy(), stop_info) # Bar plot of the total time of interruptions per day
    quantity = plot_line_total_interruptions(stop_info, list_interruptions) # Line plot of the total number of interruptions per day

    merged_df = pd.merge(time, quantity, on=['state', 'day'])
    # merged_df['day'] = pd.to_datetime(merged_df['day'])
    # merged_df = merged_df.set_index('day')
    # merged_df.resample('D').asfreq()
    
    base = alt.Chart(merged_df, title="Interruptions Time & Occurrences").encode(alt.X('day:T', title='Date', axis=alt.Axis(grid=True)), alt.Color('state:N', title='State'))
    
    bar = base.mark_bar().encode(alt.Y('time_units:Q', title='Time [units]'))
    
    line = base.mark_line(point=True, strokeDash = [4,4], strokeWidth=1).encode(alt.Y('Interruptions:Q', title='Occurrences'))
    
    chart = alt.layer(bar, line).resolve_scale(y='independent')

    #st.altair_chart(chart, use_container_width=True)
    
    return merged_df

@st.cache_data
def heat_map(df):
    df['day'] = pd.to_datetime(df['day'])
    unique_months_years = df['day'].dt.to_period('M').unique()
    
    for month_year in unique_months_years:
        # Filter the DataFrame to show only values from the desired month and year
        filtered_df = df[df['day'].dt.to_period('M') == month_year]
        
        # Generate a new DatetimeIndex with all the dates in the desired month and year
        start_date = pd.Timestamp(year=month_year.year, month=month_year.month, day=1)
        end_date = start_date + pd.offsets.MonthEnd()
        month_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        month_dates = pd.DataFrame({'day': month_dates, 'time_units1': 0, 'Interruptions1': 0})
        
        # Merge the DataFrames based on the 'day' column
        merged_df = month_dates.join(filtered_df.set_index('day'), on='day')
        #merged_df.fillna(0, inplace=True)
        merged_df['time_units'] = merged_df['time_units1'] + merged_df['time_units']
        merged_df['Interruptions'] = merged_df['Interruptions1'] + merged_df['Interruptions']
        merged_df.drop(columns=['time_units1', 'Interruptions1'], inplace=True)
        
        # Add columns with the week day and week number
        merged_df['week_day'] = merged_df['day'].dt.day_name()
        merged_df['week_number'] = merged_df['day'].dt.isocalendar().week
        merged_df.fillna(0, inplace=True)
    
        # Create a heatmap
        time_chart = alt.Chart(merged_df, title='Interruptions Total Time').mark_rect().encode(
            alt.X('week_day:N', title='', axis=alt.Axis(labelAngle=0, labelAlign='center', orient='top' ), sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            alt.Y('week_number:O', title='', axis=alt.Axis(labels=True), sort='ascending'),
            alt.Color('time_units:Q', title='Time [units]'),
            tooltip=[alt.Tooltip('day:T', title='Day'), alt.Tooltip('time_units', title='Time [units]')]
        ).properties(height=400)
        
        occurrences_chart = alt.Chart(merged_df, title='Interruptions Occurrences').mark_rect().encode(
            alt.X('week_day:N', title='', axis=alt.Axis(labelAngle=0, labelAlign='center', orient='top' ), sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            alt.Y('week_number:O', title='', axis=alt.Axis(labels=True), sort='ascending'),
            alt.Color('Interruptions:Q', title='Occurrences'),
            tooltip=[alt.Tooltip('day:T', title='Day'), alt.Tooltip('Interruptions', title='Interruptions')]
        ).properties(height=400)
        
        col1, col2 = st.columns(2)
        
        col1.altair_chart(time_chart, use_container_width=True)
        col2.altair_chart(occurrences_chart, use_container_width=True)
    
    return

@st.cache_data
def plot_interruptions_month_heatmap(df):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import july

    df = df.drop(columns=['state'])
    df = df.set_index('day')
    grouped_df = df.groupby(df.index).agg({'time_units': 'sum', 'Interruptions': 'sum'}).reset_index()
    
    heat_map(grouped_df)

@st.cache_data
def plot_interruptions_week_heatmap(stop_info, list_interruptions):
    
    for i, stop in enumerate(stop_info["name"]):
        
        list_interruptions[i]["state"] = stop_info["name"][i]
        
        # Concatenate df1 and df2
        df = pd.concat(list_interruptions, ignore_index=True)
        
    df = df[df['Interruptions'] != 0]
    df = df.drop(columns=['state'])
    df = df.rename(columns={'Interruptions': 'occurrences', 'Total': 'time'})
    df['hour'] = df['Datetime'].dt.hour
    df['week_day'] = df['Datetime'].dt.day_name()

    df = df.groupby(['hour', 'week_day']).agg({'occurrences': 'sum', 'time': 'sum'}).reset_index()
    
    # List of days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Create a list of dictionaries containing data for each day
    data = []
    for day in days_of_week:
        day_data = [{'week_day': day, 'hour': hour, 'value': 0} for hour in range(24)]
        data.extend(day_data)
    # Creating the DataFrame
    complete_df = pd.DataFrame(data)
    df = pd.merge(complete_df, df, on=['week_day', 'hour'], how='left')
    df.fillna(0, inplace=True)
    df.drop(columns=['value'], inplace=True)    
    
    time_chart = alt.Chart(df, title='Interruptions Total Time').mark_rect().encode(
            alt.X('hour:O', title='', axis=alt.Axis(labelAngle=0)),
            alt.Y('week_day:N', title='', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            alt.Color('time:Q', title='Time [units]'),
            tooltip=[alt.Tooltip('hour:O', title='Hour'), alt.Tooltip('week_day:N', title='Day'), alt.Tooltip('occurrences', title='Occurrences'), alt.Tooltip('time', title='Time [units]')]
    ).properties(height=400)
    
    occurrences_chart = alt.Chart(df, title='Interruptions Occurrences').mark_rect().encode(
            alt.X('hour:O', title='', axis=alt.Axis(labelAngle=0)),
            alt.Y('week_day:N', title='', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            alt.Color('occurrences:Q', title='Occurrences'),
            tooltip=[alt.Tooltip('hour:O', title='Hour'), alt.Tooltip('week_day:N', title='Day'), alt.Tooltip('occurrences', title='Occurrences'), alt.Tooltip('time', title='Time [units]')]
    ).properties(height=400)
    
    col1, col2 = st.columns(2)
    col1.altair_chart(time_chart, use_container_width=True)
    col2.altair_chart(occurrences_chart, use_container_width=True)

@st.cache_data
def stops_from_data(plot_variables, state_info, data, settings):
    """
    Record the changes in specified columns and return the records in a list of dictionaries.

    Parameters:
        data (pd.DataFrame): The input data.

    Returns:
        list: A list of dictionaries containing the changes information.
    """
    
    time = plot_variables["Time"]["Column"]
    working_states = state_info["working_states"]
    data.reset_index(drop=True, inplace=True)
    data.sort_values(by=time, inplace=True, ascending=True)
    
    # Specify columns to track
    
    columns_of_interest = [info["column"] for info in settings['Categorical'].values()]
    
    # Initialize variables
    records = []
    current_record = {}
    # Manually create the first record
    current_record['start_time'] = data.loc[0, time]
    current_record.update(data.loc[0, columns_of_interest].to_dict())

    # Find rows where changes occur starting from the second row
    change_rows = (data[columns_of_interest] != data[columns_of_interest].shift()).any(axis=1)
    # Adjusting to start from the second row for change detection
    change_indices = data.index[change_rows].tolist()[1:]

    # Include the last index for capturing the last segment
    change_indices.append(len(data))

    # Iterate through change indices
    for idx in change_indices:
        # Record the end time of the previous segment
        current_record['end_time'] = data.loc[idx-1, time] if idx != 0 else data.loc[idx, time]
        # Calculate the duration in minutes
        current_record['duration'] = ((current_record['end_time'] - current_record['start_time']).total_seconds() / 60) + 1 
        # Save the record if duration is not zero
        if current_record['duration'] != 0:
            records.append(current_record.copy())
        # Update values if not at the end of the DataFrame
        if idx < len(data):
            previous_values = data.loc[idx, columns_of_interest].to_dict()
            # Start a new record
            current_record = {'start_time': data.loc[idx, time]}
            current_record.update(previous_values)

    # Initialize dictionaries
    working = []
    stops = []

    # Separate records into 'working' and 'stops' based on the 'state' value
    for record in records:
        if record['state'] in working_states:
            working.append(record)
        else:
            stops.append(record)
    
    return working, stops, data

@st.cache_data
def plot_tick_2_help(df, settings, plot_variables, name, x_axis, state_info):
    
    state_column = settings['Categorical'][name]['column']
    state_name = name
    time_column = plot_variables['Time']['Column']
    time_name = plot_variables['Time']['Name']
    numerical_column = plot_variables['Numerical']['Column']
    numerical_name = plot_variables['Numerical']['Name']
    working_states = state_info["working_states"]
    
    df = df.replace('None', 'Unknown')
    
    dict = df.to_dict(orient='records')
    
    # Create a condition string for all working states
    working_states_condition = " || ".join([f"datum.{state_column} === '{state}'" for state in working_states])
    
    if name == 'State':
        color = "blue"
    else:
        color = "red"
        
    tooltip = [
        {"field": time_column, "type": "temporal", "title": "Date", "timeUnit": "utcyearmonthdate" , "format": "%a, %e %b %Y"},
        {"field": time_column, "type": "temporal", "title": "Time", "timeUnit": "utchoursminutes" , "format": "%H:%M:%S"},
        {"field": numerical_column, "type": "quantitative", "title": numerical_name},
    ]
    
    for key in settings['Categorical']:
        tooltip.append({"field": settings['Categorical'][key]['column'], "type": "nominal", "title": settings['Categorical'][key]['variable_name']})
    
    #st.write(time_column, state_column)
    
    st.vega_lite_chart(dict,{
        "mark": "tick",
        "encoding": {
            "x": {"field": time_column, 
                "type": "temporal", 
                "scale": {"type": "utc"},
                "axis": {"title": time_name, "labels": True} if x_axis==True else {"title": None, "labels": False} },
            "y": {"field": state_column, 
                "type": "nominal", 
                "axis": {"title": state_name, "labelPadding": 10, "grid": True},
                "sort": ["Em Funcionamento", "Unknown", "Operating", "Idle", "Off", "Interruption", "Parado", "Setup", "Falha"]
                },
            "color": {
                "condition": [
                    {"test": working_states_condition, "value": "green"},
                    {"test": f"datum.{state_column} === 'Unknown'", "value": "grey"},
                ],
                "value": color  # Default color for all other states
                },
            "opacity": {
                "condition": [
                    {"test": f"datum.{state_column} === 'Unknown'", "value": 1},
                    # additional conditions can go here
                ],
                "value": 1  # Default opacity for all other states
            },
            "tooltip": tooltip,
        },
    }, use_container_width=True)
    
    return

@st.cache_data
def plot_tick_2_help2(classification, cause, df, settings, plot_variables, state_info, stop_info):
    
    if classification == True:
        plot_tick_2_help(df, settings, plot_variables, 'State', False, state_info) 
        
        if cause == True:
            plot_tick_2_help(df, settings, plot_variables, stop_info['stop'], False, state_info)
            plot_tick_2_help(df, settings, plot_variables, stop_info['cause'], True, state_info)
            #plot_tick_2_help(df, settings, plot_variables, stop_info['notes'], True, state_info)
            
        else:
            plot_tick_2_help(df, settings, plot_variables, stop_info['stop'], True, state_info)
        
    else:
        plot_tick_2_help(df, settings, plot_variables, 'State', True, state_info)
        
    return

# Cannot use st.cache_data
def plot_ticks_2(df, settings, plot_variables, state_info, stop_info):
    
    # Apply the changes based on the conditions provided
    state = settings['Categorical']['State']['column']
    working_states = state_info["working_states"]
    stop_column = settings['Categorical'][stop_info['stop']]['column']
    cause = False
    
    if 'cause' in stop_info:
        cause_column = settings['Categorical'][stop_info['cause']]['column']
    
    for work_state in working_states:
        df.loc[df[state] == work_state, stop_column] = work_state
        if 'cause' in stop_info:
            df.loc[df[state] == work_state, cause_column] = work_state
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    classification = col1.toggle(f"Show {stop_info['stop']} for State", True, key="toggle_classification")
    if classification == True and 'cause' in stop_info:
        cause = col2.toggle(f"Show {stop_info['cause']} for {stop_info['stop']}", True, key="toggle_cause")
    
    plot_tick_2_help2(classification, cause, df, settings, plot_variables, state_info, stop_info)
    
    return df

@st.cache_data
def plot_pie_chart_stop_distribution(df, data, mode, state_info, settings, stops_dict):
    
    data_column = settings['Categorical'][data]['column']
    
    if mode == "Based on duration of the stops":
        # Given working states
        working_states = state_info["working_states"]

        # Filter the data to exclude the working states
        filtered_data = df[~df[data_column].isin(working_states)]

        # Count the occurrences of each state
        state_counts = filtered_data[data_column].value_counts().reset_index()
        state_counts.columns = ['state', 'count']

        # Calculate percentage
        state_counts['percentage'] = (state_counts['count'] / state_counts['count'].sum())
        
    elif mode == "Based on the number of occurrences of the stops":
        filtered_data = pd.DataFrame(stops_dict)
        # Count occurrences of each 'state'
        state_counts = filtered_data[data_column].value_counts().reset_index()
        # Rename columns
        state_counts.columns = ['state', 'count']
        # Calculate percentage
        state_counts['percentage'] = (state_counts['count'] / state_counts['count'].sum())

    # Create a pie chart
    c = alt.Chart(state_counts).mark_arc().encode(
        theta='count',
        color=alt.Color(field='state', type='nominal', title=data, legend=alt.Legend(orient="top")),
        tooltip=[
            {"field": "state", "type": "nominal", "title": data},
            {"field": "percentage", "type": "quantitative", "title": "Percentage", "format": ".1%"},
        ]
    ).properties(
        height=278,
    )

    
    return filtered_data, c

@st.cache_data
def plot_bar_chart_stop_distribution(df, stop_info, settings):
        
    if 'stop' in stop_info:
        data_column = settings['Categorical'][stop_info['stop']]['column']
    state_column = settings['Categorical']['State']['column']
        
    
    # Count the occurrences of each classification per state
    classification_count = df.groupby([state_column, data_column]).size().reset_index(name='count')

    # Calculate the total occurrences per state
    state_total = classification_count.groupby(state_column)['count'].sum().reset_index(name='total')

    # Merge the two dataframes on state to have count and total in the same dataframe
    merged_df = pd.merge(classification_count, state_total, on=state_column)

    # Calculate the percentage of each classification per state
    merged_df['percentage'] = (merged_df['count'] / merged_df['total'])
    
        
    # Create a bar chart
    chart = alt.Chart(merged_df).mark_bar(cornerRadius=15).encode(
        y=alt.Y('state:N', title=''),
        x=alt.X('percentage:Q', title=stop_info['stop'], axis=alt.Axis(format='%', orient='top', labels=False, ticks=False, grid=False, titlePadding=-3)),
        color = alt.Color(field=data_column, type='nominal', title=stop_info['stop'], legend=alt.Legend(orient="right", labelLimit=150), scale=alt.Scale(scheme='tableau20')),
        tooltip=[
            {"field": data_column, "type": "nominal", "title": stop_info['stop']},
            {"field": "percentage", "type": "quantitative", "title": "Percentage", "format": ".1%"},
        ]
    ).properties(
        height=390
    ).configure_legend(
        symbolType='circle'
    )
    
    return chart

# Cannot use st.cache_data
def stop_distribution(df, stop_info, state_info, settings, stops_dict):
    
    mode = st.selectbox("Select mode to calculate percentage:", ["Based on duration of the stops", "Based on the number of occurrences of the stops"], key="radio_stop_distribution_mode")
    
    col1, col2 = st.columns([1, 3])


    column = col1.selectbox("Information to plot in Pie Chart:", ["State", f"{stop_info['stop']}"], key="radio_stop_distribution", )
    
    # plot pie chart
    filtered_df, chart = plot_pie_chart_stop_distribution(df.copy(), column, mode, state_info, settings, stops_dict)
    col1.altair_chart(chart, use_container_width=True)
    
    # Plot bar chart
    chart = plot_bar_chart_stop_distribution(filtered_df, stop_info, settings)
    col2.altair_chart(chart, use_container_width=True)
    
    return

@st.cache_data
def plot_all_stops_histogram(df, scale):
    
    c = alt.Chart(df).mark_bar(cornerRadius=15).encode(
        x = alt.X(field='duration', 
                  type='quantitative', 
                  title='Duration [min]', 
                  axis=alt.Axis(labelAngle=0),
                  scale=alt.Scale(type=scale['x_axis'].lower(), domain=[df['duration'].min(), df['duration'].max()])
                  ),
        y = alt.Y('count()',
            title='Occurrences',
            scale=alt.Scale(type=scale['y_axis'].lower())
            )
    ).properties(
        title='Occurrences vs Duration',
    ).configure_legend(
        symbolType='circle'
    )
    
    st.altair_chart(c, use_container_width=True)
    
    return

@st.cache_data
def plot_filtered_stops_histogram(df, var_name, values, settings, scale):
    
    column_filter = settings['Categorical'][var_name]['column']
    
    # Filtering the DataFrame
    filtered_df = df[df[column_filter].isin(values)]
    
    selection = alt.selection_multi(fields=[column_filter], bind='legend')
    
    c = alt.Chart(filtered_df).mark_bar(cornerRadius=15).encode(
        x = alt.X(field='duration', 
                  type='quantitative', 
                  title='Duration [min]', 
                  axis=alt.Axis(labelAngle=0),
                  scale=alt.Scale(type=scale['x_axis'].lower(), domain=[filtered_df['duration'].min(), filtered_df['duration'].max()])
                  ),
        y = alt.Y('count()',
            title='Occurrences',
            scale=alt.Scale(type=scale['y_axis'].lower())
            ),
        color = alt.Color(field=column_filter, type='nominal', title=var_name),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.005)),
    ).add_selection(
        selection
    ).properties(
        title='Occurrences vs Duration',
    ).configure_legend(
        symbolType='circle'
    )
    
    st.altair_chart(c, use_container_width=True)
    
    return

# Cannot use st.cache_data
def stop_histogram(stop_info, settings, stops_dict, stop_type):
    
    df = pd.DataFrame(stops_dict)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    mode = col1.selectbox("Select mode of Histogram:", ["All Interruptions", "Filtered"], key="selectbox_stop_histogram_mode")
    
    scale = {}
    scale["x_axis"] = col2.selectbox("Select x-axis scale:", ["Linear", "Log"],  index=1, key="selectbox_stop_histogram_x_axis_scale")
    scale["y_axis"] = col3.selectbox("Select y-axis scale:", ["Linear", "Sqrt"],  index=1, key="selectbox_stop_histogram_y_axis_scale")
    
    col1, col2 = st.columns([1, 3])
    
    if stop_type == "From Data":
        options = ["State", f"{stop_info['stop']}"]
    elif stop_type == "Personalized":
        options = settings['Categorical'].keys()
    
    if mode == 'Filtered':
        filter_by = col1.radio("Filter by:", options, index=0,key="radio_stop_histogram_filter_by")
        
        column = settings['Categorical'][filter_by]['column']
        
        filter = col2.multiselect(f"Select {filter_by}:", df[column].unique(), default= df[column].unique() if filter_by == 'State' else df[column].unique()[0], key="multiselect_stop_histogram_filter")
        plot_filtered_stops_histogram(df, filter_by, filter, settings, scale)
    
    
    elif mode == 'All Interruptions':
        plot_all_stops_histogram(df, scale)
        
    return

@st.cache_data
def split_and_adjust_duration(df):
    """
    Splits rows in df where the machine stop spans multiple days and adjusts the duration accordingly.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing machine stop data.
    
    Returns:
    - new_df (pd.DataFrame): DataFrame with adjusted rows and duration.
    """
    # Ensure 'start_time' and 'end_time' are datetime objects
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    # Initialize an empty dataframe to store the new rows
    new_rows = []

    # Define a function to split rows when needed
    def split_rows(row):
        # Extract start_time and end_time
        start_time = row['start_time']
        end_time = row['end_time']
        
        # While end_time is on a different day than start_time
        while start_time.date() != end_time.date():
            # Calculate the end of the day for start_time
            end_of_day = pd.Timestamp(datetime.datetime(start_time.year, start_time.month, start_time.day, 23, 59, 59))
            
            # Create a new row spanning from start_time to end_of_day
            new_row = row.copy()
            new_row['start_time'] = start_time
            new_row['end_time'] = end_of_day
            new_row['duration'] = (end_of_day - start_time).total_seconds() / 60  # Convert to minutes
            new_rows.append(new_row)
            
            # Update start_time to be the start of the next day
            start_time = end_of_day + pd.Timedelta(seconds=1)
        
        # Create a new row for the remaining duration
        new_row = row.copy()
        new_row['start_time'] = start_time
        new_row['end_time'] = end_time
        new_row['duration'] = (end_time - start_time).total_seconds() / 60 + 1 # Convert to minutes 
        new_rows.append(new_row)

    # Apply the function to each row in the dataframe
    df.apply(split_rows, axis=1)

    # Convert the list of new rows into a dataframe
    new_df = pd.DataFrame(new_rows)
    
    # Round the 'duration' to 0 decimal places and convert to integer
    new_df['duration'] = new_df['duration'].round(0).astype(int)
    
    return new_df


def plot_stop_line_bar(grouped_df, column, mode, x_field, type):
    
    
    selection = alt.selection_multi(fields=[column], bind='legend')
    
    base = alt.Chart(grouped_df, title="Interruptions Duration Per Day").encode(
        x = alt.X(field=x_field, type='temporal', title='Date', axis=alt.Axis(grid=True)),
        color = alt.Color(field=column, type='nominal', title=mode),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.005)),
        )
    bar = base.mark_bar(cornerRadius=15).encode(
        y = alt.Y(field='total_duration', type='quantitative', title='Duration [min]'),
        )
    durations = alt.layer(bar).configure_legend(
        symbolType='circle'
    ).add_selection(
        selection
    )
    
    selection = alt.selection_multi(fields=[column], bind='legend')
    base = alt.Chart(grouped_df, title="Interruptions Occurrences Per Day").encode(
        x = alt.X(field=x_field, type='temporal', title='Date', axis=alt.Axis(grid=True)),
        color = alt.Color(field=column, type='nominal', title=mode),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.005)),
        )
    line = base.mark_line(point=True, strokeDash = [4,4], strokeWidth=1).encode(
        y = alt.Y(field='stop_count', type='quantitative', title='Occurrences'),
    )
    occorrences = alt.layer(line).add_selection(
        selection
    )
    
    if type == "Duration":
        st.altair_chart(durations, use_container_width=True)
    elif type == "Occurrences":
        st.altair_chart(durations, use_container_width=True)
    
    return

# Cannot use st.cache_data
def stop_line_bar(settings, stop_info, stops_dict, stop_type):
    
    df = pd.DataFrame(stops_dict)
    adjusted_df = split_and_adjust_duration(df)
    
    adjusted_df['Day'] = adjusted_df['start_time'].dt.date
    adjusted_df['Hour'] = adjusted_df['start_time'].dt.hour

    col1, col2 = st.columns([2, 2])
    
    if stop_type == "From Data":
        options = ["State", f"{stop_info['stop']}"]
    elif stop_type == "Personalized":
        options = settings['Categorical'].keys()
    
    type = col1.selectbox("Select mode:", ["Duration", "Occurrences"], key="selectbox_stop_line_bar_type")
    mode = col2.selectbox("Color by:", options, key="selectbox_stop_line_bar_mode")
    #group_mode = col2.selectbox("Group by:", ["Day", "Hour"], index=0,key="selectbox_stop_line_bar_group_mode")
    column = settings['Categorical'][mode]['column']

    # Group by the extracted date and state, then aggregate as per requirements
    grouped_df = adjusted_df.groupby([column, "Day"]).agg(
        stop_count=('start_time', 'count'),
        total_duration=('duration', 'sum')
    ).reset_index()
    
    plot_stop_line_bar(grouped_df, column, mode, 'Day', type)
    
    return

@st.cache_data
def stop_tree_diagram(df, state_info):
    
    data = df
    # Find unique values for 'state' and 'classification' inside each state
    unique_states = data['state'].unique()
    
    unique_states = [state for state in unique_states if state not in state_info["working_states"]]
    
    
    classification_per_state = {state: data[data['state'] == state]['classification'].unique() for state in unique_states}
        
    # Constructing the Graphviz DOT format string
    dot_string = "digraph {\n"
    dot_string += '    rankdir=TB;\n'  # Top to Bottom direction

    # for state, classifications in classification_per_state.items():
    #     for classification in classifications:
    #         dot_string += f'    "{state}" -> "{classification}";\n'
            
    # dot_string += "}\n"
    
    
    # Constructing the compact Graphviz DOT format string with adjustments, avoiding the backslash issue in f-strings
    compact_dot_string = "digraph {\n"
    compact_dot_string += '    rankdir=LR;\n'  # Left to Right direction
    compact_dot_string += '    node [shape=box, style=filled, fillcolor="lemonchiffon"];\n'  # Node style

    # Iterating through states and classifications with the possibility of creating clusters for better organization
    for state, classifications in classification_per_state.items():
        # Creating a cluster for each state to group classifications
        compact_dot_string += '    subgraph cluster_' + state.replace(" ", "_") + ' {\n'
        compact_dot_string += '        label="' + state + '";\n'  # State name as label of the cluster
        compact_dot_string += '        color=blue;\n'  # Color of the cluster border
        
        for classification in classifications:
            # Creating a node for each classification and connecting it to the corresponding state
            compact_dot_string += '        "' + classification + '" [label="' + classification.replace(" ", r"\n") + '"];\n'
            compact_dot_string += '        "' + state + '" -> "' + classification + '";\n'
            
        compact_dot_string += '    }\n'
        
    compact_dot_string += "}\n"
    
    
    
    st.graphviz_chart(compact_dot_string)
    
    
        
    return

def calculate_planned_production_time(start_date, end_date, shift_start_time, shift_end_time, first_work_day, last_work_day):
    # # Convert start_date and end_date to datetime objects
    # start_date = datetime.datetime.strptime(start_date)
    # end_date = datetime.datetime.strptime(end_date)
    
    # Initialize Planned Production Time to zero
    planned_production_time = timedelta()
    
    # Map weekday strings to integers (Monday=0, Sunday=6)
    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    first_work_day = weekday_map.get(first_work_day, 0)
    last_work_day = weekday_map.get(last_work_day, 6)
    
    # Create a list for workdays
    work_days = list(range(first_work_day, last_work_day+1))
    
    # Create timedelta objects for shift_start_time and shift_end_time
    shift_start_time = timedelta(hours=shift_start_time.hour, minutes=shift_start_time.minute)
    shift_end_time = timedelta(hours=shift_end_time.hour, minutes=shift_end_time.minute)
    
    # Calculate shift duration
    shift_duration = shift_end_time - shift_start_time
    
    # Loop through each day from start_date to end_date
    current_date = start_date
    while current_date <= end_date:
        # Check if the current day is a work day
        if current_date.weekday() in work_days:
            planned_production_time += shift_duration
        current_date += timedelta(days=1)
    
    # Convert planned_production_time to hours
    planned_production_time_hours = planned_production_time.total_seconds() / 3600
    
    planned_production_time_minutes = planned_production_time.total_seconds() / 60
    
    return planned_production_time_minutes

def get_metrics(df, plot_variables, state_info, settings):
    
    time = plot_variables["Time"]["Column"]
    
    shift_start_time = st.session_state.shift['start_time']
    shift_end_time = st.session_state.shift['end_time']
    
    first_work_day = st.session_state.shift['work_days'][0]
    last_work_day = st.session_state.shift['work_days'][-1]
    
    start_date = df[time].min()
    end_date = df[time].max()
    
    ppt_minutes = calculate_planned_production_time(start_date, 
                                            end_date, 
                                            shift_start_time, 
                                            shift_end_time, 
                                            first_work_day, 
                                            last_work_day)

    
    state_column = settings['Categorical']['State']['column']
    # Use the isin() function to filter the DataFrame
    filtered_df = df[df[state_column].isin(state_info['working_states'])]
    # Count the number of occurrences
    run_time = filtered_df.shape[0] 
    
    stop_time = ppt_minutes - run_time
    
    # TODO - run_time is in units of resampling and ppt_minutes is in minutes
    
    return run_time, stop_time, ppt_minutes
   
def count_workdays(start_date, end_date, first_work_day, last_work_day):
    # Convert start_date and end_date to datetime objects
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Initialize count of workdays to zero
    workday_count = 0
    
    # Map weekday strings to integers (Monday=0, Sunday=6)
    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    first_work_day = weekday_map.get(first_work_day, 0)
    last_work_day = weekday_map.get(last_work_day, 6)
    
    # Create a list for workdays
    work_days = list(range(first_work_day, last_work_day + 1))
    
    # Loop through each day from start_date to end_date
    current_date = start_date
    while current_date <= end_date:
        # Check if the current day is a work day
        if current_date.weekday() in work_days:
            workday_count += 1
        current_date += timedelta(days=1)
    
    return workday_count   
   
def metrics(small_df, big_df, plot_variables, state_info, settings):
        
    total_run_time, total_stop_time, total_ppt = get_metrics(big_df, plot_variables, state_info, settings)
    time = plot_variables["Time"]["Column"]
    start_date = big_df[time].min().strftime("%Y-%m-%d")
    end_date = big_df[time].max().strftime("%Y-%m-%d")
    first_work_day = st.session_state.shift['work_days'][0]
    last_work_day = st.session_state.shift['work_days'][1]
    total_number_workdays = count_workdays(start_date, end_date, first_work_day, last_work_day)
    
    avg_run_time = total_run_time / total_number_workdays
    avg_stop_time = total_stop_time / total_number_workdays
    avg_ppt = total_ppt / total_number_workdays
    
    
    run_time, stop_time, ppt = get_metrics(small_df, plot_variables, state_info, settings)
    start_date = small_df[time].min().strftime("%Y-%m-%d")
    end_date = small_df[time].max().strftime("%Y-%m-%d")
    number_workdays = count_workdays(start_date, end_date, first_work_day, last_work_day)
    
    # For thoose workdays
    avg_workdays_run_time = avg_run_time * number_workdays
    avg_workdays_stop_time = avg_stop_time * number_workdays
    avg_workdays_ppt = avg_ppt * number_workdays

    delta_run_time = run_time - avg_workdays_run_time
    delta_stop_time = stop_time - avg_workdays_stop_time
    
    if run_time < 1000 and stop_time < 1000: 
        run_time_str = str(run_time) + " min"
        stop_time_str = str(stop_time) + " min"
        delta_run_time_str = str(delta_run_time) + " min"
        delta_stop_time_str = str(delta_stop_time) + " min"
        
    else:
        run_time_str = str(round(run_time/60, 1)) + " h"
        stop_time_str = str(round(stop_time/60, 1)) + " h"
        delta_run_time_str = str(round(delta_run_time/60, 1)) + " h"
        delta_stop_time_str = str(round(delta_stop_time/60, 1)) + " h"
        
    
    availability = run_time / ppt * 100
    availability_str = str(round(availability, 1)) + " %"
    
    delta_availability = delta_run_time / avg_workdays_ppt * 100
    delta_availability_str = str(round(delta_availability, 1)) + " %"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Run Time", 
                run_time_str, 
                delta_run_time_str,
                help="Total time the machine was running. Delta is compared to the average run time per workday.")
    
    col2.metric("Stop Time", 
                stop_time_str, 
                delta_stop_time_str, 
                delta_color="inverse",
                help="Total time the machine was stopped during shift. Delta is compared to the average stop time per workday.")
    
    col3.metric("Availability", 
                availability_str, 
                delta_availability_str,
                help="Percentage of time the machine was running during shift. Delta is compared to the average availability per workday.")
    
    return    

def overall_metrics(device):
    
    small_df = device.data.copy()
    
    with open("temp/"+device.name+'.pkl', 'rb') as f:
        big_device = pickle.load(f)
        
    big_df = big_device['data']
    
    metrics(small_df, big_df, device.plot_variables, device.state_info, device.settings)
    
    return

def stop_activity_from_data(device):
    
    working, stops_dict, df = stops_from_data(device.plot_variables, device.state_info, device.data.copy(), device.settings)
    
    stop_info = device.stop_info.copy()
    state_info = device.state_info.copy()
    
    
    
    col1, col2 = st.columns([10, 1])
    col1.write("#### Interruptions Timeline")
    col2.write("")
    show_stop_timeline = col2.toggle("Show", key="toggle_stop_timeline", value=True if device.time_delta().days <= 92 else False)
    if show_stop_timeline:
        if device.time_delta().days > 93:
            st.warning("The time period is very large, it may take a while to load the plot.")
        df = plot_ticks_2(df, device.settings, device.plot_variables, state_info, stop_info)
    
    st.write("#### Interruptions Distribution")
    stop_distribution(df, stop_info, state_info, device.settings, stops_dict)
    st.write("#### Interruptions Histogram")
    stop_histogram(stop_info, device.settings.copy(), stops_dict, "From Data")
    st.write("#### Interruptions over Day")
    stop_line_bar(device.settings, stop_info, stops_dict, "From Data")
    st.write("#### Metrics")
    overall_metrics(device)
    st.write("#### Stop Tree Diagram")
    stop_tree_diagram(df, state_info)
    
    return

# Function to find stop type
def find_stop_type(duration, stop_info_df):
    for idx, row in stop_info_df.iterrows():
        if row['min'] <= duration <= row['max']:
            return row['name']
    return "Unknown"


def generate_timestamps(row, include_columns):
    start = pd.Timestamp(row["start_time"])
    end = pd.Timestamp(row["end_time"])
    duration = pd.Timedelta(minutes=1)
    timestamps = pd.date_range(start, end, freq=duration)
    
    # Create a dictionary for columns to be included
    extra_data = {col: row[col] for col in include_columns}
    
    # Generate DataFrame with individual timestamps and extra columns
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            **extra_data,
        }
    )

def stop_activity_personalized(device):
    
    time = device.plot_variables["Time"]["Column"]
        
    working, stops_dict, df = stops_from_data(device.plot_variables, device.state_info, device.data.copy(), device.settings)    
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(stops_dict)
    
    # Convert stop_info to DataFrame
    stop_info_df = pd.DataFrame(device.stop_info.copy())
    
    # Apply the function to find stop type based on 'duration'
    df['stop_type'] = df['duration'].apply(lambda x: find_stop_type(x, stop_info_df))   
    
    # Remove rows where 'stop_type' is None
    df_stops = df.dropna(subset=['stop_type'])
    
    new_settings = device.settings.copy()
    new_settings['Categorical']['Interruption Type'] = {'column': 'stop_type', 'variable_name': 'Interruption Type'}
       
        
    
    # To make interruptions timeline    
        
    # Columns to exclude
    exclude_columns = ['start_time', 'end_time', 'duration']

    # Columns to include
    include_columns = [col for col in df_stops.columns if col not in exclude_columns]

    # Apply function to each row and concatenate results
    df_stops_timeline = pd.concat([generate_timestamps(row[1], include_columns) for row in df_stops.iterrows()], ignore_index=True)
    df_stops_timeline = df_stops_timeline.rename(columns={'timestamp': time})
    
    df_temp = device.data.copy()
    merged_df = pd.merge(df_temp, df_stops_timeline[[time, 'stop_type']], on=time, how='outer')
    working_state = device.state_info["working_states"][0]
    merged_df['stop_type'].fillna(working_state, inplace=True)

    
    
    
    st.write("#### Interruptions Timeline")
    plot_tick_2_help(merged_df, new_settings, device.plot_variables, 'State', False, device.state_info)
    plot_tick_2_help(merged_df, new_settings, device.plot_variables, 'Interruption Type', True, device.state_info)
        
        
        
        
        
    st.write("#### Interruptions Histogram")    
    stop_histogram(device.stop_info, new_settings, df_stops, "Personalized")     
        
        
    st.write("#### Interruptions over Time")     
    stop_line_bar(new_settings, device.stop_info, df_stops, "Personalized")  
        
        
        
        
        
    time = device.plot_variables["Time"]["Column"]
    state = device.settings['Categorical']['State']['column']
    working_states = device.state_info["working_states"]
    df = device["data"].copy()
    stop_info = device.stop_info.copy()  
        
    list_interruptions = []

    for i in range(len(stop_info["name"])):
        # Get the list of interruptions
        interruptions, list = plot_stop_activity_helper(df, stop_info, i, time, state, working_states)
        # Convert the list of interruptions to a DataFrame and group by hour 
        list_interruptions.append(interruption_converter(list))
        # Convert the list of lists to a DataFrame
        df1 = pd.DataFrame(interruptions, columns=[time, state])
        # Concatenate the DataFrames vertically
        df = pd.concat([df, df1], ignore_index=True)
    
        
    device.data = df
    
    # st.write("#### Interruptions Timeline")
    # plot_ticks(df, device.settings, device.plot_variables)
    interruptions = plot_interruptions(device.data.copy(), device.stop_info, list_interruptions) 
    
    st.write("#### Interruptions Temporal Analysis")
    tab1, tab2 = st.tabs(["Week View", "Month View"])
    with tab1:
        plot_interruptions_week_heatmap(device.stop_info, list_interruptions)
    
    with tab2:
        plot_interruptions_month_heatmap(interruptions)
    
    
    st.write("#### Metrics")
    overall_metrics(device)
    
    
    return


def plot_stop_activity(device):
    
    if device.stop_info['from_data'] == True:
        stop_activity_from_data(device)
        
    else:
        stop_activity_personalized(device)

    return