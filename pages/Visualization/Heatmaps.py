import streamlit as st
import altair as alt
import copy
from datetime import datetime, timedelta, date
from datetime import time as dt_time
import itertools
from calendar import month_name
import numpy as np
import pandas as pd
from pages.Visualization.Stops import metrics

@st.cache_data
def get_weeks(start_date, end_date):

    # Initialize the result list and dictionary
    result = []
    result_dict = {}
    i = 0  # Initialize index variable

    # Iterate over each week within the time period
    current_date = start_date
    while current_date <= end_date:
        week_number = current_date.strftime("%U")  # Get the week number
        period_string = f"Week {week_number}: {current_date.strftime('%d/%m/%Y')} - {(current_date + timedelta(days=6)).strftime('%d/%m/%Y')}"  # Combine the week number and dates

        # Add the period string to the result list
        result.append(period_string)

        # Create datetime objects for the start and end dates with specific times
        start_datetime = datetime.combine(current_date, datetime.min.time())
        end_datetime = datetime.combine(current_date + timedelta(days=6), dt_time(23, 59, 00))

        # Add the start and end datetimes to the result dictionary
        result_dict[i] = {
            'start_date': start_datetime,
            'end_date': end_datetime
        }

        # Move to the next week and update the index
        current_date += timedelta(days=7)
        i += 1

    return result, result_dict

@st.cache_data
def individual_week(plot_variables, df):
    
    time = plot_variables['Time']['Column']
    # Extract the weekday and time from the 'time' column
    df['weekday'] = df[time].dt.strftime('%d/%m') + ' - ' + df[time].dt.day_name()
    df = df.sort_values(time)
    
    return df
  
@st.cache_data
def plot_week_view_numerical_time(df, plot_variables):
    
    time = plot_variables['Time']['Column']
    
    # Create the heatmap for numerical over time
    heatmap_numerical = alt.Chart(df).mark_rect().encode(
        x=alt.X('utchoursminutes(' + time + '):T', title=plot_variables['Time']['Name']),
        y=alt.Y('weekday:O', title=''),
        color=alt.Color( plot_variables['Numerical']['Column']+':Q', title=plot_variables['Numerical']['Name']),
        tooltip=[alt.Tooltip(plot_variables['Numerical']['Column'], type='quantitative', title=plot_variables['Numerical']['Name']), 
                 alt.Tooltip('weekday:O', title=''),
                 alt.Tooltip('utchoursminutes(' + time + '):T', title=plot_variables['Time']['Name']),
                 ]
    ).properties(
        height=370,
        width='container',
        title = plot_variables['Numerical']['Name'] + ' over ' + plot_variables['Time']['Name']
    )
    
    return heatmap_numerical
    
      
@st.cache_data        
def plot_week_view_categorical_time(df, plot_variables):
    
    time = plot_variables['Time']['Column']
    
    # Create the heatmap for categorical over time
    heatmap_categorical = alt.Chart(df).mark_rect().encode(
        x=alt.X('utchoursminutes(' + time + '):T', title=plot_variables['Time']['Name']),
        y=alt.Y('weekday:O', title=''),
        color=alt.Color( plot_variables['Categorical']['Column']+':N', title=plot_variables['Categorical']['Name']),
        tooltip=[alt.Tooltip(plot_variables['Categorical']['Column'], type='nominal', title=plot_variables['Categorical']['Name']), 
                 alt.Tooltip('weekday:O', title='Date'),
                 alt.Tooltip('utchoursminutes(' + time + '):T', title=plot_variables['Time']['Name']),
                 ]
    ).properties(
        height=370,
        width='container',
        title = plot_variables['Categorical']['Name'] + ' over ' + plot_variables['Time']['Name']
    ).configure_legend(
        symbolType='circle'
    )
    
    return heatmap_categorical
    
# Cannot use st.cache_data
def plot_week_view(device, df):
    
    col1, col2 = st.columns(2)
    
    heatmap_numerical = plot_week_view_numerical_time(df, device.plot_variables)
    col1.altair_chart(heatmap_numerical, use_container_width=True)
    
    df = df.dropna()
    
    heatmap_categorical = plot_week_view_categorical_time(df, device.plot_variables)
    col2.altair_chart(heatmap_categorical, use_container_width=True)     
    
    # get list of unique dates
    order = df['weekday'].unique().tolist()
    map1 = plot_distribution_per_weekday(device.plot_variables, df, order)
    col1.altair_chart(map1, use_container_width=True)  
    
    map2 = plot_distribution_per_hour(device.plot_variables, df)
    col2.altair_chart(map2, use_container_width=True)
    
    st.markdown('**Metrics**')
    metrics(df, device.data, device.plot_variables, device.state_info, device.settings)
    
    return   
   
@st.cache_data 
def bigger_than_week(df_time):
    
    # Find the minimum and maximum timestamps
    min_timestamp = df_time.min()
    max_timestamp = df_time.max()  
    
    # Calculate the time span between the minimum and maximum timestamps
    time_span = max_timestamp - min_timestamp   
    
    # Check if the time span is greater than a week
    time_span_greater_than_week = time_span > timedelta(days=7)
    
    # Step 6: Check if the timestamps include at least one Monday (0) and one Sunday (6)
    unique_weekdays = df_time.dt.weekday.unique()
    includes_monday_and_sunday = 0 in unique_weekdays and 6 in unique_weekdays
    
    if time_span_greater_than_week==True and includes_monday_and_sunday==True:
        return True
    else:
        return False
    
# Cannot use st.cache_data    
def filter_week(device, dict):  
    
    device.date_filter([dict['start_date'] , dict['end_date'] - timedelta(days=1)])
    
    return device.data
  
@st.cache_data  
def plot_mean_over_time(plot_variables, df, order):
    
    numerical = plot_variables['Numerical']['Column']
    mean_power_df = df.groupby(['weekday', 'time_of_day'])[numerical].mean().reset_index()  
    mean_power_df['time_of_day'] = pd.to_datetime(mean_power_df['time_of_day'], format='%H:%M') 
    
    heatmap = alt.Chart(mean_power_df).mark_rect().encode(
        x=alt.X('utchoursminutes(time_of_day):T', title='Time'),
        y=alt.Y('weekday:O', title='', sort=order),
        color=alt.Color( numerical+':Q', title=[f"Mean {plot_variables['Numerical']['Name']}", f"[{plot_variables['Numerical']['Unit']}]"]),
        tooltip=[
            {'field': numerical, 'type': 'quantitative', 'title': 'Mean '+plot_variables['Numerical']['Name'], 'format': '.0f'},
            {'field': 'time_of_day', 'type': 'temporal', 'title': 'Time', 'format':'%H:%M'},
            {'field': 'weekday', 'type': 'nominal', 'title': 'Day'}
        ]
    ).properties(
        height=370,
        width='container',
        title = 'Mean '+plot_variables['Numerical']['Name'] + ' over time'
    )
    
    return heatmap

@st.cache_data
def plot_mode_over_time(plot_variables, df, order):
    
    categorical = plot_variables['Categorical']['Column']
    
    grouped_df = df.groupby(['weekday', 'time_of_day'])[categorical].agg(lambda x: x.mode().iloc[0]).reset_index()
    grouped_df['time_of_day'] = pd.to_datetime(grouped_df['time_of_day'], format='%H:%M')
    
    heatmap = alt.Chart(grouped_df).mark_rect().encode(
        x=alt.X('utchoursminutes(time_of_day):T', title='Time'),
        y=alt.Y('weekday:O', title='', sort=order),
        color=alt.Color(categorical, title='Mode '+ plot_variables['Categorical']['Name'], type='nominal'),
        tooltip=[
            {'field': 'time_of_day', 'type': 'temporal', 'title': 'Time', 'format':'%H:%M'},
            {'field': 'weekday', 'type': 'nominal', 'title': 'Day'},
            {'field': categorical, 'type': 'nominal', 'title': 'Mode '+ plot_variables['Categorical']['Name']}
        ]
    ).properties(
        height=370,
        width='container',
        title = 'Mode '+ plot_variables['Categorical']['Name'] + ' over time'
    ).configure_legend(
        symbolType='circle'
    )
    
    return heatmap

@st.cache_data
def plot_distribution_per_weekday(plot_variables, df, order):
    
    categorical = plot_variables['Categorical']['Column']
    
    # Group by 'weekday' and 'material' to get counts
    grouped_df = df.groupby(['weekday', categorical]).size().reset_index(name='count')
    # Calculate the total count for each weekday
    total_count = grouped_df.groupby('weekday')['count'].transform('sum')
    # Calculate the percentage
    grouped_df['percentage'] = (grouped_df['count'] / total_count) * 100
    
    # Generate all possible combinations of 'weekday' and 'material'
    all_weekdays = order
    all_categoricals = df[categorical].unique()
    all_combinations = list(itertools.product(all_weekdays, all_categoricals))

    # Create a DataFrame with all possible combinations
    all_combinations_df = pd.DataFrame(all_combinations, columns=['weekday', categorical])
    
    # Merge with the existing DataFrame and fill missing values with zeros
    complete_df = pd.merge(
        all_combinations_df,
        grouped_df[['weekday', categorical, 'percentage']],
        on=['weekday', categorical],
        how='left'
    ).fillna(0)
    
    # Create the heatmap
    heatmap = alt.Chart(complete_df).mark_rect().encode(
        x=alt.X(categorical+':O', title=plot_variables['Categorical']['Name'], axis=alt.Axis(labelAngle=0, labelAlign='center')),
        y=alt.Y('weekday:O', title='', sort=order),
        color=alt.Color('percentage:Q', title='Percentage [%]'),
        tooltip=[
            {'field': 'weekday', 'type': 'nominal', 'title': 'Day'},
            {'field': categorical, 'type': 'nominal', 'title': plot_variables['Categorical']['Name']},
            {'field': 'percentage', 'type': 'quantitative', 'title': 'Percentage [%]', 'format': '.1f'}
        ]
    )
    
    # Add text labels
    text = heatmap.mark_text(baseline='middle').encode(
        text=alt.Text('percentage:Q', format='.0f'),
        color=alt.condition(
            alt.datum.percentage > 50,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    map = (heatmap + text).properties(
        title=plot_variables['Categorical']['Name']+ ' distribution per Weekday',
        width='container',
        height=370
    )
    
    return map

@st.cache_data
def plot_distribution_per_hour(plot_variables, df):
    
    categorical = plot_variables['Categorical']['Column']
    
    # Convert the 'time' column to datetime format and extract the hour
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour

    # Group the data by hour and material to calculate the frequency of each material in each hour
    grouped_df = df.groupby(['hour', categorical]).size().reset_index(name='count')
    # Calculate the total count for each hour
    total_count_per_hour = grouped_df.groupby('hour')['count'].sum().reset_index(name='total_count')
    # Merge the total count back to the grouped DataFrame
    grouped_df = pd.merge(grouped_df, total_count_per_hour, on='hour')
    # Calculate the percentage of each material for each hour
    grouped_df['percentage'] = (grouped_df['count'] / grouped_df['total_count']) * 100

    # Generate all possible combinations of 'hour' and 'material'
    all_hours = np.sort(df['hour'].unique())
    all_materials = df[categorical].unique()
    all_combinations = pd.MultiIndex.from_product([all_hours, all_materials], names=['hour', categorical]).to_frame(index=False)

    # Merge with the existing grouped_df, filling in missing values with 0
    complete_df = pd.merge(all_combinations, grouped_df[['hour', categorical, 'percentage']], on=['hour', categorical], how='left')
    complete_df['percentage'].fillna(0, inplace=True)
    
    # Create the heatmap
    heatmap = alt.Chart(complete_df).mark_rect().encode(
        x=alt.X(categorical, type='nominal', title=plot_variables['Categorical']['Name'], axis=alt.Axis(labelAngle=0, labelAlign='center')),
        y=alt.Y('hour:O', title='Hour'),
        color=alt.Color('percentage:Q', title='Percentage [%]'),
        tooltip=[
            {'field': categorical, 'type': 'nominal', 'title': plot_variables['Categorical']['Name']},
            {'field': 'hour', 'type': 'ordinal', 'title': 'Hour'},
            {'field': 'percentage', 'type': 'quantitative', 'title': 'Percentage [%]', 'format': '.1f'}
        ]
    ).properties(
        title=plot_variables['Categorical']['Name']+ ' distribution per Hour',
        width= 'container',
        height=370
    )
    
    return heatmap
  
# Cannot use st.cache_data  
def plot_heatmap_all_weeks(device):
    
    df = device.data.copy()
    time = device.plot_variables['Time']['Column']
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    col1, col2 = st.columns(2)
    
    # Extract the weekday and time from the 'time' column
    df['weekday'] = df[time].dt.day_name()
    df['time_of_day'] = df[time].dt.strftime('%H:%M')
    
    chart = plot_mean_over_time(device.plot_variables, df, order)
    col1.altair_chart(chart, use_container_width=True)
    
    chart = plot_mode_over_time(device.plot_variables, df, order)
    col2.altair_chart(chart, use_container_width=True)
    
    chart = plot_distribution_per_weekday(device.plot_variables, df, order)
    col1.altair_chart(chart, use_container_width=True)
    
    chart = plot_distribution_per_hour(device.plot_variables, df)
    col2.altair_chart(chart, use_container_width=True)
    
    return  
    
@st.cache_data    
def is_month(df_time):
    
    # Find the minimum and maximum timestamps
    min_timestamp = df_time.min()
    max_timestamp = df_time.max()  
    
    # Calculate the time span between the minimum and maximum timestamps
    time_span = max_timestamp - min_timestamp   
    
    # Check if the time span is greater than a month (approximately 30 days)
    time_span_greater_than_month = time_span > timedelta(days=28)
    
    # Optional: Check if the timestamps span at least one of each day of the month (from 1 to 28)
    unique_days_of_month = df_time.dt.day.unique()
    includes_min_days_of_month = all(day in unique_days_of_month for day in range(1, 29))
    
    if time_span_greater_than_month==True and includes_min_days_of_month==True:
        return True
    else:
        return False
   
@st.cache_data      
def get_months(start_date, end_date):
    
    # Initialize list to hold the months and dictionary to hold the start and end dates for each month
    months_list = []
    months_dict = {}
    
    # Convert the start_date and end_date to date objects if they are not
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise ValueError("start_date and end_date should be date objects")
    
    # Initialize the current_date to start_date
    current_date = start_date
    
    while current_date <= end_date:
        # Extract the year and month from the current_date
        year = current_date.year
        month = current_date.month
        
        # Create a key for the month (format: "Month Year")
        month_key = f"{month_name[month]} {year}"
        
        # If the month is not already in the months_list, add it
        if month_key not in months_list:
            months_list.append(month_key)
            
            # Calculate the last day of the current month
            if month == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month + 1, 1) - timedelta(days=1)
            
            # Clip the last_day to end_date if it exceeds
            last_day = min(last_day, end_date)
            
            # Update the dictionary with the start and end dates for the current month
            months_dict[month_key] = {'start_date': current_date, 'end_date': last_day}
        
        # Move to the next month
        if month == 12:
            current_date = date(year + 1, 1, 1)
        else:
            current_date = date(year, month + 1, 1)
    
    return months_list, months_dict
    
# Cannot use st.cache_data      
def filter_month(device, dict):
    
    device.date_filter([dict['start_date'] , dict['end_date']])
    
    return device.data    
    
@st.cache_data      
def mean_power_month_heatmap(plot_variables, df, order, month_name):
    
    numerical = plot_variables['Numerical']['Column']
    height = 220 + 36*df['week'].nunique()
    if height < 300:
        height = 300
    
    # Aggregate the power values for each weekday within each week
    agg_df = df.groupby(['week', 'weekday', 'day', 'year_week'])[numerical].mean().reset_index()
    
    # Define the heatmap
    base = alt.Chart(agg_df).mark_rect(cornerRadius=0).encode(
        x = alt.X('weekday:N', title='', sort=order, axis=alt.Axis(labelAngle=0, orient='top', labelAlign='center')),
        y = alt.Y('week:O', title='Week Number', axis=alt.Axis(labels=True), sort=alt.EncodingSortField(field='year_week')),
        color = alt.Color(field=numerical, type="quantitative", title=[f"Mean {plot_variables['Numerical']['Name']}", f"[{plot_variables['Numerical']['Unit']}]"]),
        tooltip=[
            {'field': numerical, 'type': 'quantitative', 'title': 'Mean '+plot_variables['Numerical']['Name'], 'format': '.0f'},   
            {'field': 'weekday', 'type': 'nominal', 'title': 'Weekday'},
            {'field': 'day', 'type': 'ordinal', 'title': 'Month Day'}, 
        ]
    )
    
    # Add text labels for day of the month
    text = base.mark_text(baseline='middle').encode(
        x = alt.X('weekday:N', title='', sort=order, axis=alt.Axis(labelAngle=0, orient='top', labelAlign='center')),
        y = alt.Y('week:O', title='', axis=alt.Axis(labels=False), sort=alt.EncodingSortField(field='year_week')),
        text='day:Q',
        #color = alt.value('black')
        color=alt.condition(
            alt.datum.ip > agg_df['ip'].max()/2,
            alt.value('white'),
            alt.value('black')
        )
    )
    heatmap = (base + text).properties(
        title='Mean '+plot_variables['Numerical']['Name']+' per day in ' + month_name,
        width='container',
        height=height,
    )
    
    return heatmap
    
@st.cache_data    
def mode_categorical_month_heatmap(plot_variables, df, order, month_name):
    
    categorical = plot_variables['Categorical']['Column']
    height = 220 + 36*df['week'].nunique()
    if height < 300:
        height = 300
    
    # Group by date and get the mode material for each day
    mode_material_per_day = df.groupby(['week', 'weekday', 'day', 'year_week'])[categorical].agg(lambda x: x.mode().iloc[0]).reset_index()
    
    # Define the heatmap
    base = alt.Chart(mode_material_per_day).mark_rect(cornerRadius=0).encode(
        x = alt.X('weekday:N', title='', sort=order, axis=alt.Axis(labelAngle=0, orient='top', labelAlign='center')),
        y = alt.Y('week:O', title='Week Number', axis=alt.Axis(labels=True), sort=alt.EncodingSortField(field='year_week')),
        color = alt.Color(field=categorical, type="nominal", title=plot_variables['Categorical']['Name']),
        tooltip=[   
            {'field': categorical, 'type': 'nominal', 'title': 'Mode '+plot_variables['Categorical']['Name']},     
            {'field': 'weekday', 'type': 'nominal', 'title': 'Weekday'},
            {'field': 'day', 'type': 'ordinal', 'title': 'Month Day'},
        ]
        #tooltip=['week', 'weekday', 'ip']
    )
    
    # Add text labels for day of the month
    text = base.mark_text(baseline='middle').encode(
        x = alt.X('weekday:N', title='', sort=order, axis=alt.Axis(labelAngle=0, orient='top', labelAlign='center')),
        y = alt.Y('week:O', title='', axis=alt.Axis(labels=False), sort=alt.EncodingSortField(field='year_week')),
        text='day:Q',
        color = alt.value('black')
    )
    heatmap = (base + text).properties(
        title='Mode '+plot_variables['Categorical']['Name']+' per day in '+month_name,
        height=height,
        width='container',
    ).configure_legend(
        symbolType='circle'
    )
    
    return heatmap
    
@st.cache_data       
def plot_individual_month(plot_variables, df):
    
    time = plot_variables['Time']['Column']
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_name = df[time].dt.month_name().unique()[0] + ' ' + str(df[time].dt.year.unique()[0])
    
    # Extract the week number and weekday from the "time" column
    df['week'] = df[time].dt.isocalendar().week
    df['weekday'] = df[time].dt.day_name()
    df['day'] = df[time].dt.day
    df['year_week'] = df[time].dt.strftime('%Y-%W')

    col1, col2 = st.columns(2)
    
    heatmap = mean_power_month_heatmap(plot_variables, df, order, month_name)
    col1.altair_chart(heatmap, use_container_width=True)
    
    heatmap = mode_categorical_month_heatmap(plot_variables, df, order, month_name)
    col2.altair_chart(heatmap, use_container_width=True)
    
    return    

# Cannot use st.cache_data
def heat_map(device):
    
    # Create tab view
    tab1, tab2 = st.tabs(["Week View", "Month View"])
    time = device.plot_variables['Time']['Column']
    
    with tab1: # Week View
        col1, col2 = st.columns([1,4])
        # Check if the time span is less than a week
        if bigger_than_week(device.data[time])==True:  # If the time span is greater than a week
            weeks, weeks_dict = get_weeks(device.data[time].min().date(), device.data[time].max().date())
            type = col1.radio("Select type of visualization", ["All weeks", "Individual week"], key="select_type_week")
            if type == "All weeks":
                plot_heatmap_all_weeks(device)
            elif type == "Individual week":
                weeks_selected = col2.multiselect("Select weeks to display", weeks, default=weeks[0], key="select_weeks")
                for week in weeks_selected:
                    df = filter_week(copy.deepcopy(device), weeks_dict[weeks.index(week)])
                    df = individual_week(device.plot_variables, df)
                    with st.expander(week, expanded=True):
                        plot_week_view(device, df) # Plot the week view
        
        else: # If the time span is less than a week
            df = device.data.copy()
            df = individual_week(device.plot_variables, df)
            plot_week_view(device, df)    
            
        
    with tab2: # Month View
        # Check if the time span is less than a month
        if is_month(device.data[time]): # If the time span is greater than a month
            months, months_dict = get_months(device.data[time].min().date(), device.data[time].max().date())
            months_selected = st.multiselect("Select months to display", months, default=months[0], key="select_months")
            for month in months_selected:
                df = filter_month(copy.deepcopy(device), months_dict[month])
                with st.expander(month, expanded=True):                        
                    plot_individual_month(device.plot_variables, df)     
        else: # If the time span is less than a month
            plot_individual_month(device.plot_variables, device.data.copy()) 
        
    return
    
    