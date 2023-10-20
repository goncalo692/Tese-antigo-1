import streamlit as st
import datetime
import numpy as np
import pandas as pd


@st.cache_data
def select_date_time_helper(start_date, end_date, start_time, end_time):
    start_date_time = datetime.datetime.combine(start_date, start_time)
    end_date_time = datetime.datetime.combine(end_date, end_time)

    # Convert the datetime object to epoch time in milliseconds
    start_date_time = str(int(start_date_time.timestamp() * 1000))
    end_date_time = str(int(end_date_time.timestamp() * 1000))

    return start_date_time, end_date_time


def select_date_time(df, key):
    col1, col2 = st.columns(2)
    col1_1, col1_2 = col1.columns(2)
    col2_1, col2_2 = col2.columns(2)

    start_date = col1_1.date_input("Start Date:", df['ts'].min().date(
    ), min_value=df['ts'].min().date(), max_value=df['ts'].max().date(), key=key+"_start_date")
    start_time = col1_2.time_input(
        "Start Time:", datetime.time(0, 0, 0), key=key+"_start_time")
    end_date = col2_1.date_input("End Date:", df['ts'].max().date(), min_value=df['ts'].min(
    ).date(), max_value=df['ts'].max().date(), key=key+"_end_date")
    end_time = col2_2.time_input(
        "End Time:", datetime.time(23, 59, 59), key=key+"_end_time")

    start_date_time, end_date_time = select_date_time_helper(
        start_date, end_date, start_time, end_time)

    start_date_time = datetime.datetime.fromtimestamp(
        int(start_date_time)/1000)
    end_date_time = datetime.datetime.fromtimestamp(int(end_date_time)/1000)

    # Ensure that the end date is after the start date
    if start_date_time > end_date_time:
        st.error("Error: End date must fall after start date.")

    # Filter the data based on the start and end date and time
    df = df[(df['ts'] >= start_date_time) & (df['ts'] <= end_date_time)]

    return df

def calculate_zscore(window):
    mean = window.mean()
    std = window.std()
    return np.abs((window - mean) / std)

@st.cache_data
def add_nan_values_to_array(array, x):
    values_to_add = np.full(x, np.nan)
    modified_array = np.concatenate([values_to_add, array, values_to_add])
    return modified_array

@st.cache_data
def remove_outliers_zscore_moving_window(series, window_size):

    z_scores = []
    # Iterate through the time series data using a rolling window
    for i in range(len(series) - window_size + 1):
        window = series.iloc[i:i + window_size]
        z_scores_window = calculate_zscore(window)
        z_scores_window.reset_index(drop=True, inplace=True)
        z_scores.append(z_scores_window[len(z_scores_window) // 2])
    
    result = add_nan_values_to_array(z_scores, window_size // 2)

    return result


def standard_deviation_elimitation(df):
    col1, col2, col3, = st.columns([1, 2, 2])
    peak_smoothing = col1.checkbox('Peak Smoothing', value=True)
    if peak_smoothing:
        window_size = col2.number_input('Window size:', min_value=1, max_value=100, value=5, step=2)
        if window_size % 2 == 0:
            st.error("Window size must be an odd number.")
            st.stop()
        z_scores = remove_outliers_zscore_moving_window(df['value'], window_size)
    
    return z_scores

def test():
    
    import altair as alt
    from vega_datasets import data

    movies = alt.UrlData(
        data.movies.url,
        format=alt.DataFormat(parse={"Release_Date":"date"})
    )
    ratings = ['G', 'NC-17', 'PG', 'PG-13', 'R']
    genres = ['Action', 'Adventure', 'Black Comedy', 'Comedy',
        'Concert/Performance', 'Documentary', 'Drama', 'Horror', 'Musical',
        'Romantic Comedy', 'Thriller/Suspense', 'Western']

    base = alt.Chart(movies, width=200, height=200).mark_point(filled=True).transform_calculate(
        Rounded_IMDB_Rating = "floor(datum.IMDB_Rating)",
        Hundred_Million_Production =  "datum.Production_Budget > 100000000.0 ? 100 : 10",
        Release_Year = "year(datum.Release_Date)"
    ).transform_filter(
        alt.datum.IMDB_Rating > 0
    ).transform_filter(
        alt.FieldOneOfPredicate(field='MPAA_Rating', oneOf=ratings)
    ).encode(
        x=alt.X('Worldwide_Gross:Q', scale=alt.Scale(domain=(100000,10**9), clamp=True)),
        y='IMDB_Rating:Q',
        tooltip="Title:N"
    )

    # A slider filter
    year_slider = alt.binding_range(min=1969, max=2018, step=1)
    slider_selection = alt.selection_single(bind=year_slider, fields=['Release_Year'], name="Release Year_")


    filter_year = base.add_selection(
        slider_selection
    ).transform_filter(
        slider_selection
    ).properties(title="Slider Filtering")

    # A dropdown filter
    genre_dropdown = alt.binding_select(options=genres)
    genre_select = alt.selection_single(fields=['Major_Genre'], bind=genre_dropdown, name="Genre")

    filter_genres = base.add_selection(
        genre_select
    ).transform_filter(
        genre_select
    ).properties(title="Dropdown Filtering")

    #color changing marks
    rating_radio = alt.binding_radio(options=ratings)

    rating_select = alt.selection_single(fields=['MPAA_Rating'], bind=rating_radio, name="Rating")
    rating_color_condition = alt.condition(rating_select,
                        alt.Color('MPAA_Rating:N', legend=None),
                        alt.value('lightgray'))

    highlight_ratings = base.add_selection(
        rating_select
    ).encode(
        color=rating_color_condition
    ).properties(title="Radio Button Highlighting")

    # Boolean selection for format changes
    input_checkbox = alt.binding_checkbox()
    checkbox_selection = alt.selection_single(bind=input_checkbox, name="Big Budget Films")

    size_checkbox_condition = alt.condition(checkbox_selection,
                                            alt.SizeValue(25),
                                            alt.Size('Hundred_Million_Production:Q')
                                        )

    budget_sizing = base.add_selection(
        checkbox_selection
    ).encode(
        size=size_checkbox_condition
    ).properties(title="Checkbox Formatting")

    ( filter_year | filter_genres) &  (highlight_ratings | budget_sizing  )

def experiments():
    
    test()
    
    # for device_id in st.session_state.devices_list:
    #     device = st.session_state.devices_list[device_id]
    #     df_original = device['Data']
    #     with st.expander("Standard Deviation Elimitation"):
    #         df = select_date_time(df_original[['value', 'ts']], 'standard_deviation_elimitation')
    #         z_scores = standard_deviation_elimitation(df)
    #         # Plot the original and smoothed data
    #         df.index = df['ts']
    #         #df['z_score'] = z_scores
    #         df['z_score'] = pd.to_numeric(z_scores, errors='coerce') 
    #         st.line_chart(df['value'])
    #         st.line_chart(df['z_score'])
            
          


# Run the Streamlit app
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Experiments")

    st.write("# Experiments Page")

    experiments()
