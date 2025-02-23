import pandas as pd
import streamlit as st
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp

import random
import numpy as np

import difflib
from geopy.distance import distance
from geopy.geocoders import Nominatim

# Set the title and page layout
st.set_page_config(page_title="Find Your Ride", page_icon=":ski:")
st.title('Find your ride: using geography to plan your next ski trip')
st.info("""One thing to know about me is that I'm a huge fan of enjoying the outdoors,
    and one of my favorite ways to do that is being on a mountain. Whether I'm skiing or
    snowboarding, I've always had a great day on the mountain.
    When I plan on going with my friends, there's always one question that makes planning
    an uphill battle: Where should we go? I have friends of all different ability levels
    across the East Coast, and pinpointing which mountain to land on becomes a multi-faceted
    question. I know I'm not alone here, so I built this web app using Streamlit to help others
    make these decisions much more quickly. Whether you're a seasoned rider or a regular on
    the bunny hill, I hope this tool can help you make a decision on where to spend your days! ❄️""")

# Load in the data
ski = pd.read_parquet('ski_resorts.parquet')
ski.drop(columns=['api_location'], inplace=True)

ski['vertical'] = ski['summit'] - ski['base']
# Split the 'resort_name' column into two columns
split_resort = ski['resort_name'].str.split(', ', n=1, expand=True)
ski['resort_name'] = split_resort[0]


# Drop rows where percents are greater than 1.0
cols_to_check = ['green_percent', 'blue_percent', 'black_percent']
for col in cols_to_check:
    ski[col] = ski[col] / 100
    ski = ski[ski[col] <= 1.0]

# Eliminate non-ski resorts: no green, no blue, no black.
ski = ski[(ski['green_percent'] != 0) | (ski['blue_percent'] != 0) | (ski['black_percent'] != 0)]

resort_names = ski["resort_name"].unique().tolist()

# Set the center coordinates for Continental US
center_latitude = 39.8283
center_longitude = -98.5795

# Create the figure directly using go.Figure
all_mountains_fig = go.Figure()

# Add the scatter points
all_mountains_fig.add_trace(go.Scattergeo(
    lon=ski['lon'],
    lat=ski['lat'],
    text=ski['resort_name'],
    mode='markers',
    marker=dict(
        size=ski['runs']/2,  # Adjust the division factor as needed
        color=ski['acres'],
        colorscale='Blugrn',
        showscale=True,
        sizemode='area',
        sizeref=2.*max(ski['runs'])/(15.**2),
    ),
    hovertemplate=
    '<b>%{text}</b><br>' +
    'State: %{customdata[0]}<br>' +
    '<extra></extra>',
    customdata=ski[['state']]
))

# Update the layout
all_mountains_fig.update_layout(
    title="Ski Resorts in North America, by number of runs and acres",
    geo=dict(
        scope='usa',  # Changed from 'north america' to 'usa'
        showland=True,
        showcoastlines=True,
        showlakes=True,
        showcountries=True,
        projection_type='albers usa',  # Changed to albers usa projection
        center=dict(lat=center_latitude, lon=center_longitude),
    ),
    width=800,
    height=600
)

st.plotly_chart(all_mountains_fig,
    use_container_width=True)


st.header("Comparing Mountains to one another")

def compare_mountains(df):
    # Select a random sample of rows from the ski DataFrame
    mountains = df
    cols_to_compare = ['green_percent', 'black_percent', 'blue_percent']

    # Scale the percent columns to 0-100
    mountains[cols_to_compare] = mountains[cols_to_compare] * 100

    # Create the r and theta lists for each mountain
    r_values = []
    theta_values = []
    for i, row in mountains.iterrows():
        r = []
        theta = []
        for col in cols_to_compare:
            r.append(row[col])
            theta.append(col.title() + ' %')
        r_values.append(r)
        theta_values.append(theta)

    # Create the polar line traces using plotly.graph_objs
    traces = []
    for i in range(mountains.shape[0]):
        trace = go.Scatterpolar(r=r_values[i], theta=theta_values[i], fill='toself', name=mountains.iloc[i]['resort_name'])
        traces.append(trace)

    # Create a layout for the chart
    layout = go.Layout(title=f"Comparison of your selected ski mountains by difficulty",
                       polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                       showlegend=True)

    # Create a figure using plotly.graph_objs
    fig = go.Figure(data=traces,
                    layout=layout)

    return(fig)

# Allow users to select mountains to compare
default_resorts = ["Jiminy Peak", "Telluride"]
selected_resorts = st.multiselect("Select ski resorts", resort_names, default=default_resorts)
filtered_resorts_df = ski[ski["resort_name"].isin(selected_resorts)]
st.write(filtered_resorts_df)

st.plotly_chart(compare_mountains(filtered_resorts_df),
                use_container_width=True)


st.info("""If you're planning a trip with multiple people,
        or you're just looking for a spot close to you,
        feel free to take a peek at what mountains are near
        you and your group! This tool takes in a zipcode and
        finds its coordinates to help you find the nearest
        mountains!""")

def is_valid_zipcode(zipcode):
    return zipcode.isdigit() or zipcode == ''

def zipcode_input():
    st.header("Input some zipcodes of who's joining you on the mountain. I'll provide some spots that work best")

    # Set the default values for zip codes
    default_zipcodes = ['06032', '06606', '02138'] + [''] * 9

    # Use the session state to store the zip codes
    if 'zipcodes' not in st.session_state:
        st.session_state.zipcodes = default_zipcodes.copy()

    num_columns = len(st.session_state.zipcodes)

    # Display the first row of zip code input fields
    first_row_cols = st.columns(3)
    for i in range(3):
        zipcode = st.session_state.zipcodes[i]
        new_zipcode = first_row_cols[i].text_input(f"Zip code {i + 1}:", value=zipcode, key=f"zip_{i}")

        if new_zipcode != zipcode:
            if is_valid_zipcode(new_zipcode):
                st.session_state.zipcodes[i] = new_zipcode
                if new_zipcode != '':
                    st.success(f"Zip code {i + 1} updated to {new_zipcode}")

    with st.expander("Edit Additional Zip Codes", expanded=False):
        # Create columns in sets of 3
        col_sets = (num_columns - 3) // 3 + (1 if (num_columns - 3) % 3 != 0 else 0)

        for set_index in range(col_sets):
            cols = st.columns(3)
            for i in range(3):
                index = set_index * 3 + i + 3
                if index < num_columns:
                    zipcode = st.session_state.zipcodes[index]
                    new_zipcode = cols[i].text_input(f"Zip code {index + 1}:", value=zipcode, key=f"zip_{index}")

                    if new_zipcode != zipcode:
                        if is_valid_zipcode(new_zipcode):
                            st.session_state.zipcodes[index] = new_zipcode
                            if new_zipcode != '':
                                st.success(f"Zip code {index + 1} updated to {new_zipcode}")

    return [zipcode for zipcode in st.session_state.zipcodes if zipcode != '']

if __name__ == "__main__":
    locations = zipcode_input()

# Read the existing parquet file or create a new one if it doesn't exist
try:
    geocoded_zipcodes = pd.read_parquet("zipcode_coordinates.parquet")
except FileNotFoundError:
    geocoded_zipcodes = pd.DataFrame(columns=["zip_code", "coordinates"])
    st.write("created new file to hold onto geocoded data")

# Check if a zip code is already in the DataFrame
def is_zip_code_in_df(zip_code):
    return geocoded_zipcodes["zip_code"].isin([zip_code]).any()

geolocator_active = 0

latitude_list = []
longitude_list = []

# Geocode the location
for location in locations:
    if is_valid_zipcode(location) and not is_zip_code_in_df(location):
        if geolocator_active == 0:
            geolocator = Nominatim(user_agent="DM_ski_project")
            geolocator_active = 1
        location_data = geolocator.geocode(location, timeout=10, country_codes="us")
        coordinates = location_data[1]
        latitude_list.append(coordinates[0])
        longitude_list.append(coordinates[1])
        st.write(f"{location} geocoded!")
        
    
        # The below code is not compatible with the Streamlit web app since it cannot alter the repo. Solution to come.
        # # Append the new coordinates to the DataFrame and save it to the file
        # geocoded_zipcodes = geocoded_zipcodes.append({"zip_code": location, "coordinates": coordinates}, ignore_index=True)
        # geocoded_zipcodes.to_parquet("geocoded_zipcodes.parquet", index=False)

    else:
        coordinates = geocoded_zipcodes.loc[geocoded_zipcodes['zip_code'] == location, 'coordinates'].iloc[0]
        latitude_list.append(coordinates[0])
        longitude_list.append(coordinates[1])

# Number of skiers
n_skiers = len(latitude_list)

# Create a dictionary to store the data for each skier
skiers_dict = {f'Skier {i+1}': [latitude_list[i], longitude_list[i]] for i in range(n_skiers)}
skiers_df = pd.DataFrame(skiers_dict).transpose()
skiers_df.columns = ['Latitude', 'Longitude']

st.info("""We now have a center of gravity for you and your group
        that we can use to find the closest mountains to you.
        Let's use this to narrow down our search!""")
st.header("How many resorts are you open to considering?")

# Get the centroid of the spots
centroid_latitude, centroid_longitude = np.mean(latitude_list), np.mean(longitude_list)

def calc_distance(row):
    resort_lat, resort_lon = row['lat'], row['lon']
    return distance((centroid_latitude, centroid_longitude), (resort_lat, resort_lon)).mi #distance in miles

ski['distance_to_centroid'] = ski.apply(calc_distance, axis=1)

# Maybe combine with above for a function
n_closest_resorts = st.slider('How many resorts do you want to consider?', min_value=1, value=4, max_value=50)
closest_resorts = ski.nsmallest(n_closest_resorts, 'distance_to_centroid')

def compare_closest_mountains(closest_resorts):
    # Select rows from the ski DataFrame based on closest_resorts
    selected_rows = ski.loc[ski['resort_name'].isin(closest_resorts.resort_name)]
    percent_cols = ['green_percent', 'black_percent', 'blue_percent']
    value_cols = ['distance_to_centroid', 'lifts', 'runs']

    # Scale the percent columns to 0-100
    selected_rows[percent_cols] = selected_rows[percent_cols] * 100

    # Find the max range value for the other columns
    max_range = max(selected_rows[value_cols].max())

    def create_trace(rows, columns):
        r_values, theta_values = [], []
        # Create a dictionary to map long names to shorter ones
        name_map = {
            'green_percent': '% Green',
            'black_percent': '% Black',
            'blue_percent': '% Blue',
            'distance_to_centroid': 'Avg Distance',
            'lifts': 'Lifts',
            'runs': 'Runs'}
    
        for i, row in rows.iterrows():
            r, theta = [], []
            for col in columns:
                # Replace long names with short names using the name_map dictionary
                theta_label = name_map.get(col, col.title())
                r.append(row[col])
                theta.append(theta_label)
            r_values.append(r)
            theta_values.append(theta)
        return r_values, theta_values

    # Create traces for the percentage columns and other columns
    percent_r, percent_theta = create_trace(selected_rows, percent_cols)
    other_r, other_theta = create_trace(selected_rows, value_cols)

    # Create polar line traces using plotly.graph_objs
    percent_traces = [go.Scatterpolar(r=r, theta=theta, fill='toself', name=row['resort_name']) for r, theta, (_, row) in zip(percent_r, percent_theta, selected_rows.iterrows())]
    other_traces = [go.Scatterpolar(r=r, theta=theta, fill='toself', name=row['resort_name']) for r, theta, (_, row) in zip(other_r, other_theta, selected_rows.iterrows())]

    # Create layouts for the charts
    percent_layout = go.Layout(title="Comparison of closest ski mountains",
                               polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                               showlegend=True)

    other_layout = go.Layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_range])),
                             showlegend=True)

    # Create figures using plotly.graph_objs
    percent_fig = go.Figure(data=percent_traces, layout=percent_layout)
    other_fig = go.Figure(data=other_traces, layout=other_layout)

    return percent_fig, other_fig

percent_fig, other_fig = compare_closest_mountains(closest_resorts)
comparison_col1, comparison_col2 = st.columns(2)
with comparison_col1:
    st.plotly_chart(percent_fig, use_container_width=True)
with comparison_col2:
    st.plotly_chart(other_fig, use_container_width=True)
st.dataframe(closest_resorts)

# Create the close resorts figure
close_resorts_fig = go.Figure()

# Add the scatter points for resorts
close_resorts_fig.add_trace(go.Scattergeo(
    lon=closest_resorts['lon'],
    lat=closest_resorts['lat'],
    text=closest_resorts['resort_name'],
    mode='markers',
    marker=dict(
        size=closest_resorts['runs']/2,
        color=closest_resorts['acres'],
        colorscale='Blugrn',
        showscale=True,
        sizemode='area',
        sizeref=2.*max(closest_resorts['runs'])/(10.**2),
    ),
    hovertemplate=
    '<b>%{text}</b><br>' +
    'State: %{customdata[0]}<br>' +
    '<extra></extra>',
    customdata=closest_resorts[['state']]
))

# Update the layout
close_resorts_fig.update_layout(
    title="Closest Ski Resorts to Your Location",
    geo=dict(
        scope='usa',
        showland=True,
        showcoastlines=True,
        showlakes=True,
        showcountries=True,
        projection_type='equirectangular',
        center=dict(lat=np.mean(closest_resorts['lat']), lon=np.mean(closest_resorts['lon'])),
    ),
    width=800,
    height=600
)

# Create trace for skiers
skiers_trace = go.Scattergeo(
    lon=skiers_df["Longitude"].round(4), 
    lat=skiers_df["Latitude"].round(4), 
    mode="markers", 
    marker=dict(size=10, color="blue"),
    text=skiers_df.index,
    name='Skiers',
    showlegend=False)

# Create trace for centroid
centroid_trace = go.Scattergeo(
    lon=[round(centroid_longitude,4)], 
    lat=[round(centroid_latitude,4)], 
    mode="markers", 
    marker=dict(size=20, color="gray", opacity=.8),
    name="Centroid",
    showlegend=False)

# Add skiers and centroid traces to the figure
close_resorts_fig.add_trace(skiers_trace)
close_resorts_fig.add_trace(centroid_trace)

st.plotly_chart(close_resorts_fig,
    use_container_width=True)

st.header("""Interested in looking at the trail maps?
Here are the resorts you're considering and links to trailmaps:""")

for index, row in closest_resorts.iterrows():
    resort_name = row['resort_name']
    trailmap_link = row['trailmap_link']
    st.markdown(f"""<div style="white-space: nowrap;">
        <a href="{trailmap_link}" style="color: black;">
        {resort_name + " Trail Map"}</a></div>""", unsafe_allow_html=True)
    
st.write('--- \n')

st.info("Don't see a mountain you're expecting to? Any ideas for improvements? Send me an email at dmarino1@me.com.")
