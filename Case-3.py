import numpy as np 
import pandas as pd
import requests
import plotly.express as px
import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

lp_data = pd.read_csv('laadpaaldata.csv')
car_data = pd.read_csv('verkeersprestaties_2015_2021.csv')

@st.cache_data(ttl=1200)
def api_call():
    response = requests.get("https://api.openchargemap.io/v3/poi/?output=geojson&countrycode=NL&maxresults=8000&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017")
    
    if response.status_code == 200:
        data = response.json()
        geo_data = gpd.GeoDataFrame.from_features(data["features"])
        return geo_data
    
    else: 
        print(f"Error: {response.status_code}")
        return None
        
cached_geo = api_call()

# Data includes Feb 29th but 2018 wasn't a leap year? Setting invalid dates to NaT and dropping them
lp_data['Started'] = pd.to_datetime(lp_data['Started'], errors='coerce')
lp_data['Ended'] = pd.to_datetime(lp_data['Ended'], errors='coerce')
lp_data = lp_data.dropna()

# Filtering car data
car_sum_data = car_data[car_data['Leeftijd'] == 'Totaal']
car_sum_data = car_sum_data.drop(columns=['Leeftijd', 'km_2021', 'km_2020', 'km_2019', 'km_2018', 'km_2017', 'km_2016', 'km_2015'])
car_sum_data = car_sum_data.drop([0])

# Plot 1
# Create a new DataFrame to store the split charging sessions
split_sessions = []

for index, row in lp_data.iterrows():
    session_start = row['Started']
    session_end = row['Ended']
    session_energy = row['TotalEnergy']
    
    while session_start.date() < session_end.date():
        # Calculate the duration of the current day's session
        current_day_end = session_start.replace(hour=23, minute=59, second=59)
        duration_hours = (current_day_end - session_start).total_seconds() / 3600
        
        # Calculate the energy assigned to the current day
        fraction_energy = (session_energy * duration_hours) / row['ConnectedTime']
        
        # Append the current day's session to the split_sessions DataFrame
        split_row = row.copy()
        split_row['Started'] = session_start
        split_row['Ended'] = current_day_end
        split_row['TotalEnergy'] = fraction_energy
        split_row['ConnectedTime'] = duration_hours
        split_sessions.append(split_row)
        
        # Update session_start for the next day
        session_start = current_day_end + pd.Timedelta(seconds=1)
    
    # Append the remaining part of the session (if any)
    if session_start <= session_end:
        duration_hours = (session_end - session_start).total_seconds() / 3600
        fraction_energy = (session_energy * duration_hours) / row['ConnectedTime']
        split_row = row.copy()
        split_row['Started'] = session_start
        split_row['Ended'] = session_end
        split_row['TotalEnergy'] = fraction_energy
        split_row['ConnectedTime'] = duration_hours
        split_sessions.append(split_row)

# Create a new DataFrame with the split charging sessions
split_lp_data = pd.DataFrame(split_sessions)

# Group by date and sum the assigned fractions of 'TotalEnergy'
daily_energy = split_lp_data.groupby(split_lp_data['Started'].dt.date)['TotalEnergy'].sum().reset_index()

# Calculating annual mean
overall_mean_energy = daily_energy['TotalEnergy'].mean()

# Creating bar plot to check for any annual spikes
fig = px.bar(daily_energy, x='Started', y='TotalEnergy', title='Total Energy Consumption per Day')
fig.add_hline(y=overall_mean_energy, line_dash='dash', line_color='red', annotation_text=f'Overall Mean: {overall_mean_energy:.2f}')
fig.update_annotations(x=1, y=1, font=dict(size=15, color="red"))

# Streamlit section
st.title("Case 3 Dashboard WIP")
st.caption("By Emma Wartena, Luuk de Goede, Xander van Altena and Salah Bentaher")

st.subheader("Data exploration")
st.plotly_chart(fig, use_container_width=True)

# Plot 2 
car_sum_data_melted = pd.melt(car_sum_data, id_vars=['Brandstofsoort'], var_name='Year', value_name='Number of Cars')

car_sum_data_melted['Year'] = car_sum_data_melted['Year'].str.extract('(\d+)').astype(int)

fig2 = px.line(car_sum_data_melted, x='Year', y='Number of Cars', color='Brandstofsoort',
              title='Number of Cars by Fuel Type (2015-2021)',
              labels={'Number of Cars': 'Number of Cars'},
              hover_name='Brandstofsoort')

fig2.update_traces(mode='lines+markers')

fig2.update_layout(showlegend=True, yaxis_type="log")

# Streamlit section
st.plotly_chart(fig2, use_container_width=True)

# Plot 3
m = folium.Map(location=[52.3788, 4.9005], zoom_start=8)
marker_cluster = MarkerCluster()

for idx, row in cached_geo.iterrows():
    popup_text = f"<b>{row['name']}</b><br>{row['description']}<br>Level: {row['level']}<br>Connection Type: {row['connectionType']}<br><a href='{row['url']}' target='_blank'>More Info</a>"
    
    folium.Marker(
        location=[row['geometry'].y, row['geometry'].x],
        popup=popup_text,
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

m.add_child(marker_cluster)

# Streamlit section
st.subheader("Charging Points Map")
st_folium(m, width=700)
