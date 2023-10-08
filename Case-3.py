import numpy as np 
import pandas as pd
import plotly.express as px
import streamlit as st

lp_data = pd.read_csv('laadpaaldata.csv')

# Data includes Feb 29th but 2018 wasn't a leap year? Setting invalid dates to NaT and dropping them
lp_data['Started'] = pd.to_datetime(lp_data['Started'], errors='coerce')
lp_data['Ended'] = pd.to_datetime(lp_data['Ended'], errors='coerce')
lp_data = lp_data.dropna()

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


# Plot 2
# Create a new DataFrame to store the expanded records with adjusted 'TotalEnergy' values
expanded_data = []

for _, row in lp_data.iterrows():
    # Calculate the duration of each record in minutes
    duration_minutes = (row['Ended'] - row['Started']).total_seconds() / 60
    
    # Calculate the 'TotalEnergy' per minute
    energy_per_minute = row['TotalEnergy'] / duration_minutes

    # Create new rows for each minute within the range with adjusted 'TotalEnergy'
    for minute in pd.date_range(row['Started'], row['Ended'], freq='T'):
        expanded_data.append({'Date': minute, 'TotalEnergy': energy_per_minute})

# Create a DataFrame from the expanded data
expanded_lp_data = pd.DataFrame(expanded_data)

# Group by 15-minute intervals, calculating the sum 'TotalEnergy' for a single day
interval_energy = expanded_lp_data.groupby(expanded_lp_data['Date'].dt.strftime('%H:%M'))['TotalEnergy'].sum().reset_index()

# Creating bar plot to show energy needs
fig2 = px.bar(interval_energy, x='Date', y='TotalEnergy', title='Total Annual Energy Over 1-Minute Intervals for 2018')


# Streamlit section
st.title("Case 3 Dashboard WIP")
st.caption("By Emma Wartena, Luuk de Goede, Xander van Altena and Salah Bentaher")

st.subheader("Data exploration")
st.plotly_chart(fig)
st.plotly_chart(fig2)