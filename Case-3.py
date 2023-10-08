import numpy as np 
import pandas as pd
import plotly.express as px
import streamlit as st

lp_data = pd.read_csv('laadpaaldata.csv')

# Data includes Feb 29th but 2018 wasn't a leap year? Setting invalid dates to NaT and dropping them
lp_data['Started'] = pd.to_datetime(lp_data['Started'], errors='coerce')
lp_data['Ended'] = pd.to_datetime(lp_data['Ended'], errors='coerce')
lp_data = lp_data.dropna()

# Adding date column to determine what date a given column should be assigned to
lp_data['Date'] = lp_data.apply(lambda row: pd.date_range(row['Started'].date(), row['Ended'].date()), axis=1)

# Calculating the TotalEnergy by 15 minute intervals for a single day based on the average over the entire year
interval_energy = lp_data.groupby(lp_data['Started'].dt.strftime('%H:%M:%S').astype('datetime64[m]') // pd.Timedelta(minutes=15) * pd.Timedelta(minutes=15))['TotalEnergy'].mean().reset_index()

# Creating bar plot to show energy need in 15 minute intervals based on annual mean
fig2 = px.bar(interval_energy, x='Started', y='TotalEnergy', title='Total Energy need by 15-Minute Intervals (based on the 2018 mean)')

# Grouping by date & calculating the sum of energy per day
lp_data = lp_data.explode('Date')
daily_energy = lp_data.groupby('Date')['TotalEnergy'].sum().reset_index()

# Creating bar plot to check for any annual spikes
fig = px.bar(daily_energy, x='Date', y='TotalEnergy', title='Total Energy per Day')

# Streamlit section
st.title("Case 3 Dashboard WIP")
st.caption("By Emma Wartena, Luuk de Goede, Xander van Altena and Salah Bentaher")

st.subheader("Data exploration")
st.plotly_chart(fig)

st.plotly_chart(fig2)