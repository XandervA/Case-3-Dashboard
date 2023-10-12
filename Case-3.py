import numpy as np 
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats


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

@st.cache_data(ttl=1200)
def api_call_full():
    response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=8000&compact=true&verbose=false&key=09e56aa7-4347-4006-bf65-237235dbe972")
    
    if response.status_code == 200:
        responsejson  = response.json()
        return responsejson
    
    else:
        print(f"Error: {response.status_code}")
        return None
    
Laadpalen = pd.json_normalize(api_call_full())
df4 = pd.json_normalize(Laadpalen.Connections)
df5 = pd.json_normalize(df4[0])
Laadpalen = pd.concat([Laadpalen, df5], axis=1)

# CSV Import
lp_data = pd.read_csv('laadpaaldata.csv')
car_data = pd.read_csv('verkeersprestaties_2015_2021.csv')
inkomens2019 = pd.read_csv('kvk2019-wb2022.csv', skiprows=range(1, 14), sep=';')

# Filtering lp_data
# Data includes Feb 29th but 2018 wasn't a leap year? Setting invalid dates to NaT and dropping them
lp_data['Started'] = pd.to_datetime(lp_data['Started'], errors='coerce')
lp_data['Ended'] = pd.to_datetime(lp_data['Ended'], errors='coerce')
lp_data = lp_data.dropna()

# Removing outliers and negative values from ConnectedTime and ChargeTime
lp_data = lp_data[(lp_data['ConnectedTime'] >= 0) & (lp_data['ChargeTime'] >= 0)]

# Define a function to remove outliers using the Z-score
def remove_outliers(data, column, z_threshold):
    z_scores = stats.zscore(data[column])
    return data[(z_scores < z_threshold) & (z_scores > -z_threshold)]

# Set a Z-score threshold to identify outliers (e.g., 3 is a common choice)
z_threshold = 3

# Remove outliers in 'ConnectedTime'
lp_data = remove_outliers(lp_data, 'ConnectedTime', z_threshold)
lp_data = remove_outliers(lp_data, 'ChargeTime', z_threshold)

# Filtering car_data
car_sum_data = car_data[car_data['Leeftijd'] == 'Totaal']
car_sum_data = car_sum_data.drop(columns=['Leeftijd', 'km_2021', 'km_2020', 'km_2019', 'km_2018', 'km_2017', 'km_2016', 'km_2015'])

# Filtering inkomens2019
inkomens2019_short = inkomens2019.iloc[:, 2:4]
# Hernoem de derde kolom naar 'Plaatsnaam' en de vierde kolom naar 'Inkomen'
inkomens2019_short = inkomens2019_short.rename(columns={'Unnamed: 2': 'Plaatsnaam', 'Unnamed: 3': 'Gemiddeld Inkomen'})
# Verwijder tekstpatronen zoals "Wijk xx" met reguliere expressies
inkomens2019_short['Plaatsnaam'] = inkomens2019_short['Plaatsnaam'].str.replace(r'Wijk \d{2}', '', regex=True)
# Verwijder eventuele extra spaties aan het begin of einde van de plaatsnamen
inkomens2019_short['Plaatsnaam'] = inkomens2019_short['Plaatsnaam'].str.strip()

# Combining datasets
gecombineerde_dataset = Laadpalen.merge(inkomens2019_short, left_on='AddressInfo.Town', right_on='Plaatsnaam', how='left')
# Converteer de "DateCreated" kolom naar datetime formaat
gecombineerde_dataset['DateCreated'] = pd.to_datetime(gecombineerde_dataset['DateCreated'])
# Filter de rijen waarin "DateCreated" voor 1 januari 2020 ligt
Laadpalen_2019 = gecombineerde_dataset[gecombineerde_dataset['DateCreated'] < '2020-01-01']
# Selecteer de gewenste kolommen
gecomprimeerde_set_2019 = Laadpalen_2019[['DateCreated', 'AddressInfo.Town', 'AddressInfo.Latitude', 'AddressInfo.Longitude', 'Gemiddeld Inkomen']]
gecomprimeerde_set_2019['Gemiddeld Inkomen'].fillna(27.1, inplace=True)
# Verwijder niet-numerieke waarden ('.') uit de 'Gemiddeld Inkomen'-kolom
gecomprimeerde_set_2019 = gecomprimeerde_set_2019[gecomprimeerde_set_2019['Gemiddeld Inkomen'] != '.']
# Converteer de 'Gemiddeld Inkomen'-kolom naar het numerieke formaat
gecomprimeerde_set_2019['Gemiddeld Inkomen'] = pd.to_numeric(gecomprimeerde_set_2019['Gemiddeld Inkomen'])


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
daily_energy['TotalEnergy'] = daily_energy['TotalEnergy']/1000

# Calculating annual mean
overall_mean_energy = daily_energy['TotalEnergy'].mean()

# Creating bar plot to check for any annual spikes
fig = px.bar(daily_energy, x='Started', y='TotalEnergy', title='Total Energie Consumptie per Dag (kWh)')
fig.add_hline(y=overall_mean_energy, line_dash='dash', line_color='red', annotation_text=f'Gemiddelde: {overall_mean_energy:.2f}')
fig.update_annotations(x=1, y=1, font=dict(size=15, color="red"))
fig.update_layout(yaxis_title='Totaal Energieverbruik (kWh)', xaxis_title='Datum')

# Plot 2 
car_sum_data_melted = pd.melt(car_sum_data, id_vars=['Brandstofsoort'], var_name='Year', value_name='Number of Cars')

car_sum_data_melted['Year'] = car_sum_data_melted['Year'].str.extract('(\d+)').astype(int)

fig2 = px.line(car_sum_data_melted, x='Year', y='Number of Cars', color='Brandstofsoort',
              title='''Hoeveelheid Auto's per brandstoftype (2015-2021)''',
              labels={'''Number of Cars': 'Hoeveelheid Auto's'''},
              hover_name='Brandstofsoort')

fig2.update_traces(mode='lines+markers')

fig2.update_layout(showlegend=True, yaxis_type="log", xaxis_title='Jaar', yaxis_title='''Hoeveelheid Auto's''')


# Map 1
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


# Plot 3
# Filter the relevant data
full_electric_data = car_sum_data_melted[car_sum_data_melted['Brandstofsoort'] == 'Full elektrisch/waterstof']

# Extract the year and number of cars
X = full_electric_data['Year'].values.reshape(-1, 1)
y = full_electric_data['Number of Cars'].values

# Create polynomial features
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
X_poly = poly.fit_transform(X)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the polynomial features
model.fit(X_poly, y)

# Predict the number of cars for 2022 to 2030
years_to_predict = np.arange(2022, 2031).reshape(-1, 1)
years_to_predict_poly = poly.transform(years_to_predict)
predicted_cars = model.predict(years_to_predict_poly)

# Create a DataFrame for the predictions
predicted_data = pd.DataFrame({'Year': years_to_predict.flatten(), 'Predicted Number of Cars': predicted_cars})

# Create a line plot
fig3 = px.line(predicted_data, x='Year', y='Predicted Number of Cars', title='''Voorspeld aantal volledig elektrische auto's (2022-2030) met polynomiale regressie''')
fig3.update_xaxes(title='Jaar')
fig3.update_yaxes(title='''Aantal auto's''')


# Plot 4 
# Create a figure for the boxplot
fig4 = go.Figure()

# Add box plots for ChargeTime and ConnectedTime
fig4.add_trace(go.Box(y=lp_data['ChargeTime'], name='Oplaadtijd'))
fig4.add_trace(go.Box(y=lp_data['ConnectedTime'], name='Connectietijd'))

# Calculate the mean for ChargeTime and ConnectedTime
charge_time_mean = lp_data['ChargeTime'].mean()
connected_time_mean = lp_data['ConnectedTime'].mean()

# Add mean lines to the plot with annotations
fig4.add_shape(type='line',
              y0=charge_time_mean,
              y1=charge_time_mean,
              x0=0.125,
              x1=0.375,
              xref='paper',
              line=dict(color='red', width=2))

fig4.add_annotation(
    x=0.375,
    y=charge_time_mean,
    xref='paper',
    yref='y',
    text=f'Mean: {charge_time_mean:.2f}',
    showarrow=True,
    arrowhead=2,
    ax=50,
    ay=-60,
    bordercolor='red',
    borderwidth=2,
)

fig4.add_shape(type='line',
              y0=connected_time_mean,
              y1=connected_time_mean,
              x0=0.625,
              x1=0.875,
              xref='paper',
              line=dict(color='blue', width=2))

fig4.add_annotation(
    x=0.875,
    y=connected_time_mean,
    xref='paper',
    yref='y',
    text=f'Mean: {connected_time_mean:.2f}',
    showarrow=True,
    arrowhead=2,
    ax=50,
    ay=-60,
    bordercolor='red',
    borderwidth=2,
)

# Update y-axis label and layout
fig4.update_layout(yaxis_title='Tijd (Uren)', title='Boxplot Oplaadtijd vs Connectietijd')

# Map 2 
# Definieer de kleurenschaal voor het gemiddelde inkomen
income_colors = ['red', 'orange', 'yellow', 'green', 'blue']
income_bins = [0, 24, 27, 28, 40, 50]  # Definieer hier je eigen grenswaarden

# Creëer de kaart
m2 = folium.Map(location=[52.379189, 4.899431], zoom_start=7, tiles='cartodbpositron')

# Loop door de rijen van de dataset en voeg punten toe aan de kaart
for index, row in gecomprimeerde_set_2019.iterrows():
    income = row['Gemiddeld Inkomen']

    # Zoek de juiste kleur op basis van het gemiddelde inkomen
    for i in range(len(income_bins) - 1):
        if income_bins[i] <= income < income_bins[i + 1]:
            color = income_colors[i]
            break

    folium.CircleMarker(
        location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
        radius=1,  # Je kunt de grootte van de punten hier aanpassen
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
    ).add_to(m2)
    

# Plot 5
replace_dict = {1: 'Private', 4: 'Public', 0: 'Unknown', 5: 'Membership', 6: 'Public/Notice', 2: 'Restricted Acces', 3: 'Private/Notice', 7: 'Pay At Location'}
Laadpalen['UsageTypeID'] = Laadpalen['UsageTypeID'].replace(replace_dict)

usage_counts = Laadpalen['UsageTypeID'].value_counts()
usage_counts = pd.DataFrame(usage_counts)

fig5 = px.pie(usage_counts, values='count', names=usage_counts.index, title='Cirkeldiagram openbaar/particulier gebruik')
fig5.update_traces(textinfo='percent+label')
fig5.update_layout(legend_title_text='Gebruikstype')

# Plot 6
replace_dict2 = {
    'Drenthe': 'Drenthe',
    'Drente': 'Drenthe',
    'Flevoland': 'Flevoland',
    'Friesland': 'Friesland',
    'FRL': 'Friesland',
    'Gelderland': 'Gelderland',
    'GLD': 'Gelderland',
    'Stadsregio Arnhem Nijmegen': 'Gelderland',
    'Groningen': 'Groningen',
    'Limburg': 'Limburg',
    'Noord-Brabant': 'Noord-Brabant',
    'North Brabant': 'Noord-Brabant',
    'Noord Brabant': 'Noord-Brabant',
    'Nordbrabant': 'Noord-Brabant',
    'Noord Brabant ': 'Noord-Brabant',
    'Samenwerkingsverband Regio Eindhoven': 'Noord-Brabant',
    'Noord-Holland': 'Noord-Holland',
    'Noord Holland': 'Noord-Holland',
    'Noord Holand': 'Noord-Holland',
    'Noord-Hooland': 'Noord-Holland',
    'North Holland': 'Noord-Holland',
    'North-Holland': 'Noord-Holland',
    'NH': 'Noord-Holland',
    'Stadsregio Amsterdam': 'Noord-Holland',
    'Holandia Północna': 'Noord-Holland',
    'Nordholland': 'Noord-Holland',
    'Regio Twente': 'Overijssel',
    'Regio Zwolle': 'Overijssel',
    'Utrecht': 'Utrecht',
    'UT': 'Utrecht',
    'Flevolaan': 'Utrecht',
    'UTRECHT': 'Utrecht',
    'Zeeland': 'Zeeland',
    'Seeland': 'Zeeland',
    'Zuid-Holland': 'Zuid-Holland',
    'South Holland': 'Zuid-Holland',
    'Zuid-Holland ': 'Zuid-Holland',
    'ZH': 'Zuid-Holland',
    'Stadsregio Rotterdam': 'Zuid-Holland',
    'Stellendam': 'Zuid-Holland',
    'Stadsgewest Haaglanden': 'Zuid-Holland',
    'MRDH': 'Zuid-Holland',
}

Laadpalen['AddressInfo.StateOrProvince'] = Laadpalen['AddressInfo.StateOrProvince'].map(replace_dict2)

dutch_provinces = ['Drenthe', 'Flevoland', 'Friesland', 'Gelderland', 'Groningen', 'Limburg', 'Noord-Brabant', 'Noord-Holland', 'Overijssel', 'Utrecht', 'Zeeland', 'Zuid-Holland']
df_dutch_provinces = Laadpalen[Laadpalen['AddressInfo.StateOrProvince'].isin(dutch_provinces)]
fig6 = px.pie(df_dutch_provinces, names='AddressInfo.StateOrProvince', title='Verdeling laadpalen per provincie')
fig6.update_layout(legend_title_text='Provincies')

# Streamlit section
st.title("De Elektrificatie van Nederland: De Transitie naar Elektrisch Rijden")
st.caption("By Emma Wartena, Luuk de Goede, Xander van Altena and Salah Bentaher")
st.image('https://www.volkswagen.nl/-/media/vwpkw/images/elektrisch-rijden/kosten/kosten-tab-2.jpg', caption='Credit: Volkswagen AG')

st.write('''Het aantal elektrische auto’s in Nederland neemt fors toe. De transitie naar elektrisch rijden is in volle gang. Bij het maken van deze transitie is het uiterst belangrijk dat de faciliteiten om de auto’s op te laden goed functioneren. Het goed verwerken van de data van deze laadpunten hiervoor cruciaal. Het API van OpenChargeMap bevat verschillende data van laadpalen over heel de wereld. Met deze data zijn een aantal interessante zaken opgevallen.  ''')

st.subheader("Huidige situatie & toekomstvoorspelling")  
st.plotly_chart(fig2, use_container_width=True)
st.write('''
Dit lijndiagram, dat loopt van 2015 tot 2021, toont de opvallende groei van volledig elektrische auto's in vergelijking met andere brandstofsoorten. Wat dit diagram uniek maakt, is het gebruik van een logaritmische schaal op de y-as, waardoor de exponentiële toename van elektrische voertuigen goed zichtbaar wordt.''')
st.plotly_chart(fig3, use_container_width=True)
st.write('''Uit een polynomiale regressieanalyse van de historische gegevens over de groei van volledig elektrische auto's tussen 2015 en 2021 blijkt dat naar schatting tegen 2030 ongeveer 1,8 miljoen volledig elektrische auto's op de Nederlandse wegen zullen rijden.''')
st.divider()

st.subheader("Laadpalen Kaart")
st_folium(m, width=700)
st.plotly_chart(fig6, use_container_width=True)
st.write('''In dit cirkeldiagram is de verspreiding van het aantal laadpalen per provincie zichtbaar. Hieruit blijkt dat in een aantal provincies duidelijk meer laadpalen te vinden zijn. Dit is duidelijk te zien bij de provincies Noord-Holland (2.952.622 inwoners) en Zuid-Holland (3.804.906 inwoners), die overduidelijk de meeste laadpalen hebben. Daarentegen heeft Drenthe (502.051 inwoners) het minste aantal laadpalen.''')
st.subheader('Invloed van gemiddeld inkomen op locaties laadpalen (Nederland 2019)')
st_folium(m2, width=700)
st.markdown('''
    <p><strong>Legenda - Gemiddeld Inkomen in Euro (x1000) 2019</strong></p>
    <p><span style="color: red;">&#9679;</span> Ver onder gemiddeld (<24)</p>
    <p><span style="color: orange;">&#9679;</span> Onder gemiddeld (24-27)</p>
    <p><span style="color: yellow;">&#9679;</span> Gemiddeld (~27.1)</p>
    <p><span style="color: green;">&#9679;</span> Boven gemiddeld (28-40)</p>
    <p><span style="color: blue;">&#9679;</span> Ver boven gemiddeld (40<)</p>
    </div>''', unsafe_allow_html=True)
st.write('''Uit deze kaart blijkt dat er een groot verschil zit in inkomens en het aantal laadpalen in Nederland. Zo is te zien dat in de randstad het inkomen hoger ligt en er veel meer laadpalen zijn. Er lijkt ook een correlatie tussen inkomens en het aantal laadpalen. Zodra het inkomen van een plaats onder het gemiddelde ligt zijn er ook minder laadpalen aanwezig. Dit verschil is goed te zien tussen steden als Enschede en Haarlem (allebei zo'n 160.000 inwoners). De data is gebasseerd op het aantal laadpalen en het gemiddeld inkomen per gemeente uit 2019. ''')
st.plotly_chart(fig5, use_container_width=True)
st.write('''Om een duidelijker beeld te krijgen van het gebruik van laadpalen in Nederland, werpt het bovenstaande cirkeldiagram licht op deze kwestie. Zoals te zien is, domineren privélaadsessies (zakelijk gebruik) het landschap. Dit is waarschijnlijk toe te schrijven aan het groeiend aantal elektrische personenauto's in zakelijk gebruik. De op een na grootste categorie betreft openbare laadsessies voor eigen gebruik, die ook een aanzienlijk deel van het totale aantal laadsessies vertegenwoordigen. Dit kan worden toegeschreven aan de groeiende populariteit van elektrische auto's in particuliere huishoudens in Nederland.''')
st.divider()

st.subheader("Laadpaal data")
st.plotly_chart(fig, use_container_width=True)
st.write('''In de staafdiagram is het totale energieverbruik van een X aantal laadpalen over het jaar 2018. Er is een duidelijk patroon te zien met de weken. Op vrijdagen vallen de pieken het hoogst uit. Verder valt het op dat in de zomervakantie en met kerst dalen te zien zijn. Dit soort data is erg nuttig voor de toekomst als er voorspellingen gemaakt moeten worden over de vraag en het aanbod van elektriciteit. ''')
st.plotly_chart(fig4, use_container_width=True)
st.write('''De boxplots geven weer wat de verdeling is van daadwerkelijke oplaadtijd ten opzichte van de connectietijd. Hieruit is te zien dat de meeste auto’s langer aan de paal staan dan nodig. De gemiddelde lijnen laten zien dat mensen gemiddeld twee keer langer aan de paal verbonden zijn dan nodig. ''')