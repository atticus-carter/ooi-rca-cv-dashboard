import streamlit as st
import pandas as pd
import pydeck as pdk

st.title("Camera Locations")

# Camera coordinates and names
camera_locations = {
    "PC01A_CAMDSC102": {"latitude": 45.515167, "longitude": -125.3899, "name": "PC01A_CAMDSC102"},  # 45° 30.910′ N, 125° 23.394′ W
    "LV01C_CAMDSB106": {"latitude": 44.3694, "longitude": -125.953867, "name": "LV01C_CAMDSB106"},    # 44° 22.164′ N, 125° 57.232′ W
    "MJ01C_CAMDSB107": {"latitude": 45.637183, "longitude": -124.3055, "name": "MJ01C_CAMDSB107"},    # 45° 38.231′ N, 124° 18.330′ W
    "MJ01B_CAMDSB103": {"latitude": 45.57, "longitude": -125.146717, "name": "MJ01B_CAMDSB103"},        # 45° 34.200′ N, 125° 8.803′ W
}

# Create a DataFrame from the camera locations
camera_data = pd.DataFrame.from_dict(camera_locations, orient='index')

# Display the camera data in a table
st.dataframe(camera_data)

# Calculate the center for the initial view
mean_lat = camera_data['latitude'].mean()
mean_lon = camera_data['longitude'].mean()

# Define a bathymetry layer using ESRI's Ocean Basemap
bathymetry_layer = pdk.Layer(
    "TileLayer",
    data="https://services.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}",
    min_zoom=0,
    max_zoom=19,
    tile_size=256,
    opacity=0.8,
)

# Define a scatter layer for the camera markers
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=camera_data,
    get_position='[longitude, latitude]',
    get_color='[255, 0, 0, 160]',  # Red markers
    get_radius=5000,
    pickable=True,
)

# Define a text layer for annotations
text_layer = pdk.Layer(
    "TextLayer",
    data=camera_data,
    get_position='[longitude, latitude]',
    get_text='name',
    get_size=16,
    get_color=[0, 0, 0, 200],
    get_alignment_baseline="'bottom'",
)

# Set the initial view state based on the mean location
view_state = pdk.ViewState(
    latitude=mean_lat,
    longitude=mean_lon,
    zoom=7,
    pitch=0,
)

# Create the deck.gl map with the bathymetry, scatter, and text layers
r = pdk.Deck(
    map_style=None,  # Custom tile layer provides the basemap
    initial_view_state=view_state,
    layers=[bathymetry_layer, scatter_layer, text_layer],
    tooltip={"text": "{name}"}
)

st.pydeck_chart(r)
