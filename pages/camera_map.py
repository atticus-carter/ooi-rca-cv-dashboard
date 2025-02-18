import streamlit as st
import pandas as pd

st.title("Underwater Camera Locations")

# Camera coordinates
camera_locations = {
    "PC01A_CAMDSC102": (45.515167, -125.3899),  # 45° 30.910′ N, 125° 23.394′ W
    "LV01C_CAMDSB106": (44.3694, -125.953867),  # 44° 22.164′ N, 125° 57.232′ W
    "MJ01C_CAMDSB107": (45.637183, -124.3055),  # 45° 38.231′ N, 124° 18.330′ W
    "MJ01B_CAMDSB103": (45.57, -125.146717),  # 45° 34.200′ N, 125° 8.803′ W
}

# Create a DataFrame from the camera locations
camera_data = pd.DataFrame.from_dict(camera_locations, orient='index', columns=['latitude', 'longitude'])

# Display the map
st.map(camera_data)
