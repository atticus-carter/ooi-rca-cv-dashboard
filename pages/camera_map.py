import streamlit as st
import pandas as pd

st.title("Camera Locations")

# Camera coordinates and names
camera_locations = {
    "PC01A_CAMDSC102": {"latitude": 45.515167, "longitude": -125.3899, "name": "PC01A_CAMDSC102"},  # 45° 30.910′ N, 125° 23.394′ W
    "LV01C_CAMDSB106": {"latitude": 44.3694, "longitude": -125.953867, "name": "LV01C_CAMDSB106"},  # 44° 22.164′ N, 125° 57.232′ W
    "MJ01C_CAMDSB107": {"latitude": 45.637183, "longitude": -124.3055, "name": "MJ01C_CAMDSB107"},  # 45° 38.231′ N, 124° 18.330′ W
    "MJ01B_CAMDSB103": {"latitude": 45.57, "longitude": -125.146717, "name": "MJ01B_CAMDSB103"},  # 45° 34.200′ N, 125° 8.803′ W
}

# Create a DataFrame from the camera locations
camera_data = pd.DataFrame.from_dict(camera_locations, orient='index')

# Display the camera data
st.dataframe(camera_data)

# Display the map
st.map(camera_data[["latitude", "longitude"]])
