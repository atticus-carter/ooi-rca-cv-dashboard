import streamlit as st
import duckdb
import pandas as pd
import re
import os
from scripts.model_generation import model_urls # Replace with your path
import json
from google.oauth2 import service_account
import io
import cv2 # opencv

# --- Camera Names ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# --- Dataview Page ---
if 'camera' not in st.session_state:
    st.write("Please select camera from main page")
else:
    st.title(f"Dataview - {st.session_state.camera}")

    # --- Load variables from Session State ---
    bucket_name = st.session_state.get("bucket_name")  # Ensure bucket_name is passed
    camera_id = st.session_state.camera
    selected_model = st.session_state.get("selected_model", "SHR_DSCAM")
    year_month = "2021-08"  # The year and month of the data
    #try:
        # Load credentials from secrets to cloud storage
    gcs = st.secrets["connections.gcs"]

        # Make your credential in json, its okay to expose everything as nothing is loaded.
       # except:
       # st.write("Missing credential. Make sure you configure your secrets manager.")
       # st.stop()
        # Authenticate into google
    con = duckdb.connect(database=':memory:', read_only=False)
    csv_filepath = f"{camera_id}_data_{year_month}/predictions.parquet"

    """
    if  not os.path.exists(csv_filepath):
        st.write("Please ensure that this session has data for the GCS")
        st.write("You can configure more data from the homepage of this session")
        st.stop()
        #st.csv will not work on links unless it is public
    else:"""
    st.write("Configuring google connect... Please wait!")
    try:
        df = pd.read_parquet(csv_filepath)

        st.header("Test Pandas: Successfully connected to GCS file")
        st.subheader("Here is the data in pandas")

        #if this is true,
        st.dataframe(df)
    except:
        st.write("Please test the code first! If this does not work, then ensure that you have data for the camera folder")
        st.stop()
    #with st.expander("Click to read more in depth about this data"):
        #st.write("This will create a better visualization for you")
    #def load_lottieurl(url: str):
    #try:
        #r = requests.get(url)
        #if r.status_code != 200:
            #return None
        #return r.json()
    except:
        st.write("Ensure you have loaded everything")
        st.stop()
        # load local css file
    con.close()
        # use local css file"""
