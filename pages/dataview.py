import streamlit as st
import duckdb
import pandas as pd
import re
import os
from scripts.model_generation import model_urls # Replace with your path
import io
import cv2 # opencv
import boto3  # Import boto3
from google.oauth2 import service_account

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
    try:
        # Create an S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["connections.s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["connections.s3"]["aws_secret_access_key"],
            region_name=st.secrets["connections.s3"]["region_name"],
        )

        # Bucket connection test
        #response = s3_client.list_buckets()
        #buckets = [bucket["Name"] for bucket in response["Buckets"]]
        #st.write(f"AWS, the buckets are: {buckets}")

    except Exception as e:
        st.error(f"The code broke during AWS authentication. Please copy and paste the AWS key code. " + str(e))
        #Ensure it exists!
        st.stop()
# Authenticate into google

    try:
        parquet_file_path = f"{camera_id}_data_{year_month}/predictions.parquet"
    except Exception as e:
        st.error(f"That parquet file doesnt exist! Double check gcs")
        st.stop()
