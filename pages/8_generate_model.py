import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import joblib
import datetime

# Page config
st.set_page_config(page_title="Predictive Model Generator", layout="wide")
st.title("Predictive Model Generator")

# --- Data Loading Functions ---
def load_local_files(base_dir, selected_csvs):
    dfs = []
    for csv_file in selected_csvs:
        file_path = os.path.join(base_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = csv_file
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading file {csv_file}: {e}")
    return dfs

def load_uploaded_files(uploaded_files):
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            df['source_file'] = uploaded_file.name
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading uploaded file {uploaded_file.name}: {e}")
    return dfs

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data Selection")
    prediction_type = st.radio(
        "What would you like to predict?",
        ["Animal Presence", "Environmental Conditions"]
    )
    
    # Camera selection
    camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]
    selected_camera = st.selectbox("Select Camera", camera_names)
    
    # File upload and selection
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
    
    base_dir = os.path.join("timeseries", selected_camera)
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    csv_files = [os.path.relpath(f, base_dir) for f in csv_files]
    selected_csvs = st.multiselect("Select CSV Files", csv_files)

# --- Load and Process Data ---
dfs = []
if selected_csvs:
    dfs.extend(load_local_files(base_dir, selected_csvs))
if uploaded_files:
    dfs.extend(load_uploaded_files(uploaded_files))

if not dfs:
    st.warning("Please select or upload CSV files to analyze.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

# --- Data Preprocessing ---
def prepare_data(data, target_type="animal"):
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Extract temporal features
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    # Select features based on prediction type
    if target_type == "animal":
        # Features for animal prediction
        feature_cols = ['temperature', 'salinity', 'oxygen', 'chlorophyll',
                       'hour', 'day', 'month', 'day_of_week']
        target_col = 'animal_count'
    else:
        # Features for environmental prediction
        feature_cols = ['animal_count', 'hour', 'day', 'month', 'day_of_week']
        target_col = st.selectbox("Select Environmental Variable to Predict",
                                ['temperature', 'salinity', 'oxygen', 'chlorophyll'])
    
    # Remove rows with missing values
    data = data.dropna(subset=feature_cols + [target_col])
    
    return data, feature_cols, target_col

# --- Model Training ---
def train_model(X, y, model_type="classifier"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == "classifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test, model_type="classifier"):
    y_pred = model.predict(X_test)
    
    if model_type == "classifier":
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual"),
                       title="Confusion Matrix")
        st.plotly_chart(fig)
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        
        # Plot actual vs predicted
        fig = px.scatter(x=y_test, y=y_pred,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="Actual vs Predicted Values")
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines', name='Perfect Prediction'))
        st.plotly_chart(fig)

# --- Model Training Interface ---
st.header("Model Training")

# Prepare data
data_prepared, feature_cols, target_col = prepare_data(data, 
                                                     "animal" if prediction_type == "Animal Presence" else "environmental")

# Display feature importance threshold
importance_threshold = st.slider("Feature Importance Threshold", 0.0, 1.0, 0.05)

# Train model button
if st.button("Train Model"):
    with st.spinner("Training model..."):
        X = data_prepared[feature_cols]
        y = data_prepared[target_col]
        
        model_type = "classifier" if prediction_type == "Animal Presence" else "regressor"
        model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test) = train_model(X, y, model_type)
        
        # Save model and scaler
        model_path = os.path.join("models", f"{selected_camera}_{prediction_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        scaler_path = os.path.join("models", f"{selected_camera}_{prediction_type}_scaler_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Evaluate model
        st.subheader("Model Evaluation")
        evaluate_model(model, X_test_scaled, y_test, model_type)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        fig = px.bar(importance[importance['importance'] >= importance_threshold],
                    x='feature', y='importance',
                    title="Feature Importance")
        st.plotly_chart(fig)

# --- Prediction Interface ---
st.header("Make Predictions")

# Input form for prediction
st.subheader("Enter Values for Prediction")
prediction_inputs = {}
for feature in feature_cols:
    if feature in ['hour', 'day', 'month', 'day_of_week']:
        prediction_inputs[feature] = st.number_input(f"Enter {feature}", 
                                                   min_value=0,
                                                   max_value={'hour': 23, 'day': 31, 'month': 12, 'day_of_week': 6}[feature],
                                                   value=0)
    else:
        prediction_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Make Prediction"):
    # Load the most recent model
    try:
        model_files = glob.glob(os.path.join("models", f"{selected_camera}_{prediction_type}_*.joblib"))
        scaler_files = glob.glob(os.path.join("models", f"{selected_camera}_{prediction_type}_scaler_*.joblib"))
        
        if not model_files or not scaler_files:
            st.error("No trained model found. Please train a model first.")
            st.stop()
            
        latest_model = max(model_files, key=os.path.getctime)
        latest_scaler = max(scaler_files, key=os.path.getctime)
        
        model = joblib.load(latest_model)
        scaler = joblib.load(latest_scaler)
        
        # Prepare input data
        X_pred = pd.DataFrame([prediction_inputs])
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        prediction = model.predict(X_pred_scaled)
        
        st.subheader("Prediction Result")
        if prediction_type == "Animal Presence":
            st.write(f"Predicted animal count: {prediction[0]}")
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_pred_scaled)
                st.write(f"Confidence: {np.max(probabilities[0]):.2%}")
        else:
            st.write(f"Predicted {target_col}: {prediction[0]:.2f}")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# --- Model Management ---
st.header("Model Management")
model_files = glob.glob(os.path.join("models", "*.joblib"))
if model_files:
    st.subheader("Saved Models")
    for model_file in model_files:
        st.write(os.path.basename(model_file))
    
    if st.button("Delete All Models"):
        try:
            for model_file in model_files:
                os.remove(model_file)
            st.success("All models deleted successfully")
        except Exception as e:
            st.error(f"Error deleting models: {e}")
else:
    st.write("No saved models found")
