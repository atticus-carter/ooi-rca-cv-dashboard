FROM python:3.11-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables (if needed)
# ENV AWS_ACCESS_KEY_ID=your_access_key
# ENV AWS_SECRET_ACCESS_KEY=your_secret_key
# ENV AWS_DEFAULT_REGION=your_region

# Expose port (if needed)
EXPOSE 8501

# Command to run the Streamlit app
CMD streamlit run main.py --server.enableCORS=false --server.enableXsrfProtection=false
