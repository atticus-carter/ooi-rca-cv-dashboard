FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN git lfs install

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port (if needed)
EXPOSE 8501

# Command to run the Streamlit app
CMD streamlit run main.py --server.enableCORS=false --server.enableXsrfProtection=false
