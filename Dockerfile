# Traffic Signs Recognition, Vehicle Plate and Person Blurring System - Docker image
FROM python:3.11-slim

# System libraries required by OpenCV and video I/O (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and configuration
COPY Traffic-Signs-Recognition-Vehicle-Plate-and-Person-Blurring-System.py config.py ./

# Run the traffic signs recognition, plate and person blurring script
CMD ["python", "Traffic-Signs-Recognition-Vehicle-Plate-and-Person-Blurring-System.py"]
