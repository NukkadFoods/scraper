# Use lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and ONNX Runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create response directory
RUN mkdir -p /app/response

# Expose port (Render sets PORT dynamically)
EXPOSE 8000

# Run the application - use shell form to expand $PORT
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
