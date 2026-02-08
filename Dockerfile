# Lightweight Python image - no heavy ML dependencies
FROM python:3.11-slim

WORKDIR /app

# Minimal system deps for Pillow (image processing)
RUN apt-get update && apt-get install -y \
    libjpeg62-turbo \
    libpng16-16 \
    libwebp7 \
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

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
