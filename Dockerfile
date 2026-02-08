# Use Playwright base image with Chromium pre-installed
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium

# Copy application code
COPY . .

# Create response directory
RUN mkdir -p /app/response

# Expose port (Render sets PORT dynamically)
EXPOSE 8000

# Run the application - use shell form to expand $PORT
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
