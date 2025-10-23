# Use an official lightweight Python image
FROM python:3.12-slim

# Set environment variables for better performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_WORKERS=8 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app
COPY . .

# Expose the port
EXPOSE 8000

# Use Gunicorn + Uvicorn for high performance
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "8", "--bind", "0.0.0.0:${PORT}", "--timeout", "120"]
