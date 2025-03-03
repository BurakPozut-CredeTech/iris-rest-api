FROM python:3.9-slim

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port (for documentation and Coolify)
EXPOSE 5000

# Run Gunicorn with production settings
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "60", "--bind", "0.0.0.0:5000", "app:app"]