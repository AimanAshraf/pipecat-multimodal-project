FROM python:3.11-slim
WORKDIR /app
# Install system dependencies for OpenCV and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
# Install CPU-only torch first (avoids pulling in unused CUDA packages)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy project files
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
