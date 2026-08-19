# =================================================================ar
# Stage 1: Build & Dependencies
# =================================================================
FROM python:3.11-slim AS builder

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for building certain Python packages (e.g., build tools, image libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy and install requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =================================================================
# Stage 2: Production Runtime
# =================================================================
FROM python:3.11-slim AS runner

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Add user local bin to PATH so installed packages are executable
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Install runtime system dependencies (e.g., OpenCV dependencies if needed for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy the rest of the project source code
COPY . .

# Create necessary runtime directories for data and vector storage to avoid permission errors
RUN mkdir -p /app/data/raw-documents /app/data/images /app/vectorstore/chroma_db

# Expose the application port (e.g., for Streamlit or FastAPI)
EXPOSE 8501

# Command to run the application
CMD ["python", "app.py"]