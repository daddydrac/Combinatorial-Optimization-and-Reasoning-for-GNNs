# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /sat_gnn

# Install system dependencies (for PyTorch and torch-geometric)
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libomp-dev \
    libzmq3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    cmake \
    python3-pip \
    && apt-get clean

# Upgrade pip and install wheel
RUN pip3 install --upgrade pip wheel

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torch-geometric \
    scikit-learn 

RUN pip install numpy==1.26.2

COPY sat_gnn /sat_gnn

# keep alive
CMD ["tail", "-f", "/dev/null"]

# tested and works on linux!
