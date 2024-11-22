# Use NVIDIA CUDA 12.0 base image with Python 3.10
FROM nvidia/cuda:12.0.1-base-ubuntu20.04

# Set the working directory inside the container
WORKDIR /sat_gnn

# Install system dependencies (for PyTorch, torch-geometric, and GPU support)
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libomp-dev \
    libzmq3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    python3-dev \
    python3-pip \
    && apt-get clean

# Install PyTorch with CUDA support and other Python dependencies
RUN pip3 install --no-cache-dir \
    torch \
    torch-geometric \
    scikit-learn 

RUN pip install numpy==1.26.2
RUN pip install cupy==12.2.0

COPY sat_gnn /sat_gnn

# keep alive
CMD ["tail", "-f", "/dev/null"]