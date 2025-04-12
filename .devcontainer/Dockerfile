FROM mcr.microsoft.com/devcontainers/python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir -p /workspaces/StayAwake-AI

# Set working directory
WORKDIR /workspaces/StayAwake-AI

# Copy requirements first to leverage Docker cache
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create Jupyter configuration directory
RUN mkdir -p /home/vscode/.jupyter

# Copy Jupyter configuration
COPY .devcontainer/jupyter/jupyter_notebook_config.py /home/vscode/.jupyter/

# Set permissions
RUN chown -R vscode:vscode /workspaces/StayAwake-AI /home/vscode/.jupyter

# Switch to vscode user
USER vscode

# Set environment variables
ENV PYTHONPATH=/workspaces/StayAwake-AI 