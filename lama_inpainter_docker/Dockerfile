FROM nvcr.io/nvidia/tritonserver:23.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add other potential OpenCV dependencies if needed, e.g.:
    # libsm6 libxext6 libxrender-dev
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt