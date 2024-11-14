FROM ubuntu:22.04

WORKDIR /opt
COPY . /opt

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

# Add necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    software-properties-common \
    libgdbm-dev \
    libc6-dev \
    liblzma-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    curl \
    gnupg2 \
    libnetcdf-dev \
    # libhdf4-alt-dev \
    libhdf4-dev \
    libhdf5-dev \
    build-essential \
    zlib1g-dev \
    libcurl4-gnutls-dev \
    libssl-dev

# Add deadsnakes PPA and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Install GDAL
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev proj-bin proj-data

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# # Install Python packages
# RUN pip3 install GDAL==3.6.4
# RUN pip3 install -r /opt/requirements.txt

# ENTRYPOINT [ "/usr/bin/python3", "/opt/src/layer_snow_cover/main.py" ]

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh \
    && bash /opt/miniconda.sh -b -p /opt/miniconda \
    && rm /opt/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/miniconda/bin:$PATH

# Create and activate conda environment
RUN conda create --name sos_test python=${PYTHON_VERSION} -y \
    && conda clean -a -y

# Activate the environment and install dependencies
RUN /bin/bash -c "source activate sos_test \
    && pip install -r /opt/requirements.txt \
    && conda install xarray netcdf4 -y \
    && conda install conda-forge::rioxarray -y"
    # && conda install conda-forge::regex -y"

# Set the entrypoint
ENTRYPOINT [ "/opt/miniconda/envs/sos_test/bin/python", "/opt/src/layer_snow_cover/main.py" ]