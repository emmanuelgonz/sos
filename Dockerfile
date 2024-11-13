# Use the official Ubuntu base image
FROM ubuntu:latest

# Set the maintainer label
LABEL maintainer="emmanuelgonzalez@asu.edu"

WORKDIR /opt
COPY . /opt

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

# Install necessary packages
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y \
    && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    zlib1g-dev \
    libssl-dev \
    libncurses5-dev \
    libsqlite3-dev \
    libreadline-dev \
    libbz2-dev \
    libffi-dev \
    libgdal-dev \
    # libhdf4-dev \
    # libhdf5-dev \
    # libnetcdf-dev \
    # gdal-bin \
    # libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

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
    && pip install --upgrade pip setuptools \
    && pip install -r /opt/requirements.txt \
    && conda install -c conda-forge libgdal-netcdf libgdal-hdf4 libgdal-hdf5 libgdal-core libgdal gdal xarray netcdf4 rioxarray -y"

# Set the entrypoint
ENTRYPOINT [ "/opt/miniconda/envs/sos_test/bin/python", "/opt/src/layer_snow_cover/main.py" ]

# # Use the official Ubuntu base image
# FROM ubuntu:latest

# # Set the maintainer label
# LABEL maintainer="emmanuelgonzalez@asu.edu"

# WORKDIR /opt
# COPY . /opt

# USER root
# ARG DEBIAN_FRONTEND=noninteractive
# ARG PYTHON_VERSION=3.12.7
# RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     wget \
#     git \
#     zlib1g-dev \
#     libssl-dev \
#     libncurses5-dev \
#     libsqlite3-dev \
#     libreadline-dev \
#     libbz2-dev \
#     libffi-dev \
#     libhdf4-dev \
#     libhdf5-dev \
#     libnetcdf-dev \
#     gdal-bin \
#     # libgdal-dev \
#     # python3-gdal \
#     libspatialindex-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Download and extract Python sources
# RUN cd /opt \
#     && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \                                              
#     && tar xzf Python-${PYTHON_VERSION}.tgz

# # Build Python and remove left-over sources
# RUN cd /opt/Python-${PYTHON_VERSION} \ 
#     && ./configure --with-ensurepip=install \
#     && make install \
#     && rm /opt/Python-${PYTHON_VERSION}.tgz /opt/Python-${PYTHON_VERSION} -rf

# # Install Python libraries
# RUN pip3 install --upgrade pip setuptools
# RUN pip3 install -r /opt/requirements.txt

# ENTRYPOINT [ "/usr/local/bin/python3.12", "/opt/src/layer_snow_cover/main.py" ]