FROM ubuntu:22.04

WORKDIR /opt
COPY . /opt

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10.12

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y
RUN apt-get update 
RUN apt-get install -y wget \
                       gdal-bin \
                       libgdal-dev \
                       libspatialindex-dev \
                       build-essential \
                       software-properties-common \
                       apt-utils \
                       libgl1-mesa-glx \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       libffi-dev \
                       libbz2-dev \
                       zlib1g-dev \
                    #    libreadline-gplv2-dev \
                       libncursesw5-dev \
                       libssl-dev \
                       libsqlite3-dev \
                       tk-dev \
                       libgdbm-dev \
                       libc6-dev \
                       liblzma-dev \
                       libsm6 \
                       libxext6 \
                       libxrender-dev \
                       libgl1-mesa-dev \
                       curl \
                       gnupg2

# Download and extract Python sources
RUN cd /opt \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \                                              
    && tar xzf Python-${PYTHON_VERSION}.tgz

# Build Python and remove left-over sources
RUN cd /opt/Python-${PYTHON_VERSION} \ 
    && ./configure --with-ensurepip=install \
    && make install \
    && rm /opt/Python-${PYTHON_VERSION}.tgz /opt/Python-${PYTHON_VERSION} -rf

# Install GDAL
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get update
RUN apt-get install gdal-bin
RUN apt-get install libgdal-dev
# RUN pip3 install --upgrade pip
# RUN pip3 install --upgrade wheel
# RUN pip3 install cython
# RUN pip3 install --upgrade cython
# RUN pip3 install setuptools==57.5.0
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip3 install GDAL==3.4.1
RUN pip3 install -r /opt/requirements.txt

# RUN wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.7.1.tar.gz
# RUN tar -xvf spatialindex-src-1.7.1.tar.gz
# RUN cd spatialindex-src-1.7.1/ && ./configure && make && make install
# RUN ldconfig                       
# RUN add-apt-repository ppa:ubuntugis/ppa
# RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
# RUN export C_INCLUDE_PATH=/usr/include/gdal

# RUN apt-get install -y locales && locale-gen en_US.UTF-8
# ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

ENTRYPOINT [ "/usr/local/bin/python3.10", "/opt/src/layer_snow_cover/main.py" ]