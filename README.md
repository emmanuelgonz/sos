# Snow Observing Strategy (SOS)
This repository contains the various SOS applications integrated within the [Novel Observing Strategies Testbed (NOS-T)](https://github.com/code-lab-org/nost-tools). 

## Introduction
The manager application orchestrates the various applications by keeping a consistent time across applications. Upon initiation of manager, the applications are triggered and each is responsible for generating derived datasets or raster layers. Below is a table describing each application:

### Applications
|Application|Category|Purpose|Data Source|Threshold|Aggregation|Developed|
|:---------:|:------:|:-----:|:---------:|:-------:|:---------:|:-------:|
|Manager|Manager|Orchestrates applications, maintains time|NA|NA|NA|Y|
|SNODAS|Merged Dataset Generator|Merges data into a single, aggregated dataset|NA|NA|NA|Y|
|MOD10C1|Merged Dataset Generator|Merges data into a single, aggregated dataset|NA|NA|NA|Y|
|Snow Cover|Raster Layer Generator|Generates snow cover layer|MOD10C1|30% (snow cover)|Weekly|Y|
|Resolution|Raster Layer Generator|Generates resolution layer|SNODAS|50 mm (Abs. SWE difference)|Monthly|Y|
|Sensor Saturation|Raster Layer Generator|Generates sensor saturation layer|SNODAS|150 mm|Daily|N|
|SWE Change|Raster Layer Generator|Generates SWE change layer|SNODAS|10 mm|Daily|Y|
|Surface Temperature|Raster Layer Generator|Generates surface temperature layer|AIRS Version 7 Level 3 Product|0 &deg;C|Daily|N|

These applications transmit status messages and base64-encoded images via the NOS-T message broker, which utilizes AMQP over RabbitMQ. The figure below illustrates the overall workflow:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRb4C-NVOJblonVF0rZEC7BxwTX_6KmPXXnGQBV3DdvzSWTwJi-1SFxFE2HkTDZawDe-GBZnitIG2lq/pub?w=1489&amp;h=669">

The input data and data generated by applications is uploaded onto an Amazon Web Services (AWS) Simple Storage Service (S3) bucket with the following data structure:

```bash
snow_observing_systems/
├── daily
│   ├── SWE Change
│   │   ├── 2024-01-01
│   │   ├── 2024-01-02
│   │   ├── ...
│   │   ├── 2024-03-30
│   │   └── 2024-03-31
│   ├── Sensor Saturation
│   │   ├── 2024-01-01
│   │   ├── 2024-01-02
│   │   ├── ...
│   │   ├── 2024-03-30
│   │   └── 2024-03-31
│   └── Surface Temperature
│       ├── 2024-01-01
│       ├── 2024-01-02
│       ├── ...
│       ├── 2024-03-30
│       └── 2024-03-31
├── weekly
│   └── Snow Cover
│       ├── 2024-W01
│       ├── 2024-W02
│       ├── ...
│       ├── 2024-W13
│       └── 2024-W14
└── monthly
    └── Resolution
        ├── 2024-01
        ├── 2024-02
        └── 2024-03
```

The applications use the AWS SDK for Python, [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html). Boto3 allows users to create, configure, and manage AWS services, including S3, Simple Notification Service (SNS), and Elastic Compute Cloud (EC2). Access to the AWS SDK is limited to SOS administrators as required by NASA's [Science Managed Cloud Environment (SMCE)](https://smce.nasa.gov/). 

## Installation

If you plan to run applications using Conda, you will need to install all dependencies. Below are the steps to install all dependencies using Conda:

You can create a Conda environment to deploy applications. Please note that this method requires advanced experience working with GDAL, as it's installation can be quite tricky. If you run into issues here, please follow the[Docker](#docker-development) or [Docker compose](#docker-compose) sections. 

To set up conda, follow the steps below:

1. Create a Conda environment using Python 3.10

```bash
conda create --name sos python=3.10
```

1. Activate your Conda environment

```bash
conda activate sos
```

1. Install dependencies for GDAL

```bash
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

1. Install requirements

```bash
python3 -m pip install -r requirements.txt
```

> NOTE: If the final steps fails, it is likely due to the incorrect GDAL version being listed in the `requirements.txt` file. To correct this, identify the correct GDAL version for your system:

### Troubleshooting

```bash
gdalinfo --version
```

Your output should look similar to:
```bash
GDAL 3.6.4, released 2023/04/17
```

Finally, install the correct GDAL version by running:
```bash
python3 -m pip install GDAL==<insert version number>
```

For example,

```bash
python3 -m pip install GDAL==3.6.4
```

## Execution

### Docker (Development)

Each container can be built individually during development, to build a local version of a container, you can use ```docker build```. 

For example, to build the "manager" comntainer:

```
cd src/manager/
docker build -t sos_manager .
docker run --rm --env-file .env -v "$(pwd)/data":/opt/data sos_manager
```

> NOTE: You can follow similar steps to build the other containers, such as satellites, layer_resolution, layer_snow_cover, etc.

### Docker Compose
Three applications, including manager, satellite, and snow cover layer applications, can be run using Docker compose. To run applications, do the following:

1. Download input data, which should be in the following structure:

    ```
    data/
    ├── Downloaded_files
    │   ├── Mo_basin_shp
    │   │   ├── WBD_10_HU2_Shape
    │   │   └── WBD_10_HU2_Shape.zip
    │   ├── Resolution_raw
    │   │   └── SNODAS
    │   └── Snow_cover_raw
    │       ├── hdf
    │       └── nc
    ├── Efficiency_files
    │   ├── Efficiency_high_resolution_Caesium
    │   │   ├── efficiency_resolution.nc
    │   │   ├── efficiency_resolution_taskable.nc
    │   │   ├── efficiency_snow_cover.nc
    │   │   └── efficiency_snow_cover_up.nc
    │   └── Efficiency_resolution20_Optimization
    │       ├── Efficiency_SWE_Change_dataset_Capella.nc
    │       ├── Efficiency_Sensor_dataset.nc
    │       ├── Efficiency_Sensor_dataset_Capella.nc
    │       ├── Efficiency_Sensor_dataset_GCOM.nc
    │       ├── Efficiency_Temperature_dataset.nc
    │       ├── Efficiency_Temperature_dataset_coarsened.nc
    │       ├── Optimization_result.geojson
    │       ├── Temperature_dataset.nc
    │       ├── coarsened_eta_output_Capella.nc
    │       ├── coarsened_eta_output_GCOM.nc
    │       ├── efficiency_resolution_layer.nc
    │       ├── efficiency_snow_cover.nc
    │       ├── eta0_resampled_to_match_coarsened_grid.nc
    │       ├── final_blocks_rewards.geojson
    │       ├── final_eta_combined_output_Capella.nc
    │       └── final_eta_combined_output_GCOM.nc
    └── Preprocessed_files
        ├── preprocessed_resolution.nc
        └── preprocessed_snow_cover.nc
    ```

1. Confirm you have a .env file in your working directory with the following contents:

    ```
    # FILE/DIR PATHS 
    path_hdf=data/Downloaded_files/Snow_cover_raw/hdf
    path_nc=data/Downloaded_files/Snow_cover_raw/nc/
    path_shp=data/Downloaded_files/Mo_basin_shp/
    path_preprocessed=data/Preprocessed_files/
    path_efficiency=data/Efficiency_files/Efficiency_high_resolution_Caesium/
    raw_path=data/Downloaded_files/Resolution_raw/

    # EARTHACCESS LOGIN
    EARTHDATA_USERNAME=<Your EarthAccess username>
    EARTHDATA_PASSWORD=<Your EarthAccess password>

    # NOS-T LOGIN
    HOST=<Contact NOS-T admins>
    KEYCLOAK_PORT=8443
    KEYCLOAK_REALM=<Contact NOS-T admins>
    RABBITMQ_PORT=5671
    USERNAME=<Your Keycloak username in NOS-T ecosystem>
    PASSWORD=<Your Keycloak password in NOS-T ecosystem>
    CLIENT_ID=<Contact NOS-T admins>
    CLIENT_SECRET_KEY=<Contact NOS-T admins>
    VIRTUAL_HOST="/"
    IS_TLS=True

    # CESIUM LOGIN
    TOKEN=<Cesium access token>
    ```

1. Orchestrate the containers:

```
docker-compose up -d
```

> NOTE: To confirm Docker containers are running, run the command: ```docker ps```. You should see three containers list: sos_manager, sos_satellites, and sos_snow_cover_layer.

1. To shutdown the Docker containers:

```
docker-compose down
```