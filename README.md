# Snow Observing Strategy (SOS)

## Introduction

## Installation

## Execution

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

### Building individual Docker containers (Development)

Each container can be built individually during development, to build a local version of a container, you can use ```docker build```. 

For example, to build the "manager" comntainer:

```
cd src/manager/
docker build -t sos_manager .
docker run --rm --env-file .env -v "$(pwd)/data":/opt/data sos_manager
```

> NOTE: You can follow similar steps to build the other containers, such as satellites, layer_resolution, layer_snow_cover, etc.