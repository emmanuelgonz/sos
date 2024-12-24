# FROM: https://github.com/code-lab-org/sos/blob/H-Update/src/layer_swe_change/main.py
import os
import sys
import logging
from typing import List, Tuple
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv, dotenv_values

# Geospatial
import geopandas as gpd
from shapely import Polygon
import xarray as xr
import numpy as np
import rioxarray

# NOS-T
from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.observer import Observer
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher
from nost_tools.simulator import Simulator, Mode

from constellation_config_files.schemas import SNODASStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Define an observer class to monitor the simulation environment
class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*
    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """
    def __init__(self, app): #, grounds):
        self.app = app
        # self.grounds = grounds

    def categorize_dates(self, start_date, end_date):
        # Initialize dictionaries to hold weeks and months
        weeks = {}
        months = {}

        # Convert start and end dates to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Iterate through each date in the range
        current_date = start_date
        while current_date <= end_date:
            # Get the week number and month name
            week_number = current_date.strftime("%Y-W%U")
            month_name = current_date.strftime("%Y-%m")

            # Add the date to the respective week and month
            if week_number not in weeks:
                weeks[week_number] = []
            if month_name not in months:
                months[month_name] = []

            weeks[week_number].append(current_date.strftime("%Y-%m-%d"))
            months[month_name].append(current_date.strftime("%Y-%m-%d"))

            # Move to the next date
            current_date += timedelta(days=1)

        return weeks, months

    def on_change(self, source, property_name, old_value, new_value):
        """
        *Standard on_change callback function format inherited from Observer object class*
        In this instance, the callback function checks when the **PROPERTY_MODE** switches to **EXECUTING** to send a :obj:`GroundLocation` message to the *PREFIX/ground/location* topic:
            .. literalinclude:: /../../NWISdemo/grounds/main_ground.py
                :lines: 56-67
        """
        # logger.info(f'>>>SOURCE: {source.get_time_scale_factor()}')
        # logger.info(f'>>>PROPERTY NAME: {property_name}')
        if property_name == 'time':
            # logger.info(f'>>>SOURCE: {source}')
            # logger.info(f'>>>PROPERTY NAME: {property_name}')
            # logger.info(f'>>>OLD VALUE: {old_value}')
            # logger.info(f'>>>NEW VALUE: {new_value}')
            # logger.info(f'>>>START TIME: {source._init_time}')
            # logger.info(f'>>>END TIME: {source.get_end_time()}')
            weeks, months = self.categorize_dates(str(source._init_time.date()), str(source.get_end_time().date()))
            logger.info(f'>>>WEEKS: {weeks}')
            logger.info(f'>>>MONTHS: {months}')
        # if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
        #     logger.info(f'ON CHANGE::::: {property_name}, {new_value}, {old_value}')
            # for index, ground in self.grounds.iterrows():
            #     self.app.send_message(
            #         self.app.app_name,
            #         "location",
            #         GroundLocation(
            #             groundId=ground.groundId,
            #             latitude=ground.latitude,
            #             longitude=ground.longitude,
            #             elevAngle=ground.elevAngle,
            #             operational=ground.operational,
            #         ).json(),
            #     )

def load_config() -> Tuple[str, ...]:
    """
    Load environment variables and return paths
    
    Returns:
        Tuple[str, ...]: Tuple of paths for the HDF, NetCDF, shapefile, preprocessed, efficiency, and SNODAS directories
    """
    load_dotenv()
    return (
        os.getenv('path_hdf'),
        os.getenv('path_nc'), 
        os.getenv('path_shp'),
        os.getenv('path_preprocessed'),
        os.getenv('path_efficiency'),
        os.getenv('path_snodas')
    )

def get_missouri_basin(file_path: str) -> gpd.GeoSeries:
    """
    Get Missouri Basin geometry
    
    Args:
        file_path (str): File path to the shapefile
    
    Returns:
        gpd.GeoSeries: GeoSeries of the Missouri Basin geometry
    """
    mo_basin = gpd.read_file(file_path)
    return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

def calculate_beta5(swe_change, threshold=10, k_value=0.2):
    """
    Calculate beta5 values based on the logistic function
    
    Args:
        swe_change (xr.DataArray): Absolute SWE change values
        threshold (float): Threshold value for the logistic function
        k_value (float): Scaling factor for the logistic function
    
    Returns:
        xr.DataArray: Beta5 values based on the logistic function    
    """
    return 1 / (1 + np.exp(-k_value * (swe_change - threshold)))

def process_snodas_data(path_merged_snodas: str, path_shp: str) -> xr.Dataset:
    """
    Process SNODAS data and return a dataset with beta5 values and SWE differences
    
    Args:
        path_snodas (str): Path to the SNODAS dataset
        path_shp (str): Path to the shapefile
    
    Returns:
        xr.Dataset: Dataset containing beta5 values and SWE differences
    """
    logger.info("Processing SNODAS data.")
    # Load the shapefile
    mo_basin = get_missouri_basin(file_path=os.path.join(path_shp, "WBD_10_HU2.shp"))

    # Load the SNODAS dataset
    file_path = os.path.join(path_merged_snodas)
    ds = xr.open_dataset(file_path)

    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")

    # Clip dataset to Missouri River Basin
    masked_ds = ds.rio.clip(mo_basin.geometry, mo_basin.crs)

    # Compute SWE values from the clipped dataset
    swe = masked_ds['Band1']

    # Squeeze any dimensions of size 1, especially for 'band'
    if 'band' in swe.dims and swe.sizes['band'] == 1:
        swe = swe.squeeze('band')

    # Check the range of SWE values to verify non-zero values
    # print("SWE min:", swe.min().values, "SWE max:", swe.max().values)

    # Mask NaN and zero values before applying the difference calculation
    swe_masked = swe.where(~np.isnan(swe))

    # Calculate the SWE difference between consecutive time steps, keeping NaN values intact
    swe_diff_abs = swe_masked.diff(dim='time').where(~np.isnan(swe_masked.diff(dim='time')))

    # Set NaN values for zero differences or areas with no changes
    swe_diff_abs = abs(swe_diff_abs).where(swe_diff_abs != 0, np.nan)

    # Add a zero difference for the first time step to match the length
    swe_diff_abs = xr.concat([xr.zeros_like(swe.isel(time=0)), swe_diff_abs], dim='time')

    # Apply the beta5 calculation to SWE changes, keeping NaN values
    beta5_values = calculate_beta5(swe_diff_abs)

    # Replace NaN values with 1 in beta5
    beta5_values = beta5_values.fillna(1)

    # Create the DataArray for beta5 values
    beta5_da = xr.DataArray(
        beta5_values,
        coords={
            'time': swe['time'],
            'y': swe['y'],
            'x': swe['x']
        },
        dims=swe_diff_abs.dims,
        name='beta5'
    )

    # Create a new dataset with beta5 values and the absolute SWE difference
    new_ds = xr.Dataset({
        'beta5': beta5_da,
        'swe_diff_abs': swe_diff_abs
    })

    # Transpose the dataset to ensure 'time' is the first dimension
    new_ds = new_ds.transpose('time', 'y', 'x')

    # Remove 'grid_mapping' attribute if it exists in the dataset
    for var in new_ds.variables:
        if 'grid_mapping' in new_ds[var].attrs:
            del new_ds[var].attrs['grid_mapping']

    # Close the datasets
    ds.close()
    masked_ds.close()

    logger.info("Processing SNODAS data successfully completed.")

    return new_ds

def save_dataset(dataset: xr.Dataset, path_efficiency: str):
    """
    Save the dataset to a NetCDF file
    
    Args:
        dataset (xr.Dataset): Dataset to save
        path_efficiency (str): Path to save the dataset
    """
    logger.info("Saving dataset to NetCDF file.")
    dataset.to_netcdf(os.path.join(path_efficiency, 'Efficiency_SWE_Change_dataset.nc'))
    logger.info("Saving dataset to NetCDF file successfully completed.")
    dataset.close()

def on_data(ch, method, properties, body):
    """
    Callback function appends a dictionary of information for a new fire to fires :obj:`list` when message detected on the *PREFIX/fires/location* topic
    Args:
        client (:obj:`MQTT Client`): Client that connects application to the event broker using the MQTT protocol. Includes user credentials, tls certificates, and host server-port information.
        userdata: User defined data of any type (not currently used)
        message (:obj:`message`): Contains *topic* the client subscribed to and *payload* message content as attributes
    """
    body = body.decode('utf-8')
    data = SNODASStatus.parse_raw(body)
    logger.info(f'SNODAS message received: {data.publish_time.date()}')
    # data.file_path
    # data.publish_time
    # data.start_date
    # data.end_date

    # Load configurations
    path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency, path_snodas = load_config()

    # Process SNODAS data
    new_ds = process_snodas_data(data.file_path, path_shp)

    # Save the new dataset to a NetCDF file
    save_dataset(new_ds, path_efficiency)

    # self.app.send_message(
    #     self.app.app_name,
    #     "data",
    #     SNODASStatus(
    #         file_path=output_path,
    #         publish_time=datetime.now(timezone.utc),
    #         start_date=start_time,
    #         end_date=stop_time,
    #     ).json(),
    # )

def main():
    # Load credentials from a .env file in current working directory
    credentials = dotenv_values(".env")
    HOST, RABBITMQ_PORT, KEYCLOAK_PORT, KEYCLOAK_REALM = credentials["HOST"], int(credentials["RABBITMQ_PORT"]), int(credentials["KEYCLOAK_PORT"]), str(credentials["KEYCLOAK_REALM"])
    USERNAME, PASSWORD = credentials["USERNAME"], credentials["PASSWORD"]
    CLIENT_ID = credentials["CLIENT_ID"]
    CLIENT_SECRET_KEY = credentials["CLIENT_SECRET_KEY"]
    VIRTUAL_HOST = credentials["VIRTUAL_HOST"]
    IS_TLS = credentials["IS_TLS"].lower() == 'true'  # Convert to boolean
    PREFIX = "sos"
    NAME = "response"
    SCALE = 1

    # Set the client credentials from the config file
    config = ConnectionConfig(
        USERNAME,
        PASSWORD,
        HOST,
        RABBITMQ_PORT,
        KEYCLOAK_PORT,
        KEYCLOAK_REALM,
        CLIENT_ID,
        CLIENT_SECRET_KEY,
        VIRTUAL_HOST,
        IS_TLS)
    # create the managed application
    app = ManagedApplication(NAME) #, config=config)
    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app)) #, GROUND))
    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))
    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        PREFIX,
        config,
        True,
        time_status_step=timedelta(seconds=10) * SCALE,
        time_status_init=datetime(2024, 1, 7, tzinfo=timezone.utc),
        time_step=timedelta(seconds=1) * SCALE,
        # shut_down_when_terminated=True,
    )
    app.add_message_callback("snodas", "data", on_data)
    # while True:
    #     pass
if __name__ == "__main__":
    main()