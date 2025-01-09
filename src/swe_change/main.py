# FROM: https://github.com/code-lab-org/sos/blob/H-Update/src/layer_swe_change/main.py

# General
import os
import sys
import logging
from typing import List, Tuple
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv, dotenv_values

# Geospatial & Data Processing
import geopandas as gpd
from shapely import Polygon
import xarray as xr
import numpy as np
import rioxarray
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# NOS-T
from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.observer import Observer
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher
from nost_tools.simulator import Simulator, Mode
from constellation_config_files.schemas import SNODASStatus
from constellation_config_files.schemas import SatelliteStatus, SnowLayer, ResolutionLayer, GcomLayer, CapellaLayer, SWEChangeLayer
from constellation_config_files.config import PREFIX, NAME, SCALE

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
    def __init__(self, app):
        self.app = app
        self.input_data_available = False
        self.message = None
        self.filepath = None
        self.initial_layer = True
        self.path_hdf, self.path_nc, self.path_shp, self.path_preprocessed, self.path_efficiency, self.path_snodas = self.load_config()
        
    def load_config(self) -> Tuple[str, ...]:
        """
        Load environment variables and return paths

        Args:
            None
        
        Returns:
            Tuple[str, ...]: Tuple of paths for the HDF, NetCDF, shapefile, preprocessed, efficiency, and SNODAS directories
        """
        # Load environment variables
        load_dotenv()
        return (
            os.getenv('path_hdf'),
            os.getenv('path_nc'), 
            os.getenv('path_shp'),
            os.getenv('path_preprocessed'),
            os.getenv('path_efficiency'),
            os.getenv('path_snodas')
        )

    def get_missouri_basin(self, file_path: str) -> gpd.GeoSeries:
        """
        Get Missouri Basin geometry
        
        Args:
            file_path (str): File path to the shapefile
        
        Returns:
            gpd.GeoSeries: GeoSeries of the Missouri Basin geometry
        """
        # Load the shapefile
        mo_basin = gpd.read_file(file_path)
        return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

    def calculate_beta5(self, swe_change, threshold=10, k_value=0.2):
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

    def process_snodas_data_to_swe_change(self) -> xr.Dataset:
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
        mo_basin = self.get_missouri_basin(file_path=os.path.join(self.path_shp, "WBD_10_HU2.shp"))

        # Load the SNODAS dataset
        file_path = os.path.join(self.message.file_path)
        ds = xr.open_dataset(file_path)

        # Check if the dataset has a coordinate reference system (CRS) and set it to EPSG:4326 if not
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
        beta5_values = self.calculate_beta5(swe_diff_abs)

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

    def save_dataset(self, dataset: xr.Dataset):
        """
        Save the dataset to a NetCDF file
        
        Args:
            dataset (xr.Dataset): Dataset to save
        
        Returns:
            None
        """
        # Save the dataset to a NetCDF file
        logger.info("Saving dataset to NetCDF file.")
        self.filepath = os.path.join(self.path_efficiency, 'Efficiency_SWE_Change_dataset.nc')
        dataset.to_netcdf(self.filepath)
        logger.info("Saving dataset to NetCDF file successfully completed.")
        dataset.close()

    def open_polygons(self, geojson_path):
        """
        Open the polygons from the GeoJSON file.

        Args:
            geojson_path (str): The path to the GeoJSON file.
        
        Returns:
            gpd.GeoSeries: The GeoSeries containing the polygons
        """
        geojson = gpd.read_file(geojson_path)
        polygons = geojson.geometry
        print('Polygons loaded.')
        return polygons
    
    def downsample_array(self, array, downsample_factor):
        """
        Downsamples the given array by the specified factor.

        Args:
            array (np.ndarray): The array to downsample.
            downsample_factor (int): The factor by which to downsample the array.

        Returns:
            np.ndarray: The downsampled array.
        """
        return array[::downsample_factor, ::downsample_factor]
    
    def get_extents(self, dataset, variable):
        """
        Get the extents of the raster layer.

        Args:
            dataset (xr.Dataset): The dataset containing the raster layer.
            variable (str): The variable name of the raster layer.
        
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]: The coordinates of the four corners of the raster layer.
        """
        # Extract the GeoTransform attribute
        geo_transform = dataset['spatial_ref'].GeoTransform.split()
        # Convert GeoTransform values to float
        geo_transform = [float(value) for value in geo_transform]
        # Calculate the extents (four corners)
        min_x = geo_transform[0]
        pixel_width = geo_transform[1]
        max_y = geo_transform[3]
        pixel_height = geo_transform[5]
        # Get the actual dimensions of the raster layer
        n_rows, n_cols = dataset[variable][0, :, :].shape
        # Calculate the coordinates of the four corners
        top_left = (min_x, max_y)
        top_right = (min_x + n_cols * pixel_width, max_y)
        bottom_left = (min_x, max_y + n_rows * pixel_height)
        bottom_right = (min_x + n_cols * pixel_width, max_y + n_rows * pixel_height)
        return top_left, top_right, bottom_left, bottom_right
    
    def encode(self, dataset, variable, output_path, time_step, scale, geojson_path, downsample_factor=1):
        """
        Encode the snow layer as a PNG image.

        Args:
            dataset (xr.Dataset): The dataset containing the raster layer.
            variable (str): The variable name of the raster layer.
            output_path (str): The path to save the encoded PNG image.
            time_step (str): The time step to encode.
            scale (str): The scale of the time step ('time', 'week', or 'month').
            geojson_path (str): The path to the GeoJSON file.
            downsample_factor (int): The factor by which to downsample the array.
        
        Returns:
            str: The base64 encoded PNG image.
        """
        logger.info('Encoding layer.')
        polygons = self.open_polygons(geojson_path=geojson_path)
        
        raster_layer = dataset[variable]

        raster_layer = raster_layer.rio.write_crs("EPSG:4326")
        clipped_layer = raster_layer.rio.clip(polygons, all_touched=True)

        try:
            raster_layer = clipped_layer.sel(time=time_step)
        except KeyError:
            try:
                raster_layer = clipped_layer.isel(week=time_step)
            except KeyError:
                raster_layer = clipped_layer.isel(month=time_step)
        
        raster_layer = self.downsample_array(raster_layer, downsample_factor=downsample_factor)

        raster_layer_min = np.nanmin(raster_layer)
        raster_layer_max = np.nanmax(raster_layer)

        na_mask = np.isnan(raster_layer)

        if raster_layer_max > raster_layer_min:
            normalized_layer = (raster_layer - raster_layer_min) / (raster_layer_max - raster_layer_min)
        else:
            normalized_layer = np.zeros_like(raster_layer)

        colormap = plt.get_cmap('Blues_r')
        rgba_image = colormap(normalized_layer)

        rgba_image[..., 3] = np.where(na_mask, 0, 1)

        rgba_image = (rgba_image * 255).astype(np.uint8)

        image = Image.fromarray(rgba_image, 'RGBA')

        image.save(output_path)

        top_left, top_right, bottom_left, bottom_right = self.get_extents(dataset, variable=variable)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        raster_layer_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')

        logger.info('Encoding layer successfully completed.')

        return raster_layer_encoded, top_left, top_right, bottom_left, bottom_right
    
    def open_dataset(self, path):
        """
        Open the dataset from the specified path.

        Args:
            path (str): The path to the dataset.
        
        Returns:
            xr.Dataset: The opened dataset.
        """
        return xr.open_dataset(path)
    
    def detect_level_change(self, new_value, old_value, level):
        """
        Detect a change in the level of the time value (day, week, or month).

        Args:
            new_value (datetime): New time value
            old_value (datetime): Old time value
            level (str): Level of time value to detect changes ('day', 'week', or 'month')
        
        Returns:
            bool: True if the level has changed, False otherwise
        """
        # Check if the level is 'day', 'week', or 'month'. Otherwise, raise an error. If change in 'day', 'week', or 'month' is detected, return True. Otherwise, return False.
        if level == 'day':
            return new_value.date() != old_value.date()
        elif level == 'week':
            return new_value.isocalendar()[1] != old_value.isocalendar()[1]
        elif level == 'month':
            return new_value.month != old_value.month
        else:
            raise ValueError("Invalid level. Choose from 'day', 'week', or 'month'.")
        
    def on_data(self, ch, method, properties, body):
        """
        Callback function to check for new data. When callback function is triggered, the input data variable is set to True, which in turn triggers the on_change function.

        Args:
            ch: Channel
            method: Method
            properties: Properties
            body: Body
        
        Returns:
            None
        """
        # Decode the message body
        self.message = SNODASStatus.parse_raw(body.decode('utf-8'))
        logger.info(f'SNODAS message received: {self.message.publish_time.date()}')

        # Process SNODAS data & save the new dataset to a NetCDF file
        self.save_dataset(self.process_snodas_data_to_swe_change())

        # Set the input data variable to True, which will trigger the on_change callback function
        self.input_data_available = True
        logger.info("New SNODAS data available. Input data variable set to 'True'.")

    def on_change(self, source, property_name, old_value, new_value):
        """
        *Standard on_change callback function format inherited from Observer object class*
        In this instance, the callback function is triggered by the on_data function. This callback function monitors changes in the time property and publishes a message if there is a change in the day, week, or month.

        Args:
            source: Source object
            property_name: Name of the property
            old_value: Old value of the property
            new_value: New value of the property
        
        Returns:
            None
        """
        if (property_name == 'time') and (self.input_data_available == True):

            # Determine if day has changed
            change = self.detect_level_change(new_value, old_value, 'day')

            # Publish message if day, week, or month has changed
            if change or self.initial_layer:
                # Open the NetCDF file
                dataset = self.open_dataset(self.filepath)

                # Select the data for the specified date
                swe_change_layer, top_left, top_right, bottom_left, bottom_right = self.encode(
                                dataset=dataset,
                                variable='swe_diff_abs',
                                output_path='swe_diff_raster_layer.png',
                                scale='time',
                                time_step=new_value.date().strftime("%Y-%m-%d"),
                                geojson_path='WBD_10_HU2_4326.geojson')
                
                # logger.info(resolution_layer)
                # logger.info(f'{top_left}, {top_right}, {bottom_left}, {bottom_right}') 
                
                # Publish the message
                logger.info(f'Publishing message for {new_value}')
                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=swe_change_layer,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right
                    ).json(),
                )
                logger.info('Publishing message successfully completed.')

                self.initial_layer = False

def main():
    """
    Main function to start the application and run the simulation.
    """
    # Load credentials from a .env file in current working directory
    credentials = dotenv_values(".env")
    HOST, RABBITMQ_PORT, KEYCLOAK_PORT, KEYCLOAK_REALM = credentials["HOST"], int(credentials["RABBITMQ_PORT"]), int(credentials["KEYCLOAK_PORT"]), str(credentials["KEYCLOAK_REALM"])
    USERNAME, PASSWORD = credentials["USERNAME"], credentials["PASSWORD"]
    CLIENT_ID = credentials["CLIENT_ID"]
    CLIENT_SECRET_KEY = credentials["CLIENT_SECRET_KEY"]
    VIRTUAL_HOST = credentials["VIRTUAL_HOST"]
    IS_TLS = credentials["IS_TLS"].lower() == 'true'  # Convert to boolean
    # PREFIX = "sos"
    # NAME = "response"
    # SCALE = 1

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
    environment = Environment(app)
    app.simulator.add_observer(environment) #, GROUND))
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
    # app.add_message_callback("snodas", "data", on_data)
    app.add_message_callback("snodas", "data", environment.on_data)

    # while True:
    #     pass
if __name__ == "__main__":
    main()