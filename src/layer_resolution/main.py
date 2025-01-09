# General
import os
import sys
import glob
import logging
from typing import List, Tuple
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv, dotenv_values
import tarfile
import requests
import gzip
import shutil
import tempfile

# Geospatial & Data Processing
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthaccess
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.enums import Resampling

# Geospatial & Data Processing
import dask as dask
from configparser import ConfigParser
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from skyfield.api import load, wgs84, EarthSatellite

# NOS-T
from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.observer import Observer
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher
from nost_tools.simulator import Simulator, Mode
from constellation_config_files.schemas import SNODASStatus
from constellation_config_files.schemas import SatelliteStatus, SnowLayer, ResolutionLayer, GcomLayer, CapellaLayer
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

    def efficiency(self, T,k,datarray):
        resolution_eta = 1 / (1 + np.exp(k * (datarray - T)))       
        return resolution_eta

    # def load_config(self):
    #     """Load environment variables and config parameters"""
    #     load_dotenv()
    #     paths = {
    #         'shp': os.getenv('path_shp'),
    #         'preprocessed': os.getenv('path_preprocessed'),
    #         'raw': os.getenv('raw_path'),
    #         'efficiency': os.getenv('path_efficiency')
    #     }
        
    #     config = ConfigParser()
    #     config.read("Input_parameters.ini")
    #     config_data = config['Resolution']
        
    #     return paths, float(config_data['threshold']), float(config_data['coefficient'])

    def load_parameters(self):
        config = ConfigParser()
        config.read("Input_parameters.ini")
        config_data = config['Resolution']

        return float(config_data['threshold']), float(config_data['coefficient'])
    
    # def download_snodas_data(self, start_date, end_date, raw_path):
    #     """Download SNODAS data for given date range"""
    #     logger.info("Downloading SNODAS data.")
    #     dates = pd.date_range(start_date, end_date)
    #     snodas_dir = os.path.join(raw_path, "SNODAS")
        
    #     if not os.path.exists(snodas_dir):
    #         os.makedirs(snodas_dir)
            
    #     for date in dates:
    #         file_label = f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001"
            
    #         if os.path.isfile(os.path.join(snodas_dir, file_label + ".nc")):
    #             print(f"Skipping {file_label}")
    #             continue
                
    #         self.process_snodas_date(date, file_label, snodas_dir)
    #     logger.info("Downloading SNODAS data successfully completed.")

    #     return snodas_dir, dates

    # def process_snodas_date(self, date, file_label, snodas_dir):
    #     """Process SNODAS data for a single date"""
    #     dir_label = f"SNODAS_{date.strftime('%Y%m%d')}"
    #     r = requests.get(
    #         "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/" + 
    #         f"{date.strftime('%Y')}/{date.strftime('%m')}_{date.strftime('%b')}/" +
    #         dir_label + ".tar"
    #     )
        
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         self.extract_and_process_tar(r, tmp_dir, dir_label, file_label, date, snodas_dir)

    # def extract_and_process_tar(self, response, tmp_dir, dir_label, file_label, date, snodas_dir):
    #     """Extract and process TAR file contents"""
    #     tar_path = os.path.join(tmp_dir, dir_label + ".tar")
    #     with open(tar_path, "wb") as tar_file:
    #         tar_file.write(response.content)
            
    #     with tarfile.open(tar_path, "r") as tar_file:
    #         tar_file.extractall(tmp_dir)
            
    #     self.process_gz_file(tmp_dir, file_label, date, snodas_dir)

    # def process_gz_file(self, tmp_dir, file_label, date, snodas_dir):
    #     """Process gzipped file and create NetCDF"""
    #     gz_path = os.path.join(tmp_dir, file_label + ".dat.gz")
    #     if os.path.isfile(gz_path):
    #         with gzip.open(gz_path, "rb") as gz_in:
    #             with open(os.path.join(tmp_dir, file_label + ".dat"), "wb") as gz_out:
    #                 shutil.copyfileobj(gz_in, gz_out)
                    
    #         self.write_hdr_file(tmp_dir, file_label)
    #         self.run_gdal_translate(tmp_dir, file_label, date, snodas_dir)

    # def write_hdr_file(self, tmp_dir, file_label):
    #     """Write ENVI header file"""
    #     with open(os.path.join(tmp_dir, file_label + ".hdr"), "w") as hdr_file:
    #         hdr_file.write(
    #             "ENVI\n"
    #             "samples = 6935\n"
    #             "lines = 3351\n"
    #             "bands = 1\n"
    #             "header offset = 0\n"
    #             "file type = ENVI Standard\n"
    #             "data type = 2\n"
    #             "interleave = bsq\n"
    #             "byte order = 1"
    #         )

    # def run_gdal_translate(self, tmp_dir, file_label, date, snodas_dir):
    #     """Run GDAL translate command"""
    #     ullr = ("-a_ullr -124.73375000000000 52.87458333333333 -66.94208333333333 24.94958333333333" 
    #             if date < datetime(2013, 10, 1)
    #             else "-a_ullr -124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000")
        
    #     command = f"gdal_translate -of NetCDF -a_srs EPSG:4326 -a_nodata -9999 {ullr} {os.path.join(tmp_dir, file_label + '.dat')} {os.path.join(snodas_dir, file_label + '.nc')}"
    #     os.system(command)

    # def merge_snodas_files(self, snodas_dir, dates):
    #     """Merge individual SNODAS files into one NetCDF"""
    #     logger.info("Merging NetCDF files.")
    #     ds = xr.combine_by_coords(
    #         [rxr.open_rasterio(
    #             os.path.join(snodas_dir, 
    #                         f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001.nc")
    #         ).drop_vars("band", errors='ignore').assign_coords(time=date).expand_dims(dim="time")
    #         for date in dates], 
    #         combine_attrs="drop_conflicts"
    #     )
    #     ds.to_netcdf(os.path.join(snodas_dir, "snodas-merged.nc"))
    #     logger.info(f"Merging NetCDF files successfully completed.")
    #     return ds

    # def get_missouri_basin(self, path_shp):
    #     """Get Missouri Basin geometry"""

    #     logger.info("Downloading Missouri Basin geometry.")
    #     mo_basin = gpd.read_file(path_shp + "WBD_10_HU2_Shape/Shape/WBDHU2.shp")
    #     logger.info("Downloading Missouri Basin geometry successfully completed.")

    #     return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")
    
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
    
    def process_resolution(self, ds, mo_basin):
        """Process resolution analysis"""
        logger.info("Processing resolution layer.")
        ds_clipped = ds.rio.clip(mo_basin.geometry, "EPSG:4326")
        
        # Remove the singleton 'band' dimension if it exists
        if 'band' in ds_clipped.dims and ds_clipped.dims['band'] == 1:
            logger.info('Removing singleton dimension "band".')
            ds_clipped = ds_clipped.squeeze('band')
        
        # Ensure the data array is 2D or 3D
        if len(ds_clipped.dims) > 3:
            raise ValueError("Only 2D and 3D data arrays supported.")
        
        # Resampling steps
        logger.info("Resampling resolution layer.")
        factor = 5
        h = ds_clipped.rio.height * factor
        w = ds_clipped.rio.width * factor
        ds_5km = ds_clipped.rio.reproject(ds_clipped.rio.crs, shape=(int(h), int(w)), resampling=Resampling.bilinear)
        ds_1km = ds_5km.rio.reproject_match(ds_clipped, 1)
        logger.info("Processing resolution layer completed successfully.")
        return ds_clipped, abs(ds_1km - ds_clipped)

    def compute_monthly_resolution(self, ds_abs, mo_basin):
        """Compute monthly resolution statistics"""
        logger.info("Computing monthly resolution.")
        # Convert the time coordinate to a standard calendar
        resolution_mo = ds_abs.convert_calendar(calendar='standard')

        # Compute the mean of the monthly resolution
        temp = resolution_mo.groupby(resolution_mo.time.dt.month).mean()
        
        # Rename the variable
        temp = temp.rename({'Band1': 'Monthly_Resolution_Abs'})
        
        # Resample the data
        temp_resampled = (temp.sel(month=resolution_mo.time.dt.month)
                        .rio.write_crs("EPSG:4326")
                        .rio.clip(mo_basin.geometry, "EPSG:4326"))
        logger.info("Computing monthly resolution successfully completed.")
        return temp_resampled
    
    def open_polygons(self, geojson_path):
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

        logger.info('Encoding snow layer.')
        polygons = self.open_polygons(geojson_path=geojson_path)
        
        raster_layer = dataset[variable]

        raster_layer = raster_layer.rio.write_crs("EPSG:4326")
        clipped_layer = raster_layer.rio.clip(polygons, all_touched=True)
        print(clipped_layer)
        if scale == 'time':
            raster_layer = clipped_layer.sel(time=time_step)
        elif scale == 'week':
            raster_layer = clipped_layer.isel(week=time_step).values
        elif scale == 'month':
            raster_layer = clipped_layer.isel(month=time_step).values
        
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

        logger.info('Encoding snow layer successfully completed.')

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
    
    def process_snodas_data_to_resolution(self) -> xr.Dataset:

        logger.info("Processing SNODAS data.")

        # Load Missouri Basin shapefile
        mo_basin = self.get_missouri_basin(file_path=os.path.join(self.path_shp, "WBD_10_HU2_Shape/Shape/WBDHU2.shp")) #"WBD_10_HU2.shp"))

        # Open the dataset
        # file_path = os.path.join(self.message.file_path)
        ds = xr.open_dataset(self.message.file_path)

        # Check if the dataset has a coordinate reference system (CRS) and set it to EPSG:4326 if not
        if not ds.rio.crs:
            ds = ds.rio.write_crs("EPSG:4326")
        
        # Process resolution
        ds_clipped, ds_abs = self.process_resolution(ds, mo_basin)
        
        # Compute monthly resolution
        temp_resampled = self.compute_monthly_resolution(ds_abs, mo_basin)
        
        # Process taskable data
        ds_abs_taskable = abs(ds_clipped - ds_clipped)
        temp_resampled_taskable = self.compute_monthly_resolution(ds_abs_taskable, mo_basin)

        # Load parameters
        threshold, coefficient = self.load_parameters()

        # Compute efficiency
        dataset = self.efficiency(threshold, coefficient, temp_resampled)
        
        efficiency_output_taskable = self.efficiency(threshold, coefficient, temp_resampled_taskable)

        return dataset, temp_resampled, efficiency_output_taskable
    
    def save_dataset(self, dataset: xr.Dataset, temp_resampled, efficiency_output_taskable):
        """
        Save the dataset to a NetCDF file
        
        Args:
            dataset (xr.Dataset): Dataset to save
            path_efficiency (str): Path to save the dataset
        
        Returns:
            None
        """
        # Save the datasets to a NetCDF file
        logger.info("Saving datasets to NetCDF file.")
        temp_resampled.to_netcdf(os.path.join(self.path_preprocessed, 'preprocessed_resolution.nc'))
        self.filepath = os.path.join(self.path_efficiency, 'efficiency_resolution.nc')
        dataset.to_netcdf(self.filepath)
        efficiency_output_taskable.to_netcdf(os.path.join(self.path_efficiency, 'efficiency_resolution_taskable.nc'))
        logger.info("Saving dataset to NetCDF file successfully completed.")
        dataset.close()

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

        # Process SNODAS data
        dataset, temp_resampled, efficiency_output_taskable = self.process_snodas_data_to_resolution()
        
        # Save the datasets to a NetCDF file
        self.save_dataset(dataset, temp_resampled, efficiency_output_taskable)
        
        # Set the input data variable to True, which will trigger the on_change callback function
        self.input_data_available = True
        logger.info("Input data now available, variable set to 'True'.")
        


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
            change = self.detect_level_change(new_value, old_value, 'month')

            # Publish message if day, week, or month has changed OR if this is the initial layer
            if change or self.initial_layer:
                # Open the NetCDF file
                dataset = self.open_dataset(self.filepath)
                logger.info(dataset)

                # Select the data for the specified date
                resolution_layer, top_left, top_right, bottom_left, bottom_right = self.encode(
                        dataset=dataset,
                        # file_path=os.path.join(path_efficiency, 'efficiency_snow_cover_up.nc'),
                        variable='Monthly_Resolution_Abs',
                        output_path='resolution_raster_layer.png',
                        scale='time',
                        time_step=new_value.date().strftime("%Y-%m-%d"),
                        geojson_path='WBD_10_HU2_4326.geojson')
                
                # Publish the message
                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    ResolutionLayer(
                        resolution_layer=resolution_layer,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right
                    ).json(),
                )

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
    IS_TLS = credentials["IS_TLS"].lower() == 'true'
    # PREFIX = "sos"
    # NAME = "resolution"
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

if __name__ == "__main__":
    main()