import os
import sys
from typing import List, Tuple
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthaccess
import geopandas as gpd
from shapely.geometry import Polygon
from datetime import datetime, timedelta, timezone
import dask as dask
from configparser import ConfigParser
from dotenv import load_dotenv
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from skyfield.api import load, wgs84, EarthSatellite
import logging
import glob
from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.observer import Observer
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher
from constellation_config_files.schemas import SNODASStatus
from constellation_config_files.schemas import SatelliteStatus, SnowLayer, ResolutionLayer, GcomLayer, CapellaLayer
from constellation_config_files.config import PREFIX, NAME, SCALE, TLES, FIELD_OF_REGARD

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

    #--------------------------------------------------
    def efficiency(self, T, k, datarray):
        logger.info("Calculating efficiency.")
        resolution_eta = 1 / (1 + np.exp(k * (datarray - T)))
        logger.info("Calculating efficiency successfully completed.")
        return resolution_eta
    
    def load_parameters(self):
        config = ConfigParser()
        config.read("Input_parameters.ini")
        config_data = config['Resolution']
        return float(config_data['threshold']), float(config_data['coefficient'])
    
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

    def download_snow_data(self, path_hdf: str, start_date: str, end_date: str) -> List[str]:
        """Download snow cover data using earthaccess"""
        logger.info("Downloading snow data.")
        earthaccess.login(strategy="environment")
        results = earthaccess.search_data(
            short_name='MOD10C1',
            temporal=(start_date, end_date)
        )
        logger.info("Downloading snow data successfully completed.")
        return earthaccess.download(results, path_hdf, threads=1)

    def process_snow_files(self, path_hdf: str, path_nc: str) -> Tuple[List[str], List[datetime]]:
        logger.info("Processing snow files.")
        ctr = 0
        lon = np.linspace(-180,180,7200)
        lat = np.flip(np.linspace(-90,90,3600))
        files = []
        time_sc = []

        for filename in os.listdir(path_hdf):    
            year = filename[9:13]
            day = filename[13:16]
            name = filename[0:34]

            # converting day of year to time

            dates = pd.to_datetime(int(day)-1,unit = 'D', origin=year)     
            time_sc.append(dates)
            f_nc = xr.open_dataset(os.path.join(path_hdf, filename),engine = 'netcdf4')  
            snow = f_nc['Day_CMG_Snow_Cover']
            temp_arr = xr.DataArray(
            data=snow,
            dims=['lat','lon'],
            coords=dict(
                lon = lon,
                lat = lat,
            )
            )
            temp_arr.to_netcdf(path_nc + name + ".nc")
        files = glob.glob(os.path.join(path_nc,"*.nc"))
        files = [f for f in files if not f.endswith("snowcover-merged.nc")]
        
        return files, time_sc
    
    def merge_netcdf_files(self, files: List[str], time_sc: List[datetime], path_nc: str) -> xr.Dataset:
        """Merge NetCDF files with dask"""
        logger.info("Merging NetCDF files.")
        output_file = os.path.join(path_nc, "snowcover-merged.nc")

        if not os.path.isfile(output_file):
            logger.info(f"Output file {output_file} does not exist. Merging NetCDF files.")
            ds = xr.combine_by_coords(
                [        
                    rxr.open_rasterio(files[i]).drop_vars("band").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
                    for i in range(len(time_sc))            
                ], 
                combine_attrs="drop_conflicts"
            )
            ds = ds.rio.write_crs("EPSG:4326")
            ds.to_netcdf(output_file)
            logger.info(f"Merging NetCDF files successfully completed.")
        else:
            logger.info(f"Output file {output_file} already exists. Skipping merging.")
        #     ds = xr.open_dataset(output_file)
    
        # return ds
    
    # def get_missouri_basin(self, path_shp: str) -> gpd.GeoSeries:
    #     """Get Missouri Basin geometry"""
        
    #     logger.info("Downloading Missouri Basin geometry.")
    #     us_map = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
    #     mo_basin = gpd.read_file(os.path.join(path_shp, "WBD_10_HU2_Shape/Shape/WBDHU2.shp"))
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

    def compute_weekly_snow_cover(self, snow_layer, mo_basin): #-> xr.Dataset:
        """
        Compute weekly snow cover
        
        Args:
            snow_layer (xr.Dataset): Snow cover dataset
        
        Returns:
            xr.Dataset: Weekly snow cover dataset
        """
        logger.info("Computing weekly snow cover.")
        # Clip the snow layer to the Missouri Basin 
        snow_layer_mo = snow_layer.rio.clip(mo_basin.envelope)

        # Convert the time coordinate to a standard calendar
        snow_layer_mo = snow_layer_mo.convert_calendar(calendar='standard')
        
        # Compute the mean of the snow cover for each week
        temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max()
        temp = temp.to_dataset()

        # Rename the variable
        temp = temp.rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})

        # Resample the data
        temp_resampled = (temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
                          .rio.write_crs("EPSG:4326")
                          .rio.clip(mo_basin.geometry, "EPSG:4326"))
        logger.info("Computing weekly snow cover successfully completed.")
        return temp_resampled

    def process_snow_layer(self) -> xr.Dataset:

        logger.info("Processing MOD10C1 data.")

        # Load Missouri Basin shapefile
        mo_basin = self.get_missouri_basin(file_path=os.path.join(self.path_shp, "WBD_10_HU2_Shape/Shape/WBDHU2.shp"))

        # Open the dataset
        # file_path = os.path.join(self.message.file_path)
        snow_layer = rxr.open_rasterio(self.message.file_path,crs = "EPSG:4326")
        
        # # snow_layer = snow_layer.rio.write_crs("EPSG:4326")  # Ensure CRS is set
        # snow_layer_mo = snow_layer.rio.clip(mo_basin.envelope)
        # snow_layer_mo = snow_layer_mo.convert_calendar(calendar='standard')
        # temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max()
        # temp = temp.to_dataset()
        # temp = temp.rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})
        # temp_resampled = temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
        # temp_resampled = temp_resampled.rio.write_crs("EPSG:4326")
        # temp_resampled = temp_resampled.rio.clip(mo_basin.geometry, "EPSG:4326")
        temp_resampled = self.compute_weekly_snow_cover(snow_layer, mo_basin)

        
        
        # Load parameters
        threshold, coefficient = self.load_parameters()

        dataset = self.efficiency(threshold, coefficient, temp_resampled)
        
        logger.info("Processing snow layer completed successfully.")

        return dataset, temp_resampled
        
    def open_polygons(self, geojson_path):
        logger.info('Loading polygons.')
        geojson = gpd.read_file(geojson_path)
        polygons = geojson.geometry
        logger.info('Loading polygons successfully completed.')
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
        
        if scale == 'time':
            raster_layer = clipped_layer.sel(time=time_step)
        
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
    
    # def on_data(self, ch, method, properties, body):
    #     """
    #     Callback function to check for new data. When callback function is triggered, the input data variable is set to True, which in turn triggers the on_change function.

    #     Args:
    #         ch: Channel
    #         method: Method
    #         properties: Properties
    #         body: Body
        
    #     Returns:
    #         None
    #     """
    #     # Decode the message body
    #     self.message = SNODASStatus.parse_raw(body.decode('utf-8'))
    #     logger.info(f'SNODAS message received: {self.message.publish_time.date()}')

    #     # Process SNODAS data & save the new dataset to a NetCDF file
    #     self.save_dataset(self.process_snodas_data_to_swe_change())

    #     # Set the input data variable to True, which will trigger the on_change callback function
    #     self.input_data_available = True
    #     logger.info("New SNODAS data available. Input data variable set to 'True'.")

    def save_dataset(self, dataset, out_path):
        """
        Save the dataset to a NetCDF file.

        Args:
            dataset (xr.Dataset): The dataset to save
            out_path (str): The path to save the dataset
        """
        logger.info("Saving dataset to NetCDF file.")
        dataset.to_netcdf(out_path)
        logger.info("Saving dataset to NetCDF file successfully completed.")
    
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
        logger.info(f'MOD10C1 message received: {self.message.publish_time.date()}')

        # Process MOD10C1 data
        dataset, temp_resampled = self.process_snow_layer()

        self.filepath = os.path.join(self.path_efficiency, 'efficiency_snow_cover_up.nc')
        self.save_dataset(temp_resampled, os.path.join(self.path_preprocessed, 'preprocessed_snow_cover.nc'))
        self.save_dataset(dataset, self.filepath)
        
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
            change = self.detect_level_change(new_value, old_value, 'week')

            # Publish message if day, week, or month has changed OR if this is the initial layer
            if change or self.initial_layer:
                # Open the NetCDF file
                dataset = self.open_dataset(self.filepath)
                logger.info(dataset)

                # Select the data for the specified date
                snow_layer, top_left, top_right, bottom_left, bottom_right = self.encode(
                    dataset=dataset,
                    # file_path=os.path.join(path_efficiency, 'efficiency_snow_cover_up.nc'),
                    variable='Weekly_Snow_Cover',
                    output_path='snow_raster_layer.png',
                    scale='time',
                    time_step=new_value.date().strftime("%Y-%m-%d"),
                    geojson_path='WBD_10_HU2_4326.geojson')
                
                # Publish the message
                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SnowLayer(
                        snow_layer=snow_layer,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right
                    ).json(),
                )

                self.initial_layer = False
                # # merged_dataset.close()
                # temp_resampled.close()
                # dataset.close()
                # xr.backends.file_manager.FILE_CACHE.clear()

                # # Delete previous results
                # output_file = os.path.join(path_nc, "snowcover-merged.nc")
                # if os.path.exists(output_file):
                #     os.remove(output_file)
                #     logger.info(f"Existing file {output_file} removed.")

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
    app.add_message_callback("mod10c1", "data", environment.on_data)

if __name__ == "__main__":
    main()