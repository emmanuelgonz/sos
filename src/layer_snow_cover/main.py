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
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher

from constellation_config_files.schemas import SatelliteStatus, SnowLayer, ResolutionLayer, GcomLayer, CapellaLayer
from constellation_config_files.config import PREFIX, NAME, SCALE, TLES, FIELD_OF_REGARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class Constellation(Entity):
    ts = load.timescale()
    PROPERTY_POSITION = "position"

    def __init__(self, cName):
        super().__init__(cName)

    def initialize(self, init_time):
        super().initialize(init_time)

class LayerPublisher(WallclockTimeIntervalPublisher):
    def __init__(
            self, app, constellation, time_status_step=None, time_status_init=None,
            snow_layer=None, resolution_layer=None, gcom_layer=None, capella_layer=None,
            top_left=None, top_right=None, bottom_left=None, bottom_right=None):
        super().__init__(app, time_status_step, time_status_init)
        self.constellation = constellation
        self.snow_layer = snow_layer
        self.resolution_layer = resolution_layer
        self.gcom_layer = gcom_layer
        self.capella_layer = capella_layer
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right

        if self.time_status_init is None:
            self.time_status_init = self.constellation.ts.now().utc_datetime()
            print(f'TIME STATUS INIT: {self.time_status_init}')
        else:
            self.time_status_init = self.time_status_init
            print(f'TIME STATUS INIT: {self.time_status_init}')
    
    def open_netcdf(self, file_path):
        # Open the NetCDF file
        dataset = xr.open_dataset(file_path)
        return dataset
    
    def efficiency(self, T, k, datarray):
        logger.info("Calculating efficiency.")
        resolution_eta = 1 / (1 + np.exp(k * (datarray - T)))
        logger.info("Calculating efficiency successfully completed.")
        return resolution_eta

    def load_config(self) -> Tuple[str, ...]:
        """Load environment variables and return paths"""
        load_dotenv()
        return (
            os.getenv('path_hdf'),
            os.getenv('path_nc'), 
            os.getenv('path_shp'),
            os.getenv('path_preprocessed'),
            os.getenv('path_efficiency')
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
        """Process HDF files to NetCDF with dask parallelization"""
        logger.info("Processing snow files.")
        lon = np.linspace(-180, 180, 7200)
        lat = np.flip(np.linspace(-90, 90, 3600))
        files = []
        time_sc = []

        for filename in os.listdir(path_hdf):    
            year = filename[9:13]
            day = filename[13:16]
            name = filename[0:34]

            # converting day of year to time
            dates = pd.to_datetime(int(day)-1, unit='D', origin=year)     
            time_sc.append(dates)
            
            # Using context manager to ensure the file is closed
            with xr.open_dataset(os.path.join(path_hdf, filename), engine='netcdf4') as f_nc:
                snow = f_nc['Day_CMG_Snow_Cover']
                temp_arr = xr.DataArray(
                    data=snow,
                    dims=['lat', 'lon'],
                    coords=dict(
                        lon=lon,
                        lat=lat,
                    )
                )
                temp_arr.to_netcdf(os.path.join(path_nc, name + ".nc"))
            
            files = glob.glob(os.path.join(path_nc, "*.nc"))
            logger.info("Processing snow files successfully completed.")
                    
        return files, time_sc
    # def process_snow_files(self, path_hdf: str, path_nc: str) -> Tuple[List[str], List[datetime]]:
    #     """Process HDF files to NetCDF with dask parallelization"""
    #     logger.info("Processing snow files.")
    #     lon = np.linspace(-180, 180, 7200)
    #     lat = np.flip(np.linspace(-90, 90, 3600))
    #     files = []
    #     time_sc = []

    #     for filename in os.listdir(path_hdf):    
    #         year = filename[9:13]
    #         day = filename[13:16]
    #         name = filename[0:34]
    #         nc_file_path = os.path.join(path_nc, name + ".nc")

    #         # Check if the NetCDF file already exists
    #         if os.path.exists(nc_file_path):
    #             logger.info(f"File {nc_file_path} already exists. Skipping processing.")
    #             continue

    #         # Converting day of year to time
    #         dates = pd.to_datetime(int(day)-1, unit='D', origin=year)     
    #         time_sc.append(dates)
            
    #         # Using context manager to ensure the file is closed
    #         with xr.open_dataset(os.path.join(path_hdf, filename), engine='netcdf4') as f_nc:
    #             snow = f_nc['Day_CMG_Snow_Cover']
    #             temp_arr = xr.DataArray(
    #                 data=snow,
    #                 dims=['lat', 'lon'],
    #                 coords=dict(
    #                     lon=lon,
    #                     lat=lat,
    #                 )
    #             )
    #             temp_arr.to_netcdf(nc_file_path)
            
    #         files = glob.glob(os.path.join(path_nc, "*.nc"))
    #         logger.info("Processing snow files successfully completed.")
                    
    #     return files, time_sc
    # def process_snow_files(self, path_hdf: str, path_nc: str) -> Tuple[List[str], List[datetime]]:
    #     """Process HDF files to NetCDF with dask parallelization"""
    #     logger.info("Processing snow files.")
    #     lon = np.linspace(-180, 180, 7200)
    #     lat = np.flip(np.linspace(-90, 90, 3600))
    #     time_sc = []

    #     for filename in os.listdir(path_hdf):    
    #         year = filename[9:13]
    #         day = filename[13:16]
    #         name = filename[0:34]
    #         nc_file_path = os.path.join(path_nc, name + ".nc")

    #         # Check if the NetCDF file already exists
    #         if os.path.exists(nc_file_path):
    #             logger.info(f"File {nc_file_path} already exists. Skipping processing.")
    #             continue

    #         # Converting day of year to time
    #         dates = pd.to_datetime(int(day)-1, unit='D', origin=year)     
    #         time_sc.append(dates)
            
    #         # Using context manager to ensure the file is closed
    #         with xr.open_dataset(os.path.join(path_hdf, filename), engine='netcdf4') as f_nc:
    #             snow = f_nc['Day_CMG_Snow_Cover']
    #             temp_arr = xr.DataArray(
    #                 data=snow,
    #                 dims=['lat', 'lon'],
    #                 coords=dict(
    #                     lon=lon,
    #                     lat=lat,
    #                 )
    #             )
    #             temp_arr.to_netcdf(nc_file_path)
        
    #     # Collect all NetCDF files in the directory after processing
    #     files = glob.glob(os.path.join(path_nc, "*.nc"))
    #     files = [file for file in files if "snowcover-merged.nc" not in file]
    #     logger.info("Processing snow files successfully completed.")
                    
    #     return files, time_sc
    
    # def merge_netcdf_files(self, files: List[str], time_sc: List[datetime], path_nc: str) -> xr.Dataset:
    #     """Merge NetCDF files with dask"""
    #     logger.info(f"Merging NetCDF files: {files}")
    #     xr.backends.file_manager.FILE_CACHE.clear()
    #     # ds.close()
    #     ds = xr.combine_by_coords(
    #         [        
    #             rxr.open_rasterio(files[i]).drop_vars("band").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
    #             for i in range(len(time_sc))            
                
    #         ], 
    #         combine_attrs="drop_conflicts"
    #     )
    #     ds = ds.rio.write_crs("EPSG:4326")
    #     ds.to_netcdf(os.path.join(path_nc, "snowcover-merged.nc"))
    #     logger.info(f"Merging NetCDF files successfully completed.")
    #     logger.info(ds)

    #     return ds

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

    # def merge_netcdf_files(self, files: List[str], time_sc: List[datetime], path_nc: str) -> xr.Dataset:
    #     """Merge NetCDF files with dask"""
    #     logger.info(f"Merging NetCDF files: {files}.")
    #     output_file = os.path.join(path_nc, "snowcover-merged.nc")
        
    #     # Check if the output file already exists and delete it
    #     if os.path.exists(output_file):
    #         os.remove(output_file)
    #         logger.info(f"Existing file {output_file} removed.")
        
    #     ds = xr.combine_by_coords(
    #         [        
    #             rxr.open_rasterio(files[i]).drop_vars("band").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
    #             for i in range(len(time_sc))            
    #         ], 
    #         combine_attrs="drop_conflicts"
    #     )
    #     ds = ds.rio.write_crs("EPSG:4326")
    #     ds.to_netcdf(output_file)
    #     logger.info(f"Merging NetCDF files successfully completed.")
    #     logger.info(ds)
        
    #     # Verify the file is created and readable
    #     if os.path.exists(output_file):
    #         logger.info(f"Output file {output_file} created successfully.")
    #         try:
    #             test_ds = xr.open_dataset(output_file)
    #             logger.info(f"Output file {output_file} is readable and contains: {test_ds}")
    #             test_ds.close()
    #         except Exception as e:
    #             logger.error(f"Failed to read the output file {output_file}: {e}")
    #     else:
    #         logger.error(f"Output file {output_file} was not created.")
        
    #     return ds
    
    def get_missouri_basin(self, path_shp: str) -> gpd.GeoSeries:
        """Get Missouri Basin geometry"""
        
        logger.info("Downloading Missouri Basin geometry.")
        us_map = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
        mo_basin = gpd.read_file(os.path.join(path_shp, "WBD_10_HU2_Shape/Shape/WBDHU2.shp"))
        logger.info("Downloading Missouri Basin geometry successfully completed.")

        return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

    def process_snow_layer(self, path_nc: str, mo_basin: gpd.GeoSeries, path_preprocessed: str) -> xr.Dataset:

        logger.info("Processing snow layer.")
        snow_layer = rxr.open_rasterio(os.path.join(path_nc,"snowcover-merged.nc"),crs = "EPSG:4326")
        snow_layer_mo = snow_layer.rio.clip(mo_basin.envelope)
        snow_layer_mo = snow_layer_mo.convert_calendar(calendar='standard')
        temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max()
        temp = temp.to_dataset()
        temp = temp.rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})
        temp_resampled = temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
        temp_resampled = temp_resampled.rio.write_crs("EPSG:4326")
        temp_resampled = temp_resampled.rio.clip(mo_basin.geometry, "EPSG:4326")
        temp_resampled.to_netcdf(path_preprocessed + 'preprocessed_snow_cover' + '.nc')
        logger.info("Processing snow layer completed successfully.")

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
           
    def publish_message(self):

        current_time = self.constellation.get_time()

        # Load configurations
        path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency = self.load_config()

        # Download data
        self.download_snow_data(path_hdf, "2024.01.01", "2024.02.02")
        
        # Process files
        files, time_sc = self.process_snow_files(path_hdf, path_nc)
        
        # Merge files
        # merged_dataset = self.merge_netcdf_files(files, time_sc, path_nc)
        self.merge_netcdf_files(files, time_sc, path_nc)

        # Get Missouri Basin
        mo_basin = self.get_missouri_basin(path_shp)

        # Process snow layer
        temp_resampled = self.process_snow_layer(path_nc, mo_basin, path_preprocessed)

        # Compute efficiency
        config = ConfigParser()
        config.read("Input_parameters.ini")
        config_data = config['Snow_cover']
        T = float(config_data['threshold'])
        k = -float(config_data['coefficient'])
        
        logger.info(path_efficiency)
        dataset = self.efficiency(T, k, temp_resampled)
        dataset.to_netcdf(os.path.join(path_efficiency, 'efficiency_snow_cover_up.nc'))
        logger.info("Efficiency calculation successfully completed.")

        snow_layer, top_left, top_right, bottom_left, bottom_right = self.encode(
            dataset=dataset,
            # file_path=os.path.join(path_efficiency, 'efficiency_snow_cover_up.nc'),
            variable='Weekly_Snow_Cover',
            output_path='snow_raster_layer.png',
            scale='time',
            time_step=str(current_time.date()),
            geojson_path='WBD_10_HU2_4326.geojson')
        
        self.app.send_message(
            self.app.app_name,
            "snow_layer",
            SnowLayer(
                snow_layer=snow_layer,
                top_left=top_left,
                top_right=top_right,
                bottom_left=bottom_left,
                bottom_right=bottom_right
            ).json(),
        )
        # merged_dataset.close()
        temp_resampled.close()
        dataset.close()
        xr.backends.file_manager.FILE_CACHE.clear()

        # Delete previous results
        output_file = os.path.join(path_nc, "snowcover-merged.nc")
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info(f"Existing file {output_file} removed.")
        
# def download_snow_data(path_hdf: str, start_date: str, end_date: str) -> List[str]:
#     """Download snow cover data using earthaccess"""
#     logger.info("Downloading snow data.")
#     earthaccess.login(strategy="environment")
#     results = earthaccess.search_data(
#         short_name='MOD10C1',
#         temporal=(start_date, end_date)
#     )
#     logger.info("Downloading snow data successfully completed.")
#     return earthaccess.download(results, path_hdf)

def load_config() -> Tuple[str, ...]:
    """Load environment variables and return paths"""
    load_dotenv()
    return (
        os.getenv('path_hdf'),
        os.getenv('path_nc'), 
        os.getenv('path_shp'),
        os.getenv('path_preprocessed'),
        os.getenv('path_efficiency')
    )

def main():

    credentials = dotenv_values(".env")
    HOST, RABBITMQ_PORT, KEYCLOAK_PORT, KEYCLOAK_REALM = credentials["HOST"], int(credentials["RABBITMQ_PORT"]), int(credentials["KEYCLOAK_PORT"]), str(credentials["KEYCLOAK_REALM"])
    USERNAME, PASSWORD = credentials["USERNAME"], credentials["PASSWORD"]
    CLIENT_ID = credentials["CLIENT_ID"]
    CLIENT_SECRET_KEY = credentials["CLIENT_SECRET_KEY"]
    VIRTUAL_HOST = credentials["VIRTUAL_HOST"]
    IS_TLS = credentials["IS_TLS"].lower() == 'true'
    PREFIX = "sos"
    NAME = "layer"#"constellation"

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

    app = ManagedApplication(NAME)

    # Load configurations
    
    path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency = load_config()
    # # Download data
    # download_snow_data(path_hdf, "2024.01.01", "2024.02.02")


    constellation = Constellation("layer") #"constellation")
    app.simulator.add_entity(constellation)
    app.simulator.add_observer(ShutDownObserver(app))

    # Initialize LayerPublisher
    layer_publisher = LayerPublisher(app, constellation, timedelta(seconds=120))
    # layer_publisher.snow_layer = snow_layer
    # layer_publisher.top_left = top_left
    # layer_publisher.top_right = top_right
    # layer_publisher.bottom_left = bottom_left
    # layer_publisher.bottom_right = bottom_right

    app.simulator.add_observer(layer_publisher)

    app.start_up(
        PREFIX,
        config,
        True,
        time_status_step=timedelta(seconds=10) * SCALE,
        time_status_init=datetime(2024, 1, 7, tzinfo=timezone.utc),
        time_step=timedelta(seconds=1) * SCALE,
    )

if __name__ == "__main__":
    main()