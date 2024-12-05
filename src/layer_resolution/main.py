import os
import sys
import tarfile
from typing import List, Tuple
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthaccess
import geopandas as gpd
from shapely.geometry import Polygon
import requests
import gzip
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from rasterio.enums import Resampling

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

    def efficiency(self, T,k,datarray):
        resolution_eta = 1 / (1 + np.exp(k * (datarray - T)))       
        return resolution_eta

    def load_config(self):
        """Load environment variables and config parameters"""
        load_dotenv()
        paths = {
            'shp': os.getenv('path_shp'),
            'preprocessed': os.getenv('path_preprocessed'),
            'raw': os.getenv('raw_path'),
            'efficiency': os.getenv('path_efficiency')
        }
        
        config = ConfigParser()
        config.read("Input_parameters.ini")
        config_data = config['Resolution']
        
        return paths, float(config_data['threshold']), float(config_data['coefficient'])

    def download_snodas_data(self, start_date, end_date, raw_path):
        """Download SNODAS data for given date range"""
        logger.info("Downloading SNODAS data.")
        dates = pd.date_range(start_date, end_date)
        snodas_dir = os.path.join(raw_path, "SNODAS")
        
        if not os.path.exists(snodas_dir):
            os.makedirs(snodas_dir)
            
        for date in dates:
            file_label = f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001"
            
            if os.path.isfile(os.path.join(snodas_dir, file_label + ".nc")):
                print(f"Skipping {file_label}")
                continue
                
            self.process_snodas_date(date, file_label, snodas_dir)
        logger.info("Downloading SNODAS data successfully completed.")

        return snodas_dir, dates

    def process_snodas_date(self, date, file_label, snodas_dir):
        """Process SNODAS data for a single date"""
        dir_label = f"SNODAS_{date.strftime('%Y%m%d')}"
        r = requests.get(
            "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/" + 
            f"{date.strftime('%Y')}/{date.strftime('%m')}_{date.strftime('%b')}/" +
            dir_label + ".tar"
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.extract_and_process_tar(r, tmp_dir, dir_label, file_label, date, snodas_dir)

    def extract_and_process_tar(self, response, tmp_dir, dir_label, file_label, date, snodas_dir):
        """Extract and process TAR file contents"""
        tar_path = os.path.join(tmp_dir, dir_label + ".tar")
        with open(tar_path, "wb") as tar_file:
            tar_file.write(response.content)
            
        with tarfile.open(tar_path, "r") as tar_file:
            tar_file.extractall(tmp_dir)
            
        self.process_gz_file(tmp_dir, file_label, date, snodas_dir)

    def process_gz_file(self, tmp_dir, file_label, date, snodas_dir):
        """Process gzipped file and create NetCDF"""
        gz_path = os.path.join(tmp_dir, file_label + ".dat.gz")
        if os.path.isfile(gz_path):
            with gzip.open(gz_path, "rb") as gz_in:
                with open(os.path.join(tmp_dir, file_label + ".dat"), "wb") as gz_out:
                    shutil.copyfileobj(gz_in, gz_out)
                    
            self.write_hdr_file(tmp_dir, file_label)
            self.run_gdal_translate(tmp_dir, file_label, date, snodas_dir)

    def write_hdr_file(self, tmp_dir, file_label):
        """Write ENVI header file"""
        with open(os.path.join(tmp_dir, file_label + ".hdr"), "w") as hdr_file:
            hdr_file.write(
                "ENVI\n"
                "samples = 6935\n"
                "lines = 3351\n"
                "bands = 1\n"
                "header offset = 0\n"
                "file type = ENVI Standard\n"
                "data type = 2\n"
                "interleave = bsq\n"
                "byte order = 1"
            )

    def run_gdal_translate(self, tmp_dir, file_label, date, snodas_dir):
        """Run GDAL translate command"""
        ullr = ("-a_ullr -124.73375000000000 52.87458333333333 -66.94208333333333 24.94958333333333" 
                if date < datetime(2013, 10, 1)
                else "-a_ullr -124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000")
        
        command = f"gdal_translate -of NetCDF -a_srs EPSG:4326 -a_nodata -9999 {ullr} {os.path.join(tmp_dir, file_label + '.dat')} {os.path.join(snodas_dir, file_label + '.nc')}"
        os.system(command)

    def merge_snodas_files(self, snodas_dir, dates):
        """Merge individual SNODAS files into one NetCDF"""
        logger.info("Merging NetCDF files.")
        ds = xr.combine_by_coords(
            [rxr.open_rasterio(
                os.path.join(snodas_dir, 
                            f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001.nc")
            ).drop_vars("band", errors='ignore').assign_coords(time=date).expand_dims(dim="time")
            for date in dates], 
            combine_attrs="drop_conflicts"
        )
        ds.to_netcdf(os.path.join(snodas_dir, "snodas-merged.nc"))
        logger.info(f"Merging NetCDF files successfully completed.")
        return ds

    def get_missouri_basin(self, path_shp):
        """Get Missouri Basin geometry"""

        logger.info("Downloading Missouri Basin geometry.")
        mo_basin = gpd.read_file(path_shp + "WBD_10_HU2_Shape/Shape/WBDHU2.shp")
        logger.info("Downloading Missouri Basin geometry successfully completed.")

        return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

    def process_resolution(self, ds, mo_basin):
        """Process resolution analysis"""
        logger.info("Processing resolution layer.")
        ds_clipped = ds.rio.clip(mo_basin.geometry, "EPSG:4326")
        
        # Print the dimensions of the dataset for debugging
        print(f"Dataset dimensions before squeezing: {ds_clipped.dims}")
        
        # Remove the singleton 'band' dimension if it exists
        if 'band' in ds_clipped.dims and ds_clipped.dims['band'] == 1:
            ds_clipped = ds_clipped.squeeze('band')
        
        # Print the dimensions of the dataset after squeezing
        print(f"Dataset dimensions after squeezing: {ds_clipped.dims}")
        
        # Ensure the data array is 2D or 3D
        if len(ds_clipped.dims) > 3:
            raise ValueError("Only 2D and 3D data arrays supported.")
        
        # Resampling steps
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
        resolution_mo = ds_abs.convert_calendar(calendar='standard')
        temp = resolution_mo.groupby(resolution_mo.time.dt.month).mean()
        
        # Rename the variable directly
        temp = temp.rename({'Band1': 'Monthly_Resolution_Abs'})
        
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
    
    def publish_message(self):

        current_time = self.constellation.get_time()

        # Load configuration
        paths, threshold, coefficient = self.load_config()
        
        # Set date range
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        # Download and process SNODAS data
        snodas_dir, dates = self.download_snodas_data(start_date, end_date, paths['raw'])
        
        # Merge SNODAS files
        ds = self.merge_snodas_files(snodas_dir, dates)
        
        # Get Missouri Basin
        mo_basin = self.get_missouri_basin(paths['shp'])
        
        # Process resolution
        ds_clipped, ds_abs = self.process_resolution(ds, mo_basin)
        
        # Compute monthly resolution
        temp_resampled = self.compute_monthly_resolution(ds_abs, mo_basin)
        temp_resampled.to_netcdf(paths['preprocessed'] + 'preprocessed_resolution.nc')
        
        # Process taskable data
        ds_abs_taskable = abs(ds_clipped - ds_clipped)
        temp_resampled_taskable = self.compute_monthly_resolution(ds_abs_taskable, mo_basin)
        
        # Compute efficiency
        dataset = self.efficiency(threshold, coefficient, temp_resampled)
        dataset.to_netcdf(paths['efficiency'] + 'efficiency_resolution.nc')
        
        # efficiency_output_taskable = efficiency(threshold, coefficient, temp_resampled_taskable)
        # efficiency_output_taskable.to_netcdf(paths['efficiency'] + 'efficiency_resolution_taskable.nc')
        
        resolution_layer, top_left, top_right, bottom_left, bottom_right = self.encode(
                dataset=dataset,
                # file_path=os.path.join(path_efficiency, 'efficiency_snow_cover_up.nc'),
                variable='Monthly_Resolution_Abs',
                output_path='resolution_raster_layer.png',
                scale='time',
                time_step=str(current_time.date()),
                geojson_path='WBD_10_HU2_4326.geojson')
        
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
    
def main():

    credentials = dotenv_values(".env")
    HOST, RABBITMQ_PORT, KEYCLOAK_PORT, KEYCLOAK_REALM = credentials["HOST"], int(credentials["RABBITMQ_PORT"]), int(credentials["KEYCLOAK_PORT"]), str(credentials["KEYCLOAK_REALM"])
    USERNAME, PASSWORD = credentials["USERNAME"], credentials["PASSWORD"]
    CLIENT_ID = credentials["CLIENT_ID"]
    CLIENT_SECRET_KEY = credentials["CLIENT_SECRET_KEY"]
    VIRTUAL_HOST = credentials["VIRTUAL_HOST"]
    IS_TLS = credentials["IS_TLS"].lower() == 'true'
    PREFIX = "sos"
    NAME = "resolution"

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

    constellation = Constellation("resolution")
    app.simulator.add_entity(constellation)
    app.simulator.add_observer(ShutDownObserver(app))

    # Initialize LayerPublisher
    layer_publisher = LayerPublisher(app, constellation, timedelta(seconds=120))

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