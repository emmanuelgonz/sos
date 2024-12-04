import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthaccess
import geopandas as gpd
from shapely.geometry import Polygon
import os
import sys
import requests
import gzip
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from rasterio.enums import Resampling
from configparser import ConfigParser
from dotenv import load_dotenv
from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher

from constellation_config_files.schemas import SatelliteStatus, SnowLayer, ResolutionLayer, GcomLayer, CapellaLayer
from constellation_config_files.config import PREFIX, NAME, SCALE, TLES, FIELD_OF_REGARD

import logging
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def efficiency(T,k,datarray):
    resolution_eta = 1 / (1 + np.exp(k * (datarray - T)))       
    return resolution_eta

def load_config():
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

def download_snodas_data(start_date, end_date, raw_path):
    """Download SNODAS data for given date range"""
    dates = pd.date_range(start_date, end_date)
    snodas_dir = os.path.join(raw_path, "SNODAS")
    
    if not os.path.exists(snodas_dir):
        os.makedirs(snodas_dir)
        
    for date in dates:
        file_label = f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001"
        
        if os.path.isfile(os.path.join(snodas_dir, file_label + ".nc")):
            print(f"Skipping {file_label}")
            continue
            
        process_snodas_date(date, file_label, snodas_dir)
    
    return snodas_dir, dates

def process_snodas_date(date, file_label, snodas_dir):
    """Process SNODAS data for a single date"""
    dir_label = f"SNODAS_{date.strftime('%Y%m%d')}"
    r = requests.get(
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/" + 
        f"{date.strftime('%Y')}/{date.strftime('%m')}_{date.strftime('%b')}/" +
        dir_label + ".tar"
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_and_process_tar(r, tmp_dir, dir_label, file_label, date, snodas_dir)

def extract_and_process_tar(response, tmp_dir, dir_label, file_label, date, snodas_dir):
    """Extract and process TAR file contents"""
    tar_path = os.path.join(tmp_dir, dir_label + ".tar")
    with open(tar_path, "wb") as tar_file:
        tar_file.write(response.content)
        
    with tarfile.open(tar_path, "r") as tar_file:
        tar_file.extractall(tmp_dir)
        
    process_gz_file(tmp_dir, file_label, date, snodas_dir)

def process_gz_file(tmp_dir, file_label, date, snodas_dir):
    """Process gzipped file and create NetCDF"""
    gz_path = os.path.join(tmp_dir, file_label + ".dat.gz")
    if os.path.isfile(gz_path):
        with gzip.open(gz_path, "rb") as gz_in:
            with open(os.path.join(tmp_dir, file_label + ".dat"), "wb") as gz_out:
                shutil.copyfileobj(gz_in, gz_out)
                
        write_hdr_file(tmp_dir, file_label)
        run_gdal_translate(tmp_dir, file_label, date, snodas_dir)

def write_hdr_file(tmp_dir, file_label):
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

def run_gdal_translate(tmp_dir, file_label, date, snodas_dir):
    """Run GDAL translate command"""
    ullr = ("-a_ullr -124.73375000000000 52.87458333333333 -66.94208333333333 24.94958333333333" 
            if date < datetime(2013, 10, 1)
            else "-a_ullr -124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000")
    
    command = f"gdal_translate -of NetCDF -a_srs EPSG:4326 -a_nodata -9999 {ullr} {os.path.join(tmp_dir, file_label + '.dat')} {os.path.join(snodas_dir, file_label + '.nc')}"
    os.system(command)

def merge_snodas_files(snodas_dir, dates):
    """Merge individual SNODAS files into one NetCDF"""
    print("Writing snodas-merged.nc")
    ds = xr.combine_by_coords(
        [rxr.open_rasterio(
            os.path.join(snodas_dir, 
                        f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001.nc")
         ).drop_vars("band", errors='ignore').assign_coords(time=date).expand_dims(dim="time")
         for date in dates], 
        combine_attrs="drop_conflicts"
    )
    ds.to_netcdf(os.path.join(snodas_dir, "snodas-merged.nc"))
    return ds

def get_missouri_basin(path_shp):
    """Get Missouri Basin geometry"""
    mo_basin = gpd.read_file(path_shp + "WBD_10_HU2_Shape/Shape/WBDHU2.shp")
    return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

def process_resolution(ds, mo_basin):
    """Process resolution analysis"""
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
    
    return ds_clipped, abs(ds_1km - ds_clipped)

def compute_monthly_resolution(ds_abs, mo_basin):
    """Compute monthly resolution statistics"""
    resolution_mo = ds_abs.convert_calendar(calendar='standard')
    temp = resolution_mo.groupby(resolution_mo.time.dt.month).mean()
    
    # Rename the variable directly
    temp = temp.rename({'Band1': 'Monthly_Resolution_Abs'})
    
    temp_resampled = (temp.sel(month=resolution_mo.time.dt.month)
                      .rio.write_crs("EPSG:4326")
                      .rio.clip(mo_basin.geometry, "EPSG:4326"))
    return temp_resampled

def open_polygons(geojson_path):
    geojson = gpd.read_file(geojson_path)
    polygons = geojson.geometry
    print('Polygons loaded.')
    return polygons

def downsample_array(array, downsample_factor):
    """
    Downsamples the given array by the specified factor.

    Args:
        array (np.ndarray): The array to downsample.
        downsample_factor (int): The factor by which to downsample the array.

    Returns:
        np.ndarray: The downsampled array.
    """
    return array[::downsample_factor, ::downsample_factor]

def get_extents(dataset, variable):
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

def encode(dataset, variable, output_path, time_step, scale, geojson_path, downsample_factor=1):

    logger.info('Encoding snow layer.')
    polygons = open_polygons(geojson_path=geojson_path)
    
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
    
    raster_layer = downsample_array(raster_layer, downsample_factor=downsample_factor)

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

    top_left, top_right, bottom_left, bottom_right = get_extents(dataset, variable=variable)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    raster_layer_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')

    logger.info('Encoding snow layer successfully completed.')

    return raster_layer_encoded, top_left, top_right, bottom_left, bottom_right
    
def main():
    # Load configuration
    paths, threshold, coefficient = load_config()
    
    # Set date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    # Download and process SNODAS data
    snodas_dir, dates = download_snodas_data(start_date, end_date, paths['raw'])
    
    # Merge SNODAS files
    ds = merge_snodas_files(snodas_dir, dates)
    
    # Get Missouri Basin
    mo_basin = get_missouri_basin(paths['shp'])
    
    # Process resolution
    ds_clipped, ds_abs = process_resolution(ds, mo_basin)
    
    # Compute monthly resolution
    temp_resampled = compute_monthly_resolution(ds_abs, mo_basin)
    temp_resampled.to_netcdf(paths['preprocessed'] + 'preprocessed_resolution.nc')
    
    # Process taskable data
    ds_abs_taskable = abs(ds_clipped - ds_clipped)
    temp_resampled_taskable = compute_monthly_resolution(ds_abs_taskable, mo_basin)
    
    # Compute efficiency
    dataset = efficiency(threshold, coefficient, temp_resampled)
    dataset.to_netcdf(paths['efficiency'] + 'efficiency_resolution.nc')
    
    # dataset = efficiency(threshold, coefficient, temp_resampled_taskable)
    # dataset.to_netcdf(paths['efficiency'] + 'efficiency_resolution_taskable.nc')
    
    snow_layer, top_left, top_right, bottom_left, bottom_right = encode(
        dataset=dataset,
        # file_path=os.path.join(path_efficiency, 'efficiency_snow_cover_up.nc'),
        variable='Monthly_Resolution_Abs',
        output_path='resolution_raster_layer.png',
        scale='time',
        time_step='2024-01-31', #str(current_time.date()),
        geojson_path='WBD_10_HU2_4326.geojson')

if __name__ == "__main__":
    main()