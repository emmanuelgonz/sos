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
from datetime import datetime
import dask as dask
from configparser import ConfigParser
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from sos_tools.efficiency import efficiency

from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher

def efficiency(T,k,datarray):
    resolution_eta = 1 / (1 + np.exp(k * (datarray - T)))       
    return resolution_eta

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

def download_snow_data(path_hdf: str, start_date: str, end_date: str) -> List[str]:
    """Download snow cover data using earthaccess"""
    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name='MOD10C1',
        temporal=(start_date, end_date)
    )
    return earthaccess.download(results, path_hdf)

def process_snow_files(path_hdf: str, path_nc: str) -> Tuple[List[str], List[datetime]]:
    """Process HDF files to NetCDF with dask parallelization"""
    lon = np.linspace(-180, 180, 7200)
    lat = np.flip(np.linspace(-90, 90, 3600))
    files = []
    time_sc = []
    
    for filename in os.listdir(path_hdf):
        year = filename[9:13]
        day = filename[13:16]
        name = filename[0:34]
        
        dates = pd.to_datetime(int(day)-1, unit='D', origin=year)
        time_sc.append(dates)
        
        with dask.config.set(scheduler='threads'):
            f_nc = xr.open_dataset(os.path.join(path_hdf, filename), engine='netcdf4')
            snow = f_nc['Day_CMG_Snow_Cover'].chunk()
            temp_arr = xr.DataArray(
                data=snow,
                dims=['lat', 'lon'],
                coords=dict(lon=lon, lat=lat)
            )
            output_path = os.path.join(path_nc, f"{name}.nc")
            temp_arr.to_netcdf(output_path)
            files.append(output_path)
            
    return files, time_sc

def merge_netcdf_files(files: List[str], time_sc: List[datetime], path_nc: str) -> xr.Dataset:
    """Merge NetCDF files with dask"""
    print("Merging NetCDF files...")
    with dask.config.set(scheduler='threads'):
        ds = xr.combine_by_coords(
            [rxr.open_rasterio(f).drop_vars("band", errors="ignore")
             .assign_coords(time=t).expand_dims(dim="time")
             for f, t in zip(files, time_sc)],
            combine_attrs="drop_conflicts"
        )
        ds = ds.rio.write_crs("EPSG:4326")
        ds.to_netcdf(os.path.join(path_nc, "snowcover-merged.nc"))
    return ds

def get_missouri_basin(path_shp: str) -> gpd.GeoSeries:
    """Get Missouri Basin geometry"""
    us_map = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
    mo_basin = gpd.read_file(os.path.join(path_shp, "WBD_10_HU2_Shape/Shape/WBDHU2.shp"))
    return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

def process_snow_layer(path_nc: str, mo_basin: gpd.GeoSeries, path_preprocessed: str) -> xr.Dataset:
    """Process snow layer data"""
    snow_layer = rxr.open_rasterio(
        os.path.join(path_nc, "snowcover-merged.nc"),
        chunks={'x': 1000, 'y': 1000}
    ).rio.write_crs("EPSG:4326")
    
    snow_layer_mo = snow_layer.rio.clip(mo_basin.envelope)
    snow_layer_mo = snow_layer_mo.convert_calendar(calendar='standard')
    
    # Calculate the average snow cover over the desired time period
    temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max() #.mean()
    temp = temp.to_dataset().rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})
    
    temp_resampled = (temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
                     .rio.write_crs("EPSG:4326")
                     .rio.clip(mo_basin.geometry, "EPSG:4326"))
    
    temp_resampled.to_netcdf(os.path.join(path_preprocessed, 'preprocessed_snow_cover.nc'))
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

    return raster_layer_encoded, top_left, top_right, bottom_left, bottom_right

def main():

    # Load configurations
    path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency = load_config()
    
    # Download data
    download_snow_data(path_hdf, "2024.01.01", "2024.02.02")
    
    # Process files
    files, time_sc = process_snow_files(path_hdf, path_nc)
    
    # Merge files
    merge_netcdf_files(files, time_sc, path_nc)
    
    # Get Missouri Basin
    mo_basin = get_missouri_basin(path_shp)
    
    # Process snow layer
    temp_resampled = process_snow_layer(path_nc, mo_basin, path_preprocessed)

    # Compute efficiency
    config = ConfigParser()
    config.read("Input_parameters.ini")
    config_data = config['Snow_cover']
    T = float(config_data['threshold'])
    k = -float(config_data['coefficient'])
    
    dataset = efficiency(T, k, temp_resampled)
    dataset.to_netcdf(os.path.join(path_efficiency, 'efficiency_snow_cover.nc'))

    snow_layer, top_left, top_right, bottom_left, bottom_right = encode(
        dataset=dataset,
        variable='Weekly_Snow_Cover',
        output_path='snow_raster_layer.png',
        scale='time',
        time_step='2024-02-02',
        geojson_path='WBD_10_HU2_4326.geojson')
    
if __name__ == "__main__":
    main()