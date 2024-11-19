import os
import sys
from typing import List, Tuple
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import dask as dask
import earthaccess
import geopandas as gpd
from shapely.geometry import Polygon
from datetime import datetime
import dask.array as da
from configparser import ConfigParser
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sos_tools.efficiency import efficiency

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
    
    temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max()
    temp = temp.to_dataset().rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})
    
    temp_resampled = (temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
                     .rio.write_crs("EPSG:4326")
                     .rio.clip(mo_basin.geometry, "EPSG:4326"))
    
    temp_resampled.to_netcdf(os.path.join(path_preprocessed, 'preprocessed_snow_cover.nc'))
    return temp_resampled

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
    
    efficiency_output = efficiency(T, k, temp_resampled)
    efficiency_output.to_netcdf(os.path.join(path_efficiency, 'efficiency_snow_cover.nc'))
    print("Processing completed successfully")

if __name__ == "__main__":
    main()


# # Preprocessing Snow cover
# # Importing the required libraries

# import os
# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import earthaccess
# import xarray as xr
# import rioxarray as rxr

# #import regex as rgx
# from datetime import datetime,date,timedelta
# import dask
# import glob
# import geopandas as gpd
# from shapely.geometry import Polygon
# import requests
# import zipfile
# from configparser import ConfigParser
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from sos_tools.efficiency import efficiency
# from dotenv import load_dotenv, dotenv_values

# # loading environment file for using the file path
# # load_dotenv('/mnt/c/Users/emgonz38/OneDrive - Arizona State University/ubuntu_files/work/code/git_repos/sos/.env')
# load_dotenv()

# path_hdf = os.getenv('path_hdf')
# path_nc = os.getenv('path_nc')
# path_shp = os.getenv('path_shp')
# path_preprocessed = os.getenv('path_preprocessed')
# file_name_preprocessed = 'preprocessed_snow_cover'
# path_efficiency = os.getenv('path_efficiency')
# file_name_efficiency = 'efficiency_snow_cover'

# # Each data has a code number that can be conveniently used to download data, first login to the earthacess account

# # Downloading Data from earthaccess
# # 1. Logging in
# earthaccess.login(strategy="environment")
# # 2. Search
# results = earthaccess.search_data(
#     short_name = 'MOD10C1',
#     temporal = ("2024.01.01","2024.02.2")
# )
# files = earthaccess.download(results,path_hdf)

# # Data prerpocessing

# # Code to loop through file in the folder, add date component, convert to netcdf, and collate to one data set
# # Code for Snow Cover

# ctr = 0
# lon = np.linspace(-180,180,7200)
# lat = np.flip(np.linspace(-90,90,3600))
# time_sc = []

# for filename in os.listdir(path_hdf):    
#     year = filename[9:13]
#     day = filename[13:16]
#     name = filename[0:34]

#     # converting day of year to time

#     dates = pd.to_datetime(int(day)-1,unit = 'D', origin=year)     
#     time_sc.append(dates)
#     f_nc = xr.open_dataset(os.path.join(path_hdf, filename),engine = 'netcdf4')  
#     snow = f_nc['Day_CMG_Snow_Cover']
#     temp_arr = xr.DataArray(
#     data=snow,
#     dims=['lat','lon'],
#     coords=dict(
#         lon = lon,
#         lat = lat,
#     )
#     )
#     temp_arr.to_netcdf(path_nc + name + ".nc")
#     files = glob.glob(os.path.join(path_nc,"*.nc"))
#     print(glob.glob(os.path.join(path_nc,"*.nc")))

# # Merged code

# print("Writing snowcover-merged.nc")
# ds = xr.combine_by_coords(
#     [        
#         rxr.open_rasterio(files[i]).drop_vars("band", errors="ignore").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
#          for i in range(len(time_sc))            
        
#     ], 
#     combine_attrs="drop_conflicts"
# )
# ds = ds.rio.write_crs("EPSG:4326")
# ds.to_netcdf(os.path.join(path_nc, "snowcover-merged.nc"))
# print("Completed writing snowcover-merged")

# print("Reading mo_basin file and clipping")
# # Access us states shapefile
# us_map = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
# conus = us_map[~us_map.STUSPS.isin(["AK", "HI", "PR"])].to_crs("EPSG:4326")
# mo_basin = gpd.read_file(path_shp+"WBD_10_HU2_Shape/Shape/WBDHU2.shp")
# mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

# print("Final data wrangling - groupby and joins")
# # Opening the merged netcdf files
# snow_layer = rxr.open_rasterio(os.path.join(path_nc,"snowcover-merged.nc"),crs = "EPSG:4326")
# snow_layer_mo = snow_layer.rio.clip(mo_basin.envelope)
# snow_layer_mo = snow_layer_mo.convert_calendar(calendar='standard')
# temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max()
# temp = temp.to_dataset()
# temp = temp.rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})
# temp_resampled = temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
# temp_resampled = temp_resampled.rio.write_crs("EPSG:4326")
# temp_resampled = temp_resampled.rio.clip(mo_basin.geometry, "EPSG:4326")
# temp_resampled.to_netcdf(path_preprocessed + file_name_preprocessed+ '.nc')
# print("Completed - the preprocessed snow cover file has been saved to the Dropbox folder")

# # Computing efficiency
# # Importing input parameters from Configuration file
# config = ConfigParser()
# config.read("Input_parameters.ini")
# print(config.sections())

# config_data = config['Snow_cover'] 
# T = float(config_data['threshold'])
# k = (-1)*float(config_data['coefficient'])
# print(type(k))
# # Caliing the efficiency function and passing the input parameters

# efficiency_output = efficiency(T,k,temp_resampled)
# efficiency_output.to_netcdf(path_efficiency + file_name_efficiency + '.nc')
# print("ALL STEPS COMPLETED")