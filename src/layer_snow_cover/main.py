# Preprocessing Snow cover
# Importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import earthaccess
import xarray as xr
import rioxarray as rxr
import os
#import regex as rgx
from datetime import datetime,date,timedelta
import dask
import glob
import geopandas as gpd
from shapely.geometry import Polygon
import os
import requests
import zipfile


# loading environment file for using the file path

from dotenv import load_dotenv
import os
load_dotenv('C:\\Users\\dramach6\\OneDrive - Arizona State University\\Documents\\sos\\src\\.env')

path_hdf = os.getenv('path_hdf')
path_nc = os.getenv('path_nc')
path_shp = os.getenv('path_shp')
path_preprocessed = os.getenv('path_preprocessed')
file_name_preprocessed = 'preprocessed_snow_cover.nc'

# Each data has a code number that can be conveniently used to download data, first login to the earthacess account

# Downloading Data from earthaccess
# 1. Logging in
earthaccess.login(strategy="environment")
# 2. Search
results = earthaccess.search_data(
    short_name = 'MOD10C1',
    temporal = ("2024.01.01","2024.02.2")
)
files = earthaccess.download(results,path_hdf)

# Data prerpocessing

# Code to loop through file in the folder, add date component, convert to netcdf, and collate to one data set
# Code for Snow Cover

ctr = 0
lon = np.linspace(-180,180,7200)
lat = np.flip(np.linspace(-90,90,3600))
time_sc = []

for filename in os.listdir(path_hdf):    
    year = filename[9:13]
    day = filename[13:16]
    name = filename[0:34]

    # converting day of year to time

    dates = pd.to_datetime(int(day)-1,unit = 'D', origin=year)     
    time_sc.append(dates)
    f_nc = xr.open_dataset(path_hdf +'\\' + filename,engine = 'netcdf4')  
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
    print(glob.glob(os.path.join(path_nc,"*.nc")))

# Merged code

print("Writing snowcover-merged.nc")
ds = xr.combine_by_coords(
    [        
        rxr.open_rasterio(files[i]).drop_vars("band").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
         for i in range(len(time_sc))            
        
    ], 
    combine_attrs="drop_conflicts"
)
ds = ds.rio.write_crs("EPSG:4326")
ds.to_netcdf(os.path.join(path_nc, "snowcover-merged.nc"))
print("Completed writing snowcover-merged")

print("Reading mo_basin file and clipping")
# Access us states shapefile
us_map = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
conus = us_map[~us_map.STUSPS.isin(["AK", "HI", "PR"])].to_crs("EPSG:4326")
mo_basin = gpd.read_file(path_shp+"WBD_10_HU2_Shape/Shape/WBDHU2.shp")
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

print("Final data wrangling - groupby and joins")
# Opening the merged netcdf files
snow_layer = rxr.open_rasterio(os.path.join(path_nc,"snowcover-merged.nc"),crs = "EPSG:4326")
snow_layer_mo = snow_layer.rio.clip(mo_basin.envelope)
snow_layer_mo = snow_layer_mo.convert_calendar(calendar='standard')
temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max()
temp = temp.to_dataset()
temp = temp.rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})
temp_resampled = temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
temp_resampled.to_netcdf(path_preprocessed + file_name_preprocessed)
print("Completed - the preprocessed snow cover file has been saved to the Dropbox folder")