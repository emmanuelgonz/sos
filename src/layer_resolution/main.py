# All the imports for libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import earthaccess
import xarray as xr
import rioxarray as rxr
import os
import sys
import regex as rgx
from datetime import datetime,date,timedelta
import dask
import glob
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.enums import Resampling
from datetime import datetime
import gzip
import requests
import shutil
import tarfile
import tempfile
# from osgeo import gdal # pylint: disable=unused-import
import pandas as pd
from datetime import datetime, timezone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sos_tools.efficiency import efficiency
from dotenv import load_dotenv

# loading environment file for using the file path
# load_dotenv('C:\\Users\\dramach6\\OneDrive - Arizona State University\\Documents\\sos\\src\\.env')
load_dotenv()

# Defining paths
path_shp = os.getenv('path_shp')
path_preprocessed = os.getenv('path_preprocessed')
file_name_preprocessed = 'preprocessed_resolution'
raw_path = os.getenv('raw_path')
path_efficiency = os.getenv('path_efficiency')
file_name_efficiency = 'efficiency_resolution'
file_name_efficiency_taskable = 'efficiency_resolution_taskable'

# Downloading SNODAS data
# define the date range over which to prepare data
s = datetime(2024, 1, 1)
e = datetime(2024, 1, 31)
dates = pd.date_range(s, e)
# defining date range with tz info
start = s.replace(tzinfo=timezone.utc)
end = e.replace(tzinfo=timezone.utc)

# define the local directory in which to work
snodas_dir = os.path.join(raw_path,"SNODAS")
# ensure the directory exists
if not os.path.exists(snodas_dir):
    os.makedirs(snodas_dir)

# iterate over each date
for date in dates:
    # prepare the SWE file label for this date
    file_label = f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001"
    # check if file already exists
    if os.path.isfile(os.path.join(snodas_dir, file_label + ".nc")):
        print("Skipping " + file_label)
        continue
    print("Processing " + file_label)
    # prepare the SNODAS directory label for this date
    dir_label = f"SNODAS_{date.strftime('%Y%m%d')}"
    # request the .tar file from NSIDC
    r = requests.get(
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/" + 
        f"{date.strftime('%Y')}/{date.strftime('%m')}_{date.strftime('%b')}/" +
        dir_label + ".tar"
    )
    # create a temporary directory in which to do work
    print("Creating temp directory")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # save the .tar file
        with open(os.path.join(tmp_dir, dir_label + ".tar"), "wb") as tar_file:
            tar_file.write(r.content)
        # open and extract the .tar file
        with tarfile.open(os.path.join(tmp_dir, dir_label + ".tar"), "r") as tar_file:
            tar_file.extractall(tmp_dir)
        # iterate through all extracted files
        for filename in os.listdir(tmp_dir):
            # check if the file matches the SWE file label
            if os.path.isfile(os.path.join(tmp_dir, filename)) and filename == file_label + ".dat.gz":
                # unzip the SWE .gz file
                with gzip.open(os.path.join(tmp_dir, file_label + ".dat.gz"), "rb") as gz_in:
                    with open(os.path.join(tmp_dir, file_label + ".dat"), "wb") as gz_out:
                        shutil.copyfileobj(gz_in, gz_out)
                # write the SWE .hdr file
                with open(os.path.join(tmp_dir, file_label + ".hdr"), "w") as hdr_file:
                    hdr_file.write(
                        "ENVI\n"
                        "samples = 6935\n" +
                        "lines = 3351\n" +
                        "bands = 1\n" +
                        "header offset = 0\n" + 
                        "file type = ENVI Standard\n" + 
                        "data type = 2\n" +
                        "interleave = bsq\n" +
                        "byte order = 1"
                    )
                # run the gdal translator using date-specific bounding box
                print("temp to destination")
                command = " ".join([
                    "gdal_translate",
                    "-of NetCDF",
                    "-a_srs EPSG:4326",
                    "-a_nodata -9999",
                    "-a_ullr -124.73375000000000 52.87458333333333 -66.94208333333333 24.94958333333333" 
                    if date < datetime(2013, 10, 1)
                    else "-a_ullr -124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000",                    
                    os.path.join(tmp_dir, file_label + ".dat"),
                    '"'+ snodas_dir + '\\' + file_label + ".nc" + '"'
                  #  os.path.join(raw_path , snodas_dir, file_label + ".nc")
                ])
                if os.system(command) > 0: 
                    print(f"Error processing command `{command}`")

# Combining SNODAS files into one combined netcdf file
print("Writing snodas-merged.nc")
ds = xr.combine_by_coords(
    [
        rxr.open_rasterio(
            os.path.join(
                snodas_dir, 
                f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001.nc"
            )
        ).drop_vars("band").assign_coords(time=date).expand_dims(dim="time")
        for date in dates
    ], 
    combine_attrs="drop_conflicts"
).to_netcdf(os.path.join(snodas_dir, "snodas-merged.nc"))
print("Completed writing snodas-merged.nc")

# Now we download the missouri basin shp file

# Mo File read
print("Read Mo basin file")
mo_basin = gpd.read_file(path_shp+"WBD_10_HU2_Shape/Shape/WBDHU2.shp")
# construct a geoseries with the exterior of the basin and WGS84 coordinates
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

# Opening the merged nc file
snodas_dir = "SNODAS"
ds = xr.open_dataset(os.path.join(raw_path + snodas_dir, "snodas-merged.nc"))

# load the SNODAS SWE data and clip to the spatial domain
ds = rxr.open_rasterio(os.path.join(raw_path + snodas_dir, "snodas-merged.nc")).sel(
    time=slice(start, end),   
).rio.clip(mo_basin.geometry, "EPSG:4326")

print("Starting the resampling step")
# Now we compute the abs difference, and before that we do coarsening
# Resampling to 5km and then 1km
factor = 5
h = ds.rio.height * 5
w = ds.rio.width * 5 
# Downsample
ds_5km = ds.rio.reproject(ds.rio.crs, shape=(int(h), int(w)), resampling=Resampling.bilinear)
# Upsample
ds_1km = ds_5km.rio.reproject_match(ds,1)
# Computing absolute difference
ds_abs = abs(ds_1km-ds)

print("Final data wrangling - groupby for aggregation and joins")
# Now we add the month column and do a groupby
resolution_mo = ds_abs.convert_calendar(calendar='standard')
temp = resolution_mo.groupby(resolution_mo.time.dt.month).mean()
temp = temp.to_dataset()
temp = temp.rename({'Band1': 'Monthly_Resolution_Abs'})
#resolution_mo = resolution_mo.assign_coords(month = (resolution_mo.time.dt.month))
temp_resampled = temp.sel(month=resolution_mo.time.dt.month)
temp_resampled = temp_resampled.rio.write_crs("EPSG:4326")
temp_resampled = temp_resampled.rio.clip(mo_basin.geometry, "EPSG:4326")
temp_resampled.to_netcdf(path_preprocessed + file_name_preprocessed +'.nc')
print("Complete - Preprocessed file saved to the dropbox folder")

# Code for taskable satellite
# Computing absolute difference
ds_abs_taskable = abs(ds-ds)

print("Taskable - Final data wrangling - groupby for aggregation and joins")
# Now we add the month column and do a groupby
resolution_mo = ds_abs_taskable.convert_calendar(calendar='standard')
temp = resolution_mo.groupby(resolution_mo.time.dt.month).mean()
temp = temp.to_dataset()
temp = temp.rename({'Band1': 'Monthly_Resolution_Abs'})
#resolution_mo = resolution_mo.assign_coords(month = (resolution_mo.time.dt.month))
temp_resampled_taskable = temp.sel(month=resolution_mo.time.dt.month)
temp_resampled_taskable = temp_resampled_taskable.rio.write_crs("EPSG:4326")
temp_resampled_taskable = temp_resampled_taskable.rio.clip(mo_basin.geometry, "EPSG:4326")

print("Complete - Preprocessing for taskable")


# Computing efficiency

import sys
import os
sys.path.append(os.path.abspath("../src"))

# importing common function from sos_tools

# Importing input parameters from Configuration file

from configparser import ConfigParser

config = ConfigParser()
config.read("Input_parameters.ini")
print(config.sections())

config_data = config['Resolution'] 
T = float(config_data['threshold'])
k = float(config_data['coefficient'])

# Calling the efficiency function and passing the input parameters

efficiency_output = efficiency(T,k,temp_resampled)
efficiency_output.to_netcdf(path_efficiency + file_name_efficiency + '.nc')

# ------------------------------TASKABLE--------------------------------------
# Taskable Calling the efficiency function and passing the input parameters

efficiency_output = efficiency(T,k,temp_resampled_taskable)
efficiency_output.to_netcdf(path_efficiency + file_name_efficiency_taskable + '.nc')
print("ALL STEPS COMPLETED")