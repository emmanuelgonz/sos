import os
from pyhdf.SD import SD, SDC
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import earthaccess
from shapely.geometry import Point
from datetime import datetime
from dotenv import load_dotenv
import geopandas as gpd
from shapely import Polygon
import requests
from typing import List, Tuple
import rioxarray  # Add this line
# import geojson
import glob
import re
import rioxarray as rxr

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

# Function to convert HDF dataset to xarray DataArray
def convert_to_xarray(hdf_file, dataset_name):
    data = hdf_file.select(dataset_name)
    data_array = data[:]
    attrs = data.attributes()
    dims = [dim_name for dim_name, _ in data.dimensions().items()]
    data_array = xr.DataArray(data_array, name=dataset_name, dims=dims, attrs=attrs)
    fill_value = attrs.get('_FillValue', None)
    if fill_value is not None:
        data_array = data_array.where(data_array != fill_value, np.nan)
    return data_array

# Function to perform bilinear interpolation
def bilinear_interpolation(da):
    interpolated_slices = []
    for t in range(da.shape[0]):
        slice_data = da.isel(time=t)
        y, x = np.where(~np.isnan(slice_data.values))
        values = slice_data.values[~np.isnan(slice_data.values)]
        points = np.array(list(zip(y, x)))
        grid_y, grid_x = np.mgrid[0:slice_data.shape[0], 0:slice_data.shape[1]]
        interpolated = griddata(points, values, (grid_y, grid_x), method='linear')
        remaining_nans = np.isnan(interpolated)
        if np.any(remaining_nans):
            interpolated[remaining_nans] = griddata(points, values, (grid_y[remaining_nans], grid_x[remaining_nans]), method='nearest')
        interpolated_slices.append(
            xr.DataArray(
                interpolated,
                coords=slice_data.coords,
                dims=slice_data.dims,
                name=slice_data.name,
                attrs=slice_data.attrs,
            )
        )
    return xr.concat(interpolated_slices, dim='time')

# Function to create a mask for the dataset based on the Missouri River Basin
def create_basin_mask(lats, lons, basin_geometry):
    mask = np.zeros(lats.shape, dtype=bool)
    for i in range(lats.shape[0]):
        for j in range(lats.shape[1]):
            point = Point(lons[i, j], lats[i, j])
            if basin_geometry.contains(point):
                mask[i, j] = True
    return mask
# def create_basin_mask(lats, lons, basin_geometry):
#     mask = np.zeros(lats.shape, dtype=bool)
#     for i in range(lats.shape[0]):
#         for j in range(lats.shape[1]):
#             point = Point(lons[i, j], lats[i, j])
#             if basin_geometry.apply(lambda geom: geom.contains(point)).any():
#                 mask[i, j] = True
#     return mask

# Function to search and download datasets from NASA Earthdata
# def download_latest_datasets(collection, start_date, end_date):
#     earthaccess.login(strategy="environment") #"netrc")
#     results = earthaccess.search_data(
#         short_name=collection,
#         temporal=(start_date, end_date),
#     )
#     # print(results)
#     local_path = os.getcwd()
#     print(local_path)
#     files = earthaccess.download(results, local_path=local_path, threads=1)
#     return files
def download_earthacces_data(out_path: str, start_date: str, end_date: str, collection = "AIRS3STD") -> List[str]:
    """Download snow cover data using earthaccess"""
    # logger.info("Downloading snow data.")
    print('Downloading data.')
    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=collection,
        temporal=(start_date, end_date)
    )
    # logger.info("Downloading snow data successfully completed.")
    file_paths = earthaccess.download(results, out_path, threads=1)
    print("Downloading snow data successfully completed.")
    return file_paths

def get_missouri_basin(file_path: str) -> gpd.GeoSeries:
    """Get Missouri Basin geometry"""
    
    # logger.info("Downloading Missouri Basin geometry.")
    mo_basin = gpd.read_file(file_path)
    # logger.info("Downloading Missouri Basin geometry successfully completed.")

    return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

def extract_date(filename):
    # Use regular expression to find the date
    match = re.search(r'\d{4}\.\d{2}\.\d{2}', filename)
    if match:
        date = match.group()
        year, month, day = date.split('.')
        return year, month, day
    else:
        return None, None, None
    
def process_files(path_hdf: str, path_nc: str) -> Tuple[List[str], List[datetime]]:
    # logger.info("Processing snow files.")
    ctr = 0
    lon = np.linspace(-180,180,360)
    lat = np.flip(np.linspace(-90,90,180))
    files = []
    time_sc = []

    datasets = []
    for filename in os.listdir(path_hdf):
        if os.path.isfile(os.path.join(path_hdf, filename)):
            print(filename)
            
            # print(f'Processing {filename}')
            name = os.path.splitext(filename)[0]
            year, month, day = extract_date(filename)
            dates = pd.to_datetime(int(day)-1,unit = 'D', origin=year)     
            time_sc.append(dates)

            if not os.path.isfile(os.path.join(path_nc, ''.join([name, ".nc"]))):
                # Extract the date from the filename
                f_nc = xr.open_dataset(os.path.join(path_hdf, filename),engine = 'netcdf4')

                # Extract the data array
                data = f_nc['SurfAirTemp_A']
                # latitude = f_nc['Latitude'][:, 0]  # Extract the first column for unique latitude values
                # longitude = f_nc['Longitude'][0, :]  # Extract the first row for unique longitude values

                # Create a DataArray and append to the list
                temp_arr = xr.DataArray(
                data=data,
                dims=['lat','lon'],
                coords=dict(
                    lon = lon,
                    lat = lat,
                )
                )

                temp_arr.to_netcdf(os.path.join(path_nc, ''.join([name, ".nc"])))

        files = glob.glob(os.path.join(path_nc,"*.nc"))
        files = [f for f in files if not f.endswith("snowcover-merged.nc")]

    return files, time_sc

def merge_netcdf_files(files: List[str], time_sc: List[datetime], path_nc: str) -> xr.Dataset:
    """Merge NetCDF files with dask"""
    # logger.info("Merging NetCDF files.")
    output_file = os.path.join(path_nc, "temperature-merged.nc")

    if not os.path.isfile(output_file):
        # logger.info(f"Output file {output_file} does not exist. Merging NetCDF files.")
        ds = xr.combine_by_coords(
            [        
                rxr.open_rasterio(files[i]).drop_vars("band").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
                for i in range(len(time_sc))            
            ], 
            combine_attrs="drop_conflicts"
        )
        ds = ds.rio.write_crs("EPSG:4326")
        ds.to_netcdf(output_file)
        # logger.info(f"Merging NetCDF files successfully completed.")
    else:
        print(f"Output file {output_file} already exists. Skipping merging.")
        # logger.info(f"Output file {output_file} already exists. Skipping merging.")
    #     ds = xr.open_dataset(output_file)

    # return ds

def main():
    # Define the start and end dates for processing
    start_date = datetime(2024, 1, 20)
    end_date = datetime(2024, 1, 21)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")

    # Define the date range for file processing
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    print(f"Dates to process: {dates}")
    
    # Load configurations
    path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency = load_config()

    # Load the shapefile
    mo_basin = get_missouri_basin(file_path=os.path.join(path_shp, "WBD_10_HU2.shp"))

    # Define the dataset collection and download data
    file_paths = download_earthacces_data(out_path='./data', start_date=start_date, end_date=end_date, collection = "AIRS3STD")

    # Check the number of files downloaded
    if len(file_paths) != len(dates):
        print(f"Warning: Number of files downloaded ({len(file_paths)}) does not match the number of dates ({len(dates)}).")
    available_dates = dates[:len(file_paths)]

    files, time_sc = process_files(path_hdf='./data', path_nc='./data')
    print(time_sc)
    # merge_netcdf_files(files, time_sc, path_nc)
    # print(datasets[1])

    # # Process each file and store datasets
    # datasets = []
    # for i, (file_path, date) in enumerate(zip(file_paths, available_dates)):
    #     if os.path.exists(file_path):
    #         hdf_file = SD(file_path, SDC.READ)
    #         temp_data_array = convert_to_xarray(hdf_file, 'SurfAirTemp_A')
    #         latitudes = hdf_file.select('Latitude')[:]
    #         longitudes = hdf_file.select('Longitude')[:]
    #         print(latitudes)
            
    #         if latitudes[0, 0] > latitudes[-1, 0]:
    #             latitudes = np.flip(latitudes, axis=0)
    #             temp_data_array = temp_data_array.isel({'YDim:ascending': slice(None, None, -1)})
            
    #         ds = xr.Dataset({'SurfAirTemp_A': temp_data_array})
    #         ds = ds.assign_coords({
    #             'lat': (('YDim:ascending', 'XDim:ascending'), latitudes),
    #             'lon': (('YDim:ascending', 'XDim:ascending'), longitudes),
    #         })
    #         ds = ds.expand_dims(time=[date])
            
    #         # Rename dimensions to 'lat' and 'lon' if necessary
    #         if 'YDim:ascending' in ds.dims and 'XDim:ascending' in ds.dims:
    #             ds = ds.rename({'YDim:ascending': 'lat', 'XDim:ascending': 'lon'})
            
    #         # Set spatial dimensions and CRS
    #         ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    #         ds = ds.rio.write_crs("EPSG:4326", inplace=True)
            
    #         datasets.append(ds)
    #     else:
    #         print(f"File not found: {file_path}")
    # print(ds)
    # # Perform bilinear interpolation and combine datasets
    # interpolated_datasets = [ds.assign(SurfAirTemp_A=bilinear_interpolation(ds['SurfAirTemp_A'])) for ds in datasets]
    # combined_dataset = xr.concat(interpolated_datasets, dim='time')
    # print(combined_dataset)
    # print(combined_dataset['lat']['YDim:ascending'])
    # print(combined_dataset['lon']['XDim:ascending'])
    # combined_dataset.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    # # Apply a buffered mask to the combined dataset
    # buffer_size = 7
    # # mo_basin_geom_buffered = mo_basin.unary_union.buffer(buffer_size)
    # mo_basin_geom_buffered = mo_basin.union_all().buffer(buffer_size)
    # mask = create_basin_mask(combined_dataset['lat'].values, combined_dataset['lon'].values, mo_basin_geom_buffered)
    # mask_da = xr.DataArray(
    #     mask,
    #     coords={
    #         'y': combined_dataset['lat']['YDim:ascending'],
    #         'x': combined_dataset['lon']['XDim:ascending'],
    #     },
    #     dims=['YDim:ascending', 'XDim:ascending'],
    # )
    # masked_combined_dataset = combined_dataset.where(mask_da, drop=True)
    # masked_combined_dataset = masked_combined_dataset.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    # masked_combined_dataset = masked_combined_dataset.rio.write_crs("EPSG:4326")
    # print(masked_combined_dataset.rio.bounds())

    # # Save the masked dataset
    # output_path = './data/Efficiency_files/Efficiency_resolution20_Optimization/'
    # masked_combined_dataset.to_netcdf(os.path.join(output_path, 'Temperature_dataset.nc'))
    # print("Temperature dataset saved successfully.")

    # ###############################################################
    # #Efficiency
    # files_to_download = [
    #     ("https://www.dropbox.com/scl/fi/jfgcwmav28oylggr9ezci/Temperature_dataset.nc?rlkey=fo31b5vkphei1rm4o6urupj3w&st=gvauxdo0&dl=1", "Temperature_dataset.nc"),
    # ]

    # for url, output_file in files_to_download:
    #     try:
    #         # Send a GET request to the URL
    #         response = requests.get(url, stream=True)
    #         response.raise_for_status()  # Check for HTTP errors

    #         # Write the content to a file in chunks to avoid memory issues
    #         with open(output_file, "wb") as file:
    #             for chunk in response.iter_content(chunk_size=8192):
    #                 file.write(chunk)

    #         print(f"Download completed successfully! File saved as: {output_file}")
    #     except requests.exceptions.HTTPError as http_err:
    #         print(f"HTTP error occurred for {output_file}: {http_err}")
    #     except Exception as err:
    #         print(f"An error occurred for {output_file}: {err}")

    # # Define constants for eta calculation
    # T = 0  # Reference temperature in Celsius
    # k = 0.5  # Increased steepness parameter
    # b = 0  # No horizontal shift

    # # Load the masked dataset
    # ds = xr.open_dataset('Temperature_dataset.nc')
    # # ds = masked_combined_dataset

    # # Rename dimensions to `x` and `y` if necessary
    # if "XDim:ascending" in ds.dims and "YDim:ascending" in ds.dims:
    #     print('renameing')
    #     # ds = ds.rename({"XDim:ascending": "x", "YDim:ascending": "y"})
    #     ds = ds.swap_dims({"XDim:ascending": "x", "YDim:ascending": "y"})

    # # Ensure the spatial dimensions are correctly set
    # print(ds)
    # if not ds.rio.crs:
    #     ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True).rio.write_crs("EPSG:4326")
    # print(ds.rio.crs)
    # # Print the bounds of the dataset
    # bounds = ds.rio.bounds()
    # print(f"Bounds of the dataset: {bounds}")

    # Reproject the Missouri River Basin geometry to match the dataset's CRS
    # mo_basin = mo_basin.to_crs(ds.rio.crs)

    # print(mo_basin.crs)

    # # Apply a buffer to the Missouri River Basin geometry
    # buffer_size = 2  # Buffer size in degrees; adjust as needed
    # mo_basin_buffered = mo_basin.union_all().buffer(buffer_size)

    # # Print the bounds of the buffered Missouri River Basin
    # mo_basin_buffered_bounds = mo_basin_buffered.bounds
    # print(f"Bounds of the buffered Missouri River Basin: {mo_basin_buffered_bounds}")

    # # Clip dataset to the buffered Missouri River Basin
    # # masked_combined_dataset = ds.rio.clip(mo_basin_buffered.geometry, mo_basin_buffered.crs)
    # # masked_combined_dataset = ds.rio.clip(mo_basin.geometry, mo_basin.crs)
    # masked_combined_dataset = ds.rio.clip(mo_basin)
    # print(masked_combined_dataset)

    # # Initialize an array to hold the eta0 values
    # eta0_values = np.empty_like(masked_combined_dataset['SurfAirTemp_A'].values)

    # # Iterate over each time slice to calculate eta0
    # for t in range(len(masked_combined_dataset['time'])):
    #     # Extract temperature and convert from Kelvin to Celsius
    #     temp_frame = masked_combined_dataset['SurfAirTemp_A'].isel(time=t).values - 273.15
        
    #     # Calculate eta0 values while ignoring NaNs
    #     exponent = np.where(~np.isnan(temp_frame), k * (temp_frame - T) + b, np.nan)
    #     eta0_values[t] = np.where(~np.isnan(exponent), 1 / (1 + np.exp(exponent)), np.nan)

    # # Convert the eta0_values array back to an xarray.DataArray
    # eta0_dataarray = xr.DataArray(
    #     eta0_values,
    #     coords=masked_combined_dataset['SurfAirTemp_A'].coords,
    #     dims=masked_combined_dataset['SurfAirTemp_A'].dims,
    #     name='eta0'
    # )

    # # Create a new dataset for eta0
    # eta0_dataset = eta0_dataarray.to_dataset(name='eta0')

    # # Copy the attributes from the original dataset (optional)
    # eta0_dataset.attrs = masked_combined_dataset.attrs
    # eta0_dataset['eta0'].attrs = masked_combined_dataset['SurfAirTemp_A'].attrs

    # # Load the coarsened eta dataset (20 km by 20 km grid)
    # coarsened_eta_ds = xr.open_dataset('coarsened_eta_output_GCOM.nc')

    # # Extract the exact latitude and longitude grid from the coarsened dataset
    # target_lat = coarsened_eta_ds['y']
    # target_lon = coarsened_eta_ds['x']

    # # Resample the eta0 data to match the coarsened dataset grid using interpolation
    # eta0_resampled = eta0_dataset['eta0'].interp(
    #     y=target_lat,
    #     x=target_lon,
    #     method='linear'  # You can also try 'nearest' if that's preferred
    # )

    # # Create a new dataset with the resampled eta0 values
    # resampled_ds = xr.Dataset({'eta0': eta0_resampled})

    # output_path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
    # resampled_ds.to_netcdf(os.path.join(output_path, 'Efficiency_Temperature_dataset.nc'))
    # print("Efficiency_Temperature_dataset saved successfully.")

    # # Clean up temporary files
    # for file_path in file_paths:
    #     try:
    #         os.remove(file_path)
    #         print(f"Deleted file: {file_path}")
    #     except OSError as e:
    #         print(f"Error deleting file {file_path}: {e}")

if __name__ == '__main__':
    main()