import os
import sys
import logging
from typing import List, Tuple
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv

# Geospatial
import geopandas as gpd
from shapely import Polygon
import xarray as xr
import numpy as np
import rioxarray

# NOS-T
from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.observer import Observer
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher
from nost_tools.simulator import Simulator, Mode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_config() -> Tuple[str, ...]:
    """
    Load environment variables and return paths
    
    Returns:
        Tuple[str, ...]: Tuple of paths for the HDF, NetCDF, shapefile, preprocessed, efficiency, and SNODAS directories
    """
    load_dotenv()
    return (
        os.getenv('path_hdf'),
        os.getenv('path_nc'), 
        os.getenv('path_shp'),
        os.getenv('path_preprocessed'),
        os.getenv('path_efficiency'),
        os.getenv('path_snodas')
    )

def get_missouri_basin(file_path: str) -> gpd.GeoSeries:
    """
    Get Missouri Basin geometry
    
    Args:
        file_path (str): File path to the shapefile
    
    Returns:
        gpd.GeoSeries: GeoSeries of the Missouri Basin geometry
    """
    mo_basin = gpd.read_file(file_path)
    return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

def calculate_beta5(swe_change, threshold=10, k_value=0.2):
    """
    Calculate beta5 values based on the logistic function
    
    Args:
        swe_change (xr.DataArray): Absolute SWE change values
        threshold (float): Threshold value for the logistic function
        k_value (float): Scaling factor for the logistic function
    
    Returns:
        xr.DataArray: Beta5 values based on the logistic function    
    """
    return 1 / (1 + np.exp(-k_value * (swe_change - threshold)))

def process_snodas_data(path_snodas: str, path_shp: str) -> xr.Dataset:
    """
    Process SNODAS data and return a dataset with beta5 values and SWE differences
    
    Args:
        path_snodas (str): Path to the SNODAS dataset
        path_shp (str): Path to the shapefile
    
    Returns:
        xr.Dataset: Dataset containing beta5 values and SWE differences
    """
    # Load the shapefile
    mo_basin = get_missouri_basin(file_path=os.path.join(path_shp, "WBD_10_HU2.shp"))

    # Load the SNODAS dataset
    file_path = os.path.join(path_snodas, "snodas-merged.nc")
    ds = xr.open_dataset(file_path)

    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")

    # Clip dataset to Missouri River Basin
    masked_ds = ds.rio.clip(mo_basin.geometry, mo_basin.crs)

    # Compute SWE values from the clipped dataset
    swe = masked_ds['Band1']

    # Squeeze any dimensions of size 1, especially for 'band'
    if 'band' in swe.dims and swe.sizes['band'] == 1:
        swe = swe.squeeze('band')

    # Check the range of SWE values to verify non-zero values
    print("SWE min:", swe.min().values, "SWE max:", swe.max().values)

    # Mask NaN and zero values before applying the difference calculation
    swe_masked = swe.where(~np.isnan(swe))

    # Calculate the SWE difference between consecutive time steps, keeping NaN values intact
    swe_diff_abs = swe_masked.diff(dim='time').where(~np.isnan(swe_masked.diff(dim='time')))

    # Set NaN values for zero differences or areas with no changes
    swe_diff_abs = abs(swe_diff_abs).where(swe_diff_abs != 0, np.nan)

    # Add a zero difference for the first time step to match the length
    swe_diff_abs = xr.concat([xr.zeros_like(swe.isel(time=0)), swe_diff_abs], dim='time')

    # Apply the beta5 calculation to SWE changes, keeping NaN values
    beta5_values = calculate_beta5(swe_diff_abs)

    # Replace NaN values with 1 in beta5
    beta5_values = beta5_values.fillna(1)

    # Create the DataArray for beta5 values
    beta5_da = xr.DataArray(
        beta5_values,
        coords={
            'time': swe['time'],
            'y': swe['y'],
            'x': swe['x']
        },
        dims=swe_diff_abs.dims,
        name='beta5'
    )

    # Create a new dataset with beta5 values and the absolute SWE difference
    new_ds = xr.Dataset({
        'beta5': beta5_da,
        'swe_diff_abs': swe_diff_abs
    })

    # Transpose the dataset to ensure 'time' is the first dimension
    new_ds = new_ds.transpose('time', 'y', 'x')

    # Remove 'grid_mapping' attribute if it exists in the dataset
    for var in new_ds.variables:
        if 'grid_mapping' in new_ds[var].attrs:
            del new_ds[var].attrs['grid_mapping']

    # Close the datasets
    ds.close()
    masked_ds.close()

    return new_ds

def save_dataset(dataset: xr.Dataset, path_efficiency: str):
    """
    Save the dataset to a NetCDF file
    
    Args:
        dataset (xr.Dataset): Dataset to save
        path_efficiency (str): Path to save the dataset
    """
    logger.info("Saving dataset to NetCDF file.")
    dataset.to_netcdf(os.path.join(path_efficiency, 'Efficiency_SWE_Change_dataset.nc'))
    logger.info("Saving dataset to NetCDF file successfully completed.")
    dataset.close()

def main():
    # Load configurations
    path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency, path_snodas = load_config()

    # Process SNODAS data
    new_ds = process_snodas_data(path_snodas, path_shp)

    # Save the new dataset to a NetCDF file
    save_dataset(new_ds, path_efficiency)

if __name__ == "__main__":
    main()
# import os
# import sys
# import logging
# from typing import List, Tuple
# from datetime import timedelta, datetime, timezone
# from dotenv import load_dotenv

# # Geospatial
# import geopandas as gpd
# from shapely import Polygon
# import xarray as xr
# import numpy as np
# import rioxarray

# # NOS-T
# from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
# from nost_tools.entity import Entity
# from nost_tools.observer import Observer
# from nost_tools.managed_application import ManagedApplication
# from nost_tools.publisher import WallclockTimeIntervalPublisher
# from nost_tools.simulator import Simulator, Mode

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()

# def load_config() -> Tuple[str, ...]:
#     """
#     Load environment variables and return paths
    
#     Returns:
#         Tuple[str, ...]: Tuple of paths for the HDF, NetCDF, shapefile, preprocessed, efficiency, and SNODAS directories
#     """
#     load_dotenv()
#     return (
#         os.getenv('path_hdf'),
#         os.getenv('path_nc'), 
#         os.getenv('path_shp'),
#         os.getenv('path_preprocessed'),
#         os.getenv('path_efficiency'),
#         os.getenv('path_snodas')
#     )

# def get_missouri_basin(file_path: str) -> gpd.GeoSeries:
#     """
#     Get Missouri Basin geometry
    
#     Args:
#         file_path (str): File path to the shapefile
    
#     Returns:
#         gpd.GeoSeries: GeoSeries of the Missouri Basin geometry
#     """
    
#     # logger.info("Downloading Missouri Basin geometry.")
#     mo_basin = gpd.read_file(file_path)
#     # logger.info("Downloading Missouri Basin geometry successfully completed.")

#     return gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

# def calculate_beta5(swe_change, threshold=10, k_value=0.2):
#     """
#     Calculate beta5 values based on the logistic function
    
#     Args:
#         swe_change (xr.DataArray): Absolute SWE change values
#         threshold (float): Threshold value for the logistic function
#         k_value (float): Scaling factor for the logistic function
    
#     Returns:
#         xr.DataArray: Beta5 values based on the logistic function    
#     """
#     return 1 / (1 + np.exp(-k_value * (swe_change - threshold)))

# # Load configurations
# path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency, path_snodas = load_config()

# # Load the shapefile
# mo_basin = get_missouri_basin(file_path=os.path.join(path_shp, "WBD_10_HU2.shp"))

# # Load the SNODAS dataset
# file_path = os.path.join(path_snodas, "snodas-merged.nc")

# # Open the dataset
# ds = xr.open_dataset(file_path)

# if not ds.rio.crs:
#     ds = ds.rio.write_crs("EPSG:4326")
# # Clip dataset to Missouri River Basin (assuming you have the mo_basin as a GeoDataFrame)
# # Ensure that the mo_basin has the same CRS as the dataset
# masked_ds = ds.rio.clip(mo_basin.geometry, mo_basin.crs)

# # Compute SWE values from the clipped dataset
# swe = masked_ds['Band1']

# # Squeeze any dimensions of size 1, especially for 'band'
# if 'band' in swe.dims and swe.sizes['band'] == 1:
#     swe = swe.squeeze('band')

# # Check the range of SWE values to verify non-zero values
# print("SWE min:", swe.min().values, "SWE max:", swe.max().values)

# # Mask NaN and zero values before applying the difference calculation
# swe_masked = swe.where(~np.isnan(swe))

# # Calculate the SWE difference between consecutive time steps, keeping NaN values intact
# swe_diff_abs = swe_masked.diff(dim='time').where(~np.isnan(swe_masked.diff(dim='time')))

# # Set NaN values for zero differences or areas with no changes
# swe_diff_abs = abs(swe_diff_abs).where(swe_diff_abs != 0, np.nan)

# # Add a zero difference for the first time step to match the length
# swe_diff_abs = xr.concat([xr.zeros_like(swe.isel(time=0)), swe_diff_abs], dim='time')

# # Apply the beta5 calculation to SWE changes, keeping NaN values
# beta5_values = calculate_beta5(swe_diff_abs)

# # Replace NaN values with 1 in beta5
# beta5_values = beta5_values.fillna(1)

# # Create the DataArray for beta5 values
# beta5_da = xr.DataArray(
#     beta5_values,
#     coords={
#         'time': swe['time'],
#         'y': swe['y'],
#         'x': swe['x']
#     },
#     dims=swe_diff_abs.dims,
#     name='beta5'
# )

# # Create a new dataset with beta5 values and the absolute SWE difference
# new_ds = xr.Dataset({
#     'beta5': beta5_da,
#     'swe_diff_abs': swe_diff_abs
# })

# # Transpose the dataset to ensure 'time' is the first dimension
# new_ds = new_ds.transpose('time', 'y', 'x')

# # Remove 'grid_mapping' attribute if it exists in the dataset
# for var in new_ds.variables:
#     if 'grid_mapping' in new_ds[var].attrs:
#         del new_ds[var].attrs['grid_mapping']

# # Save the new dataset to a NetCDF file
# logger.info("Saving dataset to NetCDF file.")
# # path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
# new_ds.to_netcdf(os.path.join(path_efficiency, 'Efficiency_SWE_Change_dataset.nc'))
# logger.info("Saving dataset to NetCDF file successfully completed.")

# # Close the datasets
# ds.close()
# masked_ds.close()
# new_ds.close()