import base64
import io
import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO

import boto3
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from boto3.s3.transfer import TransferConfig
from joblib import Parallel, delayed
from nost_tools.application_utils import ShutDownObserver
from nost_tools.config import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import Observer
from nost_tools.simulator import Mode, Simulator
from PIL import Image
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from scipy.interpolate import griddata
from shapely.geometry import Polygon, box
from tatc import utils

# TATC
from tatc.analysis import compute_ground_track
from tatc.schemas import Instrument, Satellite, TwoLineElements

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


# define an observer to manage data updates
class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*

    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """

    def __init__(self, app):  # , grounds):
        self.app = app
        # self.path_hdf, self.path_nc, self.path_shp, self.path_preprocessed, self.path_efficiency, self.path_snodas = self.load_config()

    # AWS-related helper functions
    def start_session(self):
        session = boto3.Session()
        return session

    def get_session_token(
        self, session, mfa_serial=None, mfa_token=None, mfa_required=True
    ):
        sts = session.client("sts")
        if mfa_required:
            return sts.get_session_token(SerialNumber=mfa_serial, TokenCode=mfa_token)
        else:
            return sts.get_session_token()

    def decompose_token(self, token):
        credentials = token.get("Credentials", {})
        session_token = credentials.get("SessionToken")
        secret_access_key = credentials.get("SecretAccessKey")
        access_key_id = credentials.get("AccessKeyId")
        return session_token, secret_access_key, access_key_id

    def get_data(self, s3_filepath):
        """
        Get data from an S3 bucket

        Args:
            s3_file (str): The S3 file path

        Returns:
            dataset (xarray.Dataset): The dataset
        """
        # Start a new session
        fs = s3fs.S3FileSystem(anon=False)

        # Get the file contents
        fgrab = fs.open(s3_filepath)

        # Open the file
        dataset = xr.open_dataset(fgrab, engine="h5netcdf")

        return dataset

    # Function to interpolate variables from a dataset
    def interpolate_dataset(
        self, dataset, variables_to_interpolate, lat_coords, lon_coords, time_coords
    ):
        # Create meshgrid for interpolation
        xi = np.stack(np.meshgrid(lon_coords, lat_coords), axis=2).reshape(
            (lat_coords.size * lon_coords.size, 2)
        )

        # Initialize a dictionary to hold interpolated variables
        interpolated_vars = {}

        for var_name in variables_to_interpolate:
            # Get variable values, lons, and lats
            values = dataset[var_name].values
            lons = dataset.lon.values
            lats = dataset.lat.values

            # Perform interpolation
            zi = griddata(
                list(zip(lons.flatten(), lats.flatten())),  # Input coordinates
                values.flatten(),  # Input values
                xi,  # Output grid
                method="linear",  # Interpolation method
            )

            # Add interpolated variable to the dictionary
            interpolated_vars[var_name] = xr.DataArray(
                np.reshape(zi, (1, len(lat_coords), len(lon_coords))),
                coords={"time": time_coords, "y": lat_coords, "x": lon_coords},
                dims=["time", "y", "x"],
            ).rio.write_crs("EPSG:4326")

        # Create a new dataset with the interpolated variables
        new_dataset = xr.Dataset(interpolated_vars)
        return new_dataset

    def generate_combined_dataset(self, dataset1, dataset2, mo_basin):

        # Define coordinates for the regular grid
        time_coords_ds1 = np.array([dataset1.time[0].values])
        time_coords_ds2 = np.array([dataset2.time[0].values])
        lat_coords = np.linspace(37.024602, 49.739086, 76)
        lon_coords = np.linspace(-113.938141, -90.114221, 142)

        # Variables to interpolate
        variables_to_interpolate = ["SWE_tavg", "AvgSurfT_tavg"]

        # Interpolate datasets
        new_ds1 = self.interpolate_dataset(
            dataset1, variables_to_interpolate, lat_coords, lon_coords, time_coords_ds1
        )
        new_ds2 = self.interpolate_dataset(
            dataset2, variables_to_interpolate, lat_coords, lon_coords, time_coords_ds2
        )

        # # Save the new datasets to NetCDF files
        # new_ds1.to_netcdf("interpolated_dataset_ds1.nc")
        # new_ds2.to_netcdf("interpolated_dataset_ds2.nc")
        last_date_ds1 = np.datetime_as_string(
            dataset1.time[-1].values, unit="D"
        ).replace("-", "")
        last_date_ds2 = np.datetime_as_string(
            dataset2.time[-1].values, unit="D"
        ).replace("-", "")
        print(last_date_ds1)
        print(last_date_ds2)
        # Save the new datasets to NetCDF files with the last date in the file name
        # new_ds1.to_netcdf(f"interpolated_dataset_{last_date_ds1}.nc")
        # new_ds2.to_netcdf(f"interpolated_dataset_{last_date_ds2}.nc")

        # Combine the datasets along the time dimension
        combined_dataset = xr.concat([new_ds1, new_ds2], dim="time")
        print("Combined dataset dimensions:", combined_dataset.dims)

        # # Load the Missouri Basin shapefile
        # mo_basin = gpd.read_file("data/Downloaded_files/Mo_basin_shp/WBD_10_HU2_Shape/Shape/WBDHU2.shp") #"WBD_10_HU2_Shape/Shape/WBDHU2.shp")
        # mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

        # Reproject the Missouri Basin to match the dataset CRS
        mo_basin = mo_basin.to_crs("EPSG:4326")

        # Ensure the combined dataset has a CRS
        combined_dataset = combined_dataset.rio.write_crs("EPSG:4326")

        # Clip the dataset to the Missouri Basin
        clipped_dataset = combined_dataset.rio.clip(
            mo_basin.geometry, all_touched=True, drop=True
        )

        # Remove 'grid_mapping' attribute from all variables to avoid NetCDF conflicts
        for var in clipped_dataset.data_vars:
            if "grid_mapping" in clipped_dataset[var].attrs:
                del clipped_dataset[var].attrs["grid_mapping"]

        last_date = str(combined_dataset.time[-1].values)[:10].replace(
            "-", ""
        )  # Format as 'yyyymmdd'
        print(last_date)

        # Create the output file name with the last date
        output_file = f"LIS_dataset_{last_date}.nc"
        # clipped_dataset.to_netcdf(output_file)

        print(f"Final clipped dataset saved to '{output_file}'")

        return output_file

    def generate_swe_difference(self, ds):
        # Extract the SWE variable
        swe = ds["SWE_tavg"]

        # Debugging: Print shape of SWE
        print("Shape of SWE:", swe.shape)

        # Mask NaN values only (retain zero values for calculation)
        swe_masked = swe.where(~np.isnan(swe))

        # Calculate the SWE difference between consecutive time steps
        swe_diff = swe_masked.diff(dim="time", label="lower")

        # Debugging: Print shape of SWE_diff
        print("Shape of SWE_diff:", swe_diff.shape)

        # Take the absolute value of differences
        swe_diff_abs = abs(swe_diff)

        # Add a zero difference for the first time step to match the length
        zero_diff = xr.full_like(swe.isel(time=0), fill_value=0)
        swe_diff_abs = xr.concat([zero_diff, swe_diff_abs], dim="time")

        # Debugging: Print shape of SWE_diff_abs
        print("Shape of SWE_diff_abs:", swe_diff_abs.shape)

        # Ensure `time` dimension alignment
        swe_diff_abs = swe_diff_abs.assign_coords(time=swe["time"])

        # Define constants for eta calculation
        T = 10  # Threshold for the logistic function
        k = 0.2  # Scaling factor for the logistic function

        # Apply the eta calculation to SWE changes
        eta5_values = self.calculate_eta(swe_diff_abs)

        # Debugging: Print shape of eta5_values
        print("Shape of eta5_values:", eta5_values.shape)

        # Retain NaN values from the original mask (areas outside the SWE dataset)
        eta5_values = eta5_values.where(~np.isnan(swe_masked), np.nan)

        # Ensure eta5_values has the correct dimensions
        eta5_values = eta5_values.broadcast_like(swe)  # Align dimensions with SWE

        # Debugging: Verify shape after broadcasting
        print("Shape of eta5_values (after broadcasting):", eta5_values.shape)

        # Create the DataArray for eta values
        eta5_da = xr.DataArray(
            eta5_values.values,  # Use the raw values of eta5
            coords={
                "time": swe["time"],  # Use the correct 'time' coordinate
                "y": swe["y"],  # Use the 'y' (latitude) dimension
                "x": swe["x"],  # Use the 'x' (longitude) dimension
            },
            dims=["time", "y", "x"],
            name="eta5",
        )

        # Create a new dataset with eta values and the absolute SWE difference
        new_ds = xr.Dataset({"eta5": eta5_da, "swe_diff_abs": swe_diff_abs}).transpose(
            "time", "y", "x"
        )

        last_date = str(swe["time"][-1].values)[:10].replace("-", "")

        # Define the output file name with the last date
        output_file = f"Efficiency_SWE_Change_{last_date}.nc"

        # Save the dataset
        new_ds.to_netcdf(output_file)

        print(f"Dataset saved to '{output_file}'")

        # Check min and max of the eta values
        print("eta5 min:", eta5_da.min().values, "eta5 max:", eta5_da.max().values)

        return output_file

    def generate_surface_temp(self, ds):
        # Extract the SWE variable
        swe = ds["SWE_tavg"]

        # Extract the surface temperature variable (AvgSurfT_tavg) and convert from Kelvin to Celsius
        surface_temp = ds["AvgSurfT_tavg"] - 273.15

        # Debugging: Print shape and range of surface temperature
        print("Shape of Surface Temperature:", surface_temp.shape)
        print(
            "Surface Temperature min:",
            surface_temp.min().values,
            "max:",
            surface_temp.max().values,
        )

        # Mask NaN values (retain NaN outside valid regions)
        temp_masked = surface_temp.where(~np.isnan(surface_temp))

        # Define constants for eta0 calculation
        T = 0  # Reference temperature in Celsius
        k = 0.5  # Scaling factor for the logistic function

        # Define the eta0 calculation function
        def calculate_eta0_temp(temp_values, threshold=T, k_value=k):
            # Apply logistic function
            return 1 / (1 + np.exp(-k_value * (temp_values - threshold)))

        # Apply the eta0 calculation to the temperature values
        eta0_values = calculate_eta0_temp(temp_masked)

        # Retain NaN values from the original mask
        eta0_values = eta0_values.where(~np.isnan(surface_temp), np.nan)

        # Ensure eta0_values has the correct dimensions and coordinates
        eta0_values = eta0_values.broadcast_like(surface_temp)

        # Debugging: Verify shape and range of eta0_values
        print("Shape of eta0_values:", eta0_values.shape)
        print(
            "Temperature eta0 min:",
            eta0_values.min().values,
            "Temperature eta0 max:",
            eta0_values.max().values,
        )

        # Create the DataArray for eta0 values
        eta0_da = xr.DataArray(
            eta0_values.values,  # Use the calculated eta0 values
            coords={
                "time": surface_temp["time"],  # Use the correct 'time' coordinate
                "y": surface_temp["y"],  # Use the 'y' (latitude) dimension
                "x": surface_temp["x"],  # Use the 'x' (longitude) dimension
            },
            dims=["time", "y", "x"],
            name="eta0",
        )

        # Create a new dataset with eta0 values
        new_ds = xr.Dataset({"eta0": eta0_da})

        # Transpose the dataset to ensure 'time' is the first dimension
        new_ds = new_ds.transpose("time", "y", "x")

        # Remove 'grid_mapping' attribute if it exists in the dataset
        for var in new_ds.variables:
            if "grid_mapping" in new_ds[var].attrs:
                del new_ds[var].attrs["grid_mapping"]

        # Save the new dataset to a NetCDF file
        # output_file = 'Efficiency_Temperature_Eta0_Calculation.nc'
        # new_ds.to_netcdf(output_file)

        last_date = str(swe["time"][-1].values)[:10].replace("-", "")

        # Define the output file name with the last date
        output_file = f"Efficiency_SurfTemp_{last_date}.nc"

        # Save the dataset
        new_ds.to_netcdf(output_file)

        print(f"Dataset saved to '{output_file}'")

        return output_file

    def generate_sensor_gcom(self, ds):
        # Extract the SWE variable
        swe = ds["SWE_tavg"]

        # Check the range of SWE values
        print(
            "SWE min (combined):",
            swe.min().values,
            "SWE max (combined):",
            swe.max().values,
        )

        # Define constants for eta2 calculation
        T = 150  # Threshold for the logistic function
        k = 0.03  # Scaling factor for the logistic function
        epsilon = 0.05  # Intercept to ensure eta is bounded below

        # Define the eta2 calculation function with an intercept
        def calculate_eta2(swe_value, threshold=T, k_value=k, intercept=epsilon):
            # Logistic function with intercept
            return intercept + (1 - intercept) / (
                1 + np.exp(k_value * (swe_value - threshold))
            )

        # Mask NaN values to retain regions outside the Missouri Basin as NaN
        swe_masked = swe.where(~np.isnan(swe))

        # Apply the eta2 calculation to SWE values, preserving NaN values
        eta2_values = calculate_eta2(swe_masked)

        # Ensure eta2_values has the correct dimensions
        eta2_values = eta2_values.broadcast_like(swe)  # Align dimensions with SWE

        # Debugging: Verify shape and range of eta2_values
        print("Shape of eta2_values:", eta2_values.shape)
        print(
            "Eta2 min:", eta2_values.min().values, "Eta2 max:", eta2_values.max().values
        )

        # Create the DataArray for eta2 values
        eta2_da = xr.DataArray(
            eta2_values.values,  # Use the raw values of eta2
            coords={
                "time": swe["time"],  # Use the correct 'time' coordinate
                "y": swe["y"],  # Retain 'y' (latitude) dimension
                "x": swe["x"],  # Retain 'x' (longitude) dimension
            },
            dims=["time", "y", "x"],
            name="eta2",
        )

        # Create a new dataset with eta2 values
        new_ds = xr.Dataset({"eta2": eta2_da}).transpose("time", "y", "x")

        # Remove 'grid_mapping' attribute if it exists in the dataset
        for var in new_ds.variables:
            if "grid_mapping" in new_ds[var].attrs:
                del new_ds[var].attrs["grid_mapping"]

        # Save the new dataset to a NetCDF file
        # output_file = 'Efficiency_SWE_Eta2_Calculation_GCOM.nc'
        # new_ds.to_netcdf(output_file)

        last_date = str(swe["time"][-1].values)[:10].replace("-", "")

        # Define the output file name with the last date
        output_file = f"Efficiency_Sensor_GCOM_{last_date}.nc"

        # Save the dataset
        new_ds.to_netcdf(output_file)

        print(f"Dataset saved to '{output_file}'")

        return output_file

    def generate_sensor_capella(self, ds):

        # Extract the SWE variable
        swe = ds["SWE_tavg"]

        # Create a new array filled with 1s, matching the SWE dimensions
        eta2_capella_values = xr.DataArray(
            np.ones_like(swe.values),  # Fill with ones
            coords={
                "time": swe["time"],  # Use the same time coordinate
                "y": swe["y"],  # Use the same latitude dimension
                "x": swe["x"],  # Use the same longitude dimension
            },
            dims=["time", "y", "x"],
            name="eta2",
        )

        # Retain NaN values from the original SWE dataset
        eta2_capella_values = eta2_capella_values.where(~np.isnan(swe), np.nan)

        # Create a new dataset for Capella with eta2 values set to 1, retaining NaN values
        capella_ds = xr.Dataset({"eta2": eta2_capella_values}).transpose(
            "time", "y", "x"
        )

        # Remove 'grid_mapping' attribute if it exists in the dataset
        for var in capella_ds.variables:
            if "grid_mapping" in capella_ds[var].attrs:
                del capella_ds[var].attrs["grid_mapping"]

        # Save the new dataset to a NetCDF file
        # output_file = 'Efficiency_SWE_Eta2_Capella.nc'
        # capella_ds.to_netcdf(output_file)

        last_date = str(swe["time"][-1].values)[:10].replace("-", "")

        # Define the output file name with the last date
        output_file = f"Efficiency_Sensor_Capella_{last_date}.nc"

        # Save the dataset
        capella_ds.to_netcdf(output_file)

        print(f"Capella dataset saved to '{output_file}'")

        return output_file

    def combine_and_multiply_datasets(
        self, ds, eta5_file, eta0_file, eta2_file, weights, output_file
    ):
        """
        Combine three datasets by applying weights and performing grid-cell multiplication.

        Parameters:
            eta5_file (str): Path to the NetCDF file for the eta5 dataset.
            eta0_file (str): Path to the NetCDF file for the eta0 dataset.
            eta2_file (str): Path to the NetCDF file for the eta2 dataset.
            weights (dict): Weights for eta5, eta0, and eta2 (e.g., {'eta5': 0.5, 'eta0': 0.3, 'eta2': 0.2}).
            output_file (str): Path to save the resulting combined dataset.

        Returns:
            None
        """
        # Get SWE
        swe = ds["SWE_tavg"]

        # Load the datasets
        eta5_ds = eta5_file
        eta0_ds = eta0_file
        eta2_ds = eta2_file

        # Extract variables
        eta5 = eta5_ds["eta5"]
        eta0 = eta0_ds["eta0"]
        eta2 = eta2_ds["eta2"]

        # Apply weights to each dataset
        weighted_eta5 = eta5 * weights["eta5"]
        weighted_eta0 = eta0 * weights["eta0"]
        weighted_eta2 = eta2 * weights["eta2"]

        # Multiply each grid cell across the weighted datasets
        combined_values = weighted_eta5 * weighted_eta0 * weighted_eta2

        # Create a new dataset for the combined results
        combined_ds = xr.Dataset({"combined_eta": combined_values})

        # Retain the attributes and coordinates from one of the datasets
        combined_ds["combined_eta"] = combined_ds["combined_eta"].assign_coords(
            {"time": eta5["time"], "y": eta5["y"], "x": eta5["x"]}
        )

        # Save the combined dataset to a NetCDF file
        # combined_ds.to_netcdf(output_file)
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")

        # Define the output file name with the last date
        output_file = f"{output_file}_{last_date}.nc"

        # Save the dataset
        combined_ds.to_netcdf(output_file)
        print(f"Combined dataset saved to '{output_file}'")

        return output_file

    def process(self, ds, mo_basin, gcom_ds, capella_ds):

        # Calculate SWE
        swe = ds["SWE_tavg"]

        # Define dates for ground track analysis
        start = datetime(2019, 3, 9, tzinfo=timezone.utc)
        end = datetime(2019, 3, 10, tzinfo=timezone.utc)
        frame_duration = timedelta(days=1)
        num_frames = int(1 + (end - start) / frame_duration)

        # Define instruments
        amsr2 = Instrument(
            name="AMSR2",
            field_of_regard=utils.swath_width_to_field_of_regard(700e3, 1450e3),
        )
        sar = Instrument(
            name="SAR",
            field_of_regard=utils.swath_width_to_field_of_regard(500e3, 30e3) + 30,
        )

        # Define satellites
        gcom_w = Satellite(
            name="GCOM-W",
            orbit=TwoLineElements(
                tle=[
                    "1 38337U 12025A   24117.59466874  .00002074  00000+0  46994-3 0  9995",
                    "2 38337  98.2005  58.4072 0001734  89.8752  83.0178 14.57143724635212",
                ]
            ),
            instruments=[amsr2],
        )

        capella_14 = Satellite(
            name="Capella_14",
            orbit=TwoLineElements(
                tle=[
                    "1 59444U 24066C   24147.51039534  .00009788  00000+0  10218-2 0  9997",
                    "2 59444  45.6083 186.3601 0001084 293.2330  66.8433 14.90000003  7233",
                ]
            ),
            instruments=[sar],
        )

        # Helper function to compute ground tracks
        def get_ground_tracks(
            start, frame_duration, frame, satellite_instrument_pairs, clip_geo
        ):
            return pd.concat(
                [
                    compute_ground_track(
                        satellite_instrument_pair[0],  # satellite
                        satellite_instrument_pair[1],  # instrument
                        pd.date_range(
                            start + frame * frame_duration,
                            start + (frame + 1) * frame_duration,
                            freq=timedelta(seconds=10),
                        ),
                        crs="EPSG:3857",
                    )
                    for satellite_instrument_pair in satellite_instrument_pairs
                ]
            ).clip(clip_geo)

        # # Load the Missouri Basin boundary
        # mo_basin = gpd.read_file("WBD_10_HU2_Shape/Shape/WBDHU2.shp")
        # mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

        # Compute GCOM and Capella ground tracks
        gcom_tracks = pd.concat(
            Parallel(n_jobs=-1)(
                delayed(get_ground_tracks)(
                    start,
                    frame_duration,
                    frame,
                    [(gcom_w, amsr2)],
                    mo_basin.envelope,
                )
                for frame in range(num_frames)
            ),
            ignore_index=True,
        )

        capella_tracks = pd.concat(
            Parallel(n_jobs=-1)(
                delayed(get_ground_tracks)(
                    start,
                    frame_duration,
                    frame,
                    [(capella_14, sar)],
                    mo_basin.envelope,
                )
                for frame in range(num_frames)
            ),
            ignore_index=True,
        )

        capella_tracks["time"] = pd.to_datetime(capella_tracks["time"]).dt.tz_localize(
            None
        )
        gcom_tracks["time"] = pd.to_datetime(gcom_tracks["time"]).dt.tz_localize(None)

        # Select the ground tracks for the specific date
        capella_tracks = capella_tracks[
            capella_tracks["time"] == pd.to_datetime("2019-03-10")
        ]
        gcom_tracks = gcom_tracks[gcom_tracks["time"] == pd.to_datetime("2019-03-10")]

        # # Load GCOM and Capella efficiency datasets
        # gcom_ds= get_data(s3_filepath="snow-observing-systems/Combined_Efficiency_Weighted_Product_GCOM_20190310.nc")
        # capella_ds= get_data(s3_filepath="snow-observing-systems/Combined_Efficiency_Weighted_Product_Capella_20190310.nc")

        # Extract the second time step
        gcom_eta = gcom_ds["combined_eta"].isel(time=1).rio.write_crs("EPSG:4326")
        capella_eta = capella_ds["combined_eta"].isel(time=1).rio.write_crs("EPSG:4326")

        # Create masks for ground tracks
        if capella_tracks.unary_union.geom_type == "MultiPolygon":
            capella_geometries = [geom for geom in capella_tracks.unary_union.geoms]
        else:
            capella_geometries = [capella_tracks.unary_union]

        if gcom_tracks.unary_union.geom_type == "MultiPolygon":
            gcom_geometries = [geom for geom in gcom_tracks.unary_union.geoms]
        else:
            gcom_geometries = [gcom_tracks.unary_union]

        capella_mask = capella_eta.rio.clip(capella_geometries, drop=False).notnull()
        gcom_mask = gcom_eta.rio.clip(gcom_geometries, drop=False).notnull()

        # Compute final_eta values
        final_eta = xr.full_like(capella_eta, np.nan)  # Initialize with NaN
        final_eta = final_eta.where(capella_mask)  # Retain only Capella-covered areas
        final_eta = final_eta.where(
            ~gcom_mask, capella_eta - gcom_eta
        )  # Subtract where overlap
        final_eta = final_eta.where(
            ~capella_mask, capella_eta
        )  # Retain Capella-only values
        final_eta = final_eta.where(
            capella_mask, np.nan
        )  # Assign NaN outside Capella coverage

        # Convert to polygons
        final_eta_df = final_eta.to_dataframe(name="final_eta").reset_index()

        # Drop rows with missing values
        final_eta_df = final_eta_df.dropna(subset=["x", "y", "final_eta"])

        # Calculate cell size (assume uniform grid)
        cell_size = abs(final_eta["x"].diff(dim="x").mean().values)

        # Create polygons for grid cells
        polygons = [
            box(x, y, x + cell_size, y + cell_size)
            for x, y in zip(final_eta_df["x"], final_eta_df["y"])
        ]

        # Convert to GeoDataFrame with polygons
        final_eta_gdf = gpd.GeoDataFrame(
            final_eta_df,
            geometry=polygons,
            crs="EPSG:4326",
        )

        # Save final_eta_gdf as a GeoJSON file
        # output_geojson = "Final_Eta_GDF.geojson"
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")

        # Define the output file name with the last date
        output_file = f"Reward_{last_date}.geojson"

        # Save the dataset
        final_eta_gdf.to_file(output_file, driver="GeoJSON")
        print(f"Final eta GeoDataFrame with polygons saved to '{output_file}'")

        return output_file, gcom_tracks, capella_tracks

    def find_optimal_solution(self, final_eta_gdf):

        # Define the number of cells to select
        N = 150  # Maximum number of cells to select

        # Ensure there are no NaN or Inf values in the final rewards
        final_eta_gdf["final_eta"] = final_eta_gdf["final_eta"].replace(
            [np.inf, -np.inf], np.nan
        )
        final_eta_gdf = final_eta_gdf.dropna(subset=["final_eta"])

        # Initialize the optimization model
        model = LpProblem("Final_Eta_Optimization", LpMaximize)

        # Add binary decision variables for each cell
        x = {i: LpVariable(f"x_{i}", cat="Binary") for i in final_eta_gdf.index}

        # Set the objective to maximize the sum of selected `final_eta` values
        objective = lpSum(
            x[i] * final_eta_gdf.loc[i, "final_eta"] for i in final_eta_gdf.index
        )
        model += objective

        # Add a constraint to limit the selection to N cells
        model += lpSum(x[i] for i in final_eta_gdf.index) <= N, "Max_Selections"

        # Solve the model
        model.solve()

        # Save the selected cells
        if value(model.objective) is not None:
            print("Optimal solution found.")
            selected_blocks = []

            for i in final_eta_gdf.index:
                if x[i].value() > 0.5:
                    selected_blocks.append(
                        {
                            "geometry": final_eta_gdf.loc[i, "geometry"],
                            "final_eta": final_eta_gdf.loc[i, "final_eta"],
                        }
                    )
            # Convert to GeoDataFrame
            selected_blocks_gdf = gpd.GeoDataFrame(selected_blocks, crs="EPSG:4326")

            # Save the selected cells to a GeoJSON file
            output_geojson = "Selected_Cells_Optimization.geojson"
            selected_blocks_gdf.to_file(output_geojson, driver="GeoJSON")
            print(f"Selected cells saved to '{output_geojson}'.")
        else:
            print("No optimal solution found.")

        return output_geojson

    def get_data_geojson(self, s3_filepath):
        print(f"Starting get_data_geojson with s3_filepath: {s3_filepath}")
        try:
            # Start a new session
            fs = s3fs.S3FileSystem(anon=False)
            print("S3FileSystem initialized.")

            # Get the file contents
            fgrab = fs.open(s3_filepath)
            print("File opened from S3.")

            # Read the file into a GeoDataFrame
            geojson_data = gpd.read_file(fgrab)
            print("GeoDataFrame created from file.")

            return geojson_data
        except Exception as e:
            print(f"An error occurred in get_data_geojson: {e}")
            raise

    def get_data_geojson_alternative(self, s3, bucket_name, key):

        # Get the object from the bucket
        response = s3.get_object(Bucket=bucket_name, Key=key)

        # Read the contents of the file
        file_contents = response["Body"].read()

        # Parse the contents as GeoJSON
        final_eta_gdf = gpd.read_file(BytesIO(file_contents))

        return final_eta_gdf

    def get_missouri_basin(self, file_path: str) -> gpd.GeoSeries:
        """
        Get Missouri Basin geometry

        Args:
            file_path (str): File path to the shapefile

        Returns:
            gpd.GeoSeries: GeoSeries of the Missouri Basin geometry
        """
        # Load the shapefile
        logger.info(f"Loading Missouri Basin boundary.")
        mo_basin = gpd.read_file(file_path)
        logger.info(f"Loading Missouri Basin boundary successfully completed.")
        return gpd.GeoSeries(
            Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326"
        )

    def get_us_states(self, file_path: str) -> gpd.GeoDataFrame:
        """
        Get US states geometry

        Args:
            file_path (str): File path to the shapefile

        Returns:
            gpd.GeoDataFrame: GeoDataFrame of the US states geometry
        """
        # Load the shapefile
        logger.info(f"Loading US states boundary.")
        us_map = gpd.read_file(file_path)
        logger.info(f"Loading US states boundary successfully completed.")
        return us_map[~us_map.STUSPS.isin(["AK", "HI", "PR"])].to_crs("EPSG:4326")

    def open_polygons(self, geojson_path):
        geojson = gpd.read_file(geojson_path)
        polygons = geojson.geometry
        print("Polygons loaded.")
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
        geo_transform = dataset["spatial_ref"].GeoTransform.split()
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

    def encode(
        self,
        dataset,
        variable,
        output_path,
        time_step,
        scale,
        geojson_path,
        downsample_factor=1,
        rotate=False,
    ):

        # logger.info('Encoding snow layer.')
        polygons = self.open_polygons(geojson_path=geojson_path)

        raster_layer = dataset[variable]

        raster_layer = raster_layer.rio.write_crs("EPSG:4326")
        clipped_layer = raster_layer.rio.clip(polygons, all_touched=True)
        # print(clipped_layer)

        raster_layer = clipped_layer.isel(time=time_step)
        # if scale == 'time':
        #     raster_layer = clipped_layer.sel(time=time_step, method='nearest')
        # elif scale == 'week':
        #     raster_layer = clipped_layer.isel(week=time_step).values
        # elif scale == 'month':
        #     raster_layer = clipped_layer.isel(month=time_step).values

        # raster_layer = downsample_array(raster_layer, downsample_factor=downsample_factor)

        raster_layer_min = np.nanmin(raster_layer)
        raster_layer_max = np.nanmax(raster_layer)

        na_mask = np.isnan(raster_layer)

        if raster_layer_max > raster_layer_min:
            normalized_layer = (raster_layer - raster_layer_min) / (
                raster_layer_max - raster_layer_min
            )
        else:
            normalized_layer = np.zeros_like(raster_layer)

        colormap = plt.get_cmap("Blues_r")
        rgba_image = colormap(normalized_layer)

        rgba_image[..., 3] = np.where(na_mask, 0, 1)

        rgba_image = (rgba_image * 255).astype(np.uint8)

        if rotate:
            # Rotate the image about the x-axis by 180 degrees
            rgba_image = np.flipud(rgba_image)

        image = Image.fromarray(rgba_image, "RGBA")
        image.save(output_path)
        try:
            if rotate:
                bottom_left, bottom_right, top_left, top_right = self.get_extents(
                    dataset, variable=variable
                )
            else:
                top_left, top_right, bottom_left, bottom_right = self.get_extents(
                    dataset, variable=variable
                )
        except:
            top_left = top_right = bottom_left = bottom_right = None

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        raster_layer_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # logger.info('Encoding snow layer successfully completed.')

        return raster_layer_encoded, top_left, top_right, bottom_left, bottom_right

    def download_file(self, s3, bucket, key, filename):
        """
        Download a file from an S3 bucket

        Args:
            s3: S3 client
            bucket: S3 bucket name
            key: S3 object key
            filename: Filename to save the file as
        """
        config = TransferConfig(use_threads=False)
        s3.download_file(Bucket=bucket, Key=key, Filename=filename, Config=config)
        # Open the file
        dataset = xr.open_dataset(filename)  # , engine="h5netcdf")
        return dataset

    def download_geojson(self, s3, bucket, key, filename):
        """
        Download a GeoJSON file from an S3 bucket

        Args:
            s3: S3 client
            bucket: S3 bucket name
            key: S3 object key
            filename: Filename to save the file as
        """
        config = TransferConfig(use_threads=False)
        s3.download_file(Bucket=bucket, Key=key, Filename=filename, Config=config)
        # Open the file
        dataset = gpd.read_file(filename)
        return dataset

    def on_change(self, source, property_name, old_value, new_value):
        """
        *Standard on_change callback function format inherited from Observer object class*

        In this instance, the callback function checks when the **PROPERTY_MODE** switches to **EXECUTING** to send a :obj:`GroundLocation` message to the *PREFIX/ground/location* topic:

            .. literalinclude:: /../../NWISdemo/grounds/main_ground.py
                :lines: 56-67

        """
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            # Create a boto3 session
            session = self.start_session()

            # Get a session token
            token = self.get_session_token(session=session, mfa_required=False)
            session_token, secret_access_key, access_key_id = self.decompose_token(
                token
            )

            # Create an S3 client
            s3 = session.client(
                "s3",
                aws_session_token=session_token,
                aws_secret_access_key=secret_access_key,
                aws_access_key_id=access_key_id,
            )

            # Load the Missouri River Basin boundary
            mo_basin = self.get_missouri_basin(
                file_path="data/Downloaded_files/Mo_basin_shp/WBD_10_HU2_Shape/Shape/WBDHU2.shp"
            )

            ##################
            # Combined dataset#
            ##################
            combined_output_file = "LIS_dataset_20190310.nc"

            # # Load the combined dataset
            # combined_dataset = self.get_data(s3_filepath=f"snow-observing-systems/{combined_output_file}")

            combined_dataset = self.download_file(
                s3=s3,
                bucket="snow-observing-systems",
                key=combined_output_file,
                filename=combined_output_file,
            )
            print(combined_dataset)

            # # # # Select the SWE_tavg variable for a specific time step (e.g., first time step)
            # # # swe_data = combined_dataset["SWE_tavg"].isel(time=0) # SEND AS MESSAGE

            # swe_layer_encoded, top_left, top_right, bottom_left, bottom_right = (
            #     self.encode(
            #         dataset=combined_dataset,
            #         variable="SWE_tavg",
            #         output_path="swe_data.png",
            #         time_step=0,
            #         scale="time",
            #         geojson_path="WBD_10_HU2_4326.geojson",
            #         rotate=True,
            #     )
            # )
            # del combined_dataset

            # self.app.send_message(
            #     self.app.app_name,
            #     "layer",
            #     SWEChangeLayer(
            #         swe_change_layer=swe_layer_encoded,
            #         top_left=top_left,
            #         top_right=top_right,
            #         bottom_left=bottom_left,
            #         bottom_right=bottom_right,
            #     ).json(),
            # )
            # logger.info("Publishing message successfully completed.")
            # time.sleep(5)

            # # ##############
            # # #ETA5 dataset#
            # # ##############
            # swe_output_file = "Efficiency_SWE_Change_20190310.nc"

            # # # Load the clipped dataset
            # # eta5_file = self.get_data(s3_filepath=f"snow-observing-systems/{swe_output_file}")

            # eta5_file = self.download_file(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=swe_output_file,
            #     filename=swe_output_file,
            # )

            # # # Select the eta5 variable for a specific time step (e.g., first time step)
            # # eta5_data = eta5_file["eta5"].isel(time=1)

            # eta5_layer_encoded, _, _, _, _ = self.encode(
            #     dataset=eta5_file,
            #     variable="eta5",
            #     output_path="eta5_data.png",
            #     time_step=1,
            #     scale="time",
            #     geojson_path="WBD_10_HU2_4326.geojson",
            #     rotate=True,
            # )
            # del eta5_file

            # self.app.send_message(
            #     self.app.app_name,
            #     "layer",
            #     SWEChangeLayer(
            #         swe_change_layer=eta5_layer_encoded,
            #         top_left=top_left,
            #         top_right=top_right,
            #         bottom_left=bottom_left,
            #         bottom_right=bottom_right,
            #     ).json(),
            # )
            # logger.info("Publishing message successfully completed.")
            # time.sleep(5)

            # # ##############
            # # #ETA0 dataset#
            # # ##############
            # surfacetemp_output_file = "Efficiency_SurfTemp_20190310.nc"

            # # # Load the clipped dataset
            # # eta0_file = self.get_data(s3_filepath=f"snow-observing-systems/{surfacetemp_output_file}")

            # eta0_file = self.download_file(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=surfacetemp_output_file,
            #     filename=surfacetemp_output_file,
            # )

            # # # Select the eta0 variable for a specific time step (e.g., first time step)
            # # eta0_data = eta0_file["eta0"].isel(time=1)

            # eta0_layer_encoded, _, _, _, _ = self.encode(
            #     dataset=eta0_file,
            #     variable="eta0",
            #     output_path="eta0_data.png",
            #     time_step=1,
            #     scale="time",
            #     geojson_path="WBD_10_HU2_4326.geojson",
            #     rotate=True,
            # )
            # del eta0_file

            # self.app.send_message(
            #     self.app.app_name,
            #     "layer",
            #     SWEChangeLayer(
            #         swe_change_layer=eta0_layer_encoded,
            #         top_left=top_left,
            #         top_right=top_right,
            #         bottom_left=bottom_left,
            #         bottom_right=bottom_right,
            #     ).json(),
            # )
            # logger.info("Publishing message successfully completed.")
            # time.sleep(5)

            # # ###################
            # # #ETA2 GCOM dataset#
            # # ###################
            # sensor_gcom_output_file = "Efficiency_Sensor_GCOM_20190310.nc"

            # # # Load the clipped dataset
            # # eta2_file_GCOM = self.get_data(s3_filepath=f"snow-observing-systems/{sensor_gcom_output_file}")

            # eta2_file_GCOM = self.download_file(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=sensor_gcom_output_file,
            #     filename=sensor_gcom_output_file,
            # )

            # # # Select the eta2 variable for a specific time step (e.g., first time step)
            # # eta2_data_GCOM = eta2_file_GCOM["eta2"].isel(time=1)

            # eta2_gcom_layer_encoded, _, _, _, _ = self.encode(
            #     dataset=eta2_file_GCOM,
            #     variable="eta2",
            #     output_path="eta2_gcom_data.png",
            #     time_step=1,
            #     scale="time",
            #     geojson_path="WBD_10_HU2_4326.geojson",
            #     rotate=True,
            # )
            # del eta2_file_GCOM

            # self.app.send_message(
            #     self.app.app_name,
            #     "layer",
            #     SWEChangeLayer(
            #         swe_change_layer=eta2_gcom_layer_encoded,
            #         top_left=top_left,
            #         top_right=top_right,
            #         bottom_left=bottom_left,
            #         bottom_right=bottom_right,
            #     ).json(),
            # )
            # logger.info("Publishing message successfully completed.")
            # time.sleep(5)

            # # ######################
            # # #ETA2 Capella dataset#
            # # ######################
            # sensor_capella_output_file = "Efficiency_Sensor_Capella_20190310.nc"

            # # # Load the clipped dataset
            # # eta2_file_Capella = self.get_data(s3_filepath=f"snow-observing-systems/{sensor_capella_output_file}")

            # eta2_file_Capella = self.download_file(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=sensor_capella_output_file,
            #     filename=sensor_capella_output_file,
            # )

            # # # Select the eta2 variable for a specific time step (e.g., first time step)
            # # eta2_data_Capella = eta2_file_Capella["eta2"].isel(time=1)

            # eta2_capella_layer_encoded, _, _, _, _ = self.encode(
            #     dataset=eta2_file_Capella,
            #     variable="eta2",
            #     output_path="eta2_capella_data.png",
            #     time_step=1,
            #     scale="time",
            #     geojson_path="WBD_10_HU2_4326.geojson",
            #     rotate=True,
            # )
            # del eta2_file_Capella

            # self.app.send_message(
            #     self.app.app_name,
            #     "layer",
            #     SWEChangeLayer(
            #         swe_change_layer=eta2_capella_layer_encoded,
            #         top_left=top_left,
            #         top_right=top_right,
            #         bottom_left=bottom_left,
            #         bottom_right=bottom_right,
            #     ).json(),
            # )
            # logger.info("Publishing message successfully completed.")
            # time.sleep(5)

            # # ##########
            # # #GCOM ETA#
            # # ##########
            # gcom_combine_multiply_output_file = (
            #     "Combined_Efficiency_Weighted_Product_GCOM_20190310.nc"
            # )

            # # # Load the clipped dataset
            # # gcom_dataset = self.get_data(s3_filepath=f"snow-observing-systems/{gcom_combine_multiply_output_file}")

            # gcom_dataset = self.download_file(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=gcom_combine_multiply_output_file,
            #     filename=gcom_combine_multiply_output_file,
            # )

            # # # Select the combined_eta variable for a specific time step (e.g., first time step)
            # # gcom_eta = gcom_dataset["combined_eta"].isel(time=1)

            # gcom_eta_layer_encoded, _, _, _, _ = self.encode(
            #     dataset=gcom_dataset,
            #     variable="combined_eta",
            #     output_path="gcom_eta_combined_data.png",
            #     time_step=1,
            #     scale="time",
            #     geojson_path="WBD_10_HU2_4326.geojson",
            #     rotate=True,
            # )
            # del gcom_dataset

            # self.app.send_message(
            #     self.app.app_name,
            #     "layer",
            #     SWEChangeLayer(
            #         swe_change_layer=gcom_eta_layer_encoded,
            #         top_left=top_left,
            #         top_right=top_right,
            #         bottom_left=bottom_left,
            #         bottom_right=bottom_right,
            #     ).json(),
            # )
            # logger.info("Publishing message successfully completed.")
            # time.sleep(5)

            # # #############
            # # #Capella ETA#
            # # #############
            # capella_combine_multiply_output_file = (
            #     "Combined_Efficiency_Weighted_Product_Capella_20190310.nc"
            # )

            # # # Load the clipped dataset
            # # capella_dataset= self.get_data(s3_filepath=f"snow-observing-systems/{capella_combine_multiply_output_file}")

            # capella_dataset = self.download_file(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=capella_combine_multiply_output_file,
            #     filename=capella_combine_multiply_output_file,
            # )

            # # # Select the combined_eta variable for a specific time step (e.g., first time step)
            # # capella_eta = capella_dataset["combined_eta"].isel(time=1)

            # capella_eta_layer_encoded, _, _, _, _ = self.encode(
            #     dataset=capella_dataset,
            #     variable="combined_eta",
            #     output_path="capella_eta_combined_data.png",
            #     time_step=1,
            #     scale="time",
            #     geojson_path="WBD_10_HU2_4326.geojson",
            #     rotate=True,
            # )
            # del capella_dataset

            # self.app.send_message(
            #     self.app.app_name,
            #     "layer",
            #     SWEChangeLayer(
            #         swe_change_layer=capella_eta_layer_encoded,
            #         top_left=top_left,
            #         top_right=top_right,
            #         bottom_left=bottom_left,
            #         bottom_right=bottom_right,
            #     ).json(),
            # )
            # logger.info("Publishing message successfully completed.")
            # # time.sleep(5)

            # # ###########
            # # #Final ETA#
            # # ###########
            # logger.info("Loading all cells polygon...")
            # final_eta_output_file = "Reward_20190310.geojson"

            # # Load the GeoDataFrame
            # # final_eta_gdf = self.get_data_geojson(s3_filepath=f"snow-observing-systems/{final_eta_output_file}")
            # final_eta_gdf = self.download_geojson(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=final_eta_output_file,
            #     filename=final_eta_output_file,
            # )
            # # final_eta_gdf = get_data_geojson_alternative(s3=s3, bucket_name='snow-observing-systems', key='Reward_20190310.geojson')

            # # Clip Final Eta GDF and ground tracks to the Missouri Basin
            # final_eta_gdf_clipped = final_eta_gdf  # gpd.clip(final_eta_gdf, mo_basin)

            # # Convert the clipped GeoDataFrame to GeoJSON and send as message
            # all_json_data = final_eta_gdf_clipped.drop(
            #     "time", axis=1, errors="ignore"
            # ).to_json()
            # self.app.send_message(
            #     self.app.app_name,
            #     "all",
            #     VectorLayer(vector_layer=all_json_data).json(),
            # )
            # logger.info("(ALL) Publishing message successfully completed.")
            # time.sleep(5)

            # # #######################
            # # #Find Optimal Solution#
            # # #######################
            # print("Loading selected cells polygon...")
            # output_geojson = "Selected_Cells_Optimization.geojson"

            # # Load the GeoDataFrame
            # # final_eta_gdf = gpd.read_file("Final_Eta_GDF.geojson")  # All cells with final_eta values
            # # selected_cells_gdf = self.get_data_geojson(s3_filepath=f"snow-observing-systems/{output_geojson}") #gpd.read_file("Selected_Cells_Optimization.geojson")  # Selected cells
            # selected_cells_gdf = self.download_geojson(
            #     s3=s3,
            #     bucket="snow-observing-systems",
            #     key=output_geojson,
            #     filename=output_geojson,
            # )
            # # Clip Final Eta GDF and ground tracks to the Missouri Basin
            # selected_cells_gdf_clipped = (
            #     selected_cells_gdf  # gpd.clip(selected_cells_gdf, mo_basin)
            # )

            # # Convert the clipped GeoDataFrame to GeoJSON and send as message
            # selected_json_data = selected_cells_gdf_clipped.drop(
            #     "time", axis=1, errors="ignore"
            # ).to_json()
            # self.app.send_message(
            #     self.app.app_name,
            #     "selected",
            #     VectorLayer(vector_layer=selected_json_data).json(),
            # )
            # logger.info("(SELECTED) Publishing message successfully completed.")
            # time.sleep(5)

            # # # Ensure CRS matches for consistency
            # # selected_cells_gdf = selected_cells_gdf.to_crs(mo_basin.crs)
            # # final_eta_gdf = final_eta_gdf.to_crs(mo_basin.crs)
            # s3.close()


def main():
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")

    # Define the simulation parameters
    NAME = "swe_change"

    # create the managed application
    app = ManagedApplication(NAME)

    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app))

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        config.rc.simulation_configuration.execution_parameters.prefix,
        config,
        # True,
        # time_status_step=timedelta(seconds=10) * SCALE,
        # time_status_init=datetime(2024, 1, 7, tzinfo=timezone.utc),
        # time_step=timedelta(seconds=1) * SCALE,
        # # shut_down_when_terminated=True,
    )


if __name__ == "__main__":
    main()
