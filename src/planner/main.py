import base64
import io
import logging
import os
import time
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
from constellation_config_files.schemas import SWEChangeLayer, VectorLayer
from nost_tools.application_utils import ShutDownObserver
from nost_tools.config import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import Observer
from PIL import Image
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from rasterio.features import geometry_mask
from scipy.interpolate import griddata
from shapely.geometry import Polygon, box
from tatc import utils
from tatc.analysis import collect_orbit_track, compute_ground_track
from tatc.schemas import (
    Instrument,
    PointedInstrument,
    Satellite,
    SunSynchronousOrbit,
    TwoLineElements,
    WalkerConstellation,
)
from tatc.utils import swath_width_to_field_of_regard, swath_width_to_field_of_view

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
        # Determine the output file name based on the last date
        last_date_ds1 = np.datetime_as_string(
            dataset1.time[-1].values, unit="D"
        ).replace("-", "")
        last_date_ds2 = np.datetime_as_string(
            dataset2.time[-1].values, unit="D"
        ).replace("-", "")
        last_date = max(last_date_ds1, last_date_ds2)
        output_file = f"LIS_dataset_{last_date}.nc"

        # Check if the file already exists
        if os.path.exists(output_file):
            logger.info(
                f"File {output_file} already exists. Reading the existing file."
            )
            clipped_dataset = xr.open_dataset(output_file)
            return output_file, clipped_dataset

        logger.info("Combining the two datasets.")

        # Define coordinates for the regular grid
        time_coords_ds1 = np.array([dataset1.time[0].values])
        time_coords_ds2 = np.array([dataset2.time[0].values])
        lat_coords = np.linspace(37.024602, 49.739086, 29)
        lon_coords = np.linspace(-113.938141, -90.114221, 40)

        # Variables to interpolate
        variables_to_interpolate = ["SWE_tavg", "AvgSurfT_tavg"]

        # Interpolate datasets
        new_ds1 = self.interpolate_dataset(
            dataset1, variables_to_interpolate, lat_coords, lon_coords, time_coords_ds1
        )
        new_ds2 = self.interpolate_dataset(
            dataset2, variables_to_interpolate, lat_coords, lon_coords, time_coords_ds2
        )

        # Combine the datasets along the time dimension
        combined_dataset = xr.concat([new_ds1, new_ds2], dim="time")

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

        clipped_dataset.to_netcdf(output_file)
        logger.info(
            f"Combining the two datasets successfully completed. Final clipped dataset saved to: {output_file}."
        )

        return output_file, clipped_dataset

    def calculate_eta(self, swe_change, threshold, k_value):
        # Apply logistic function directly, keeping zeros
        return 1 / (1 + np.exp(-k_value * (swe_change - threshold)))

    def generate_swe_difference(self, ds):
        logger.info("Generating the SWE difference dataset.")
        # Extract the SWE variable
        swe = ds["SWE_tavg"]

        # Debugging: Print shape of SWE
        # print("Shape of SWE:", swe.shape)

        # Mask NaN values only (retain zero values for calculation)
        swe_masked = swe.where(~np.isnan(swe))

        # Calculate the SWE difference between consecutive time steps
        swe_diff = swe_masked.diff(dim="time", label="lower")

        # Debugging: Print shape of SWE_diff
        # print("Shape of SWE_diff:", swe_diff.shape)

        # Take the absolute value of differences
        swe_diff_abs = abs(swe_diff)

        # Add a zero difference for the first time step to match the length
        zero_diff = xr.full_like(swe.isel(time=0), fill_value=0)
        swe_diff_abs = xr.concat([zero_diff, swe_diff_abs], dim="time")

        # Debugging: Print shape of SWE_diff_abs
        # print("Shape of SWE_diff_abs:", swe_diff_abs.shape)

        # Ensure `time` dimension alignment
        swe_diff_abs = swe_diff_abs.assign_coords(time=swe["time"])

        # Define constants for eta calculation
        T = 10  # Threshold for the logistic function
        k = 0.2  # Scaling factor for the logistic function

        # Apply the eta calculation to SWE changes
        eta5_values = self.calculate_eta(swe_diff_abs, T, k)

        # Debugging: Print shape of eta5_values
        # print("Shape of eta5_values:", eta5_values.shape)

        # Retain NaN values from the original mask (areas outside the SWE dataset)
        eta5_values = eta5_values.where(~np.isnan(swe_masked), np.nan)

        # Ensure eta5_values has the correct dimensions
        eta5_values = eta5_values.broadcast_like(swe)  # Align dimensions with SWE

        # Debugging: Verify shape after broadcasting
        # print("Shape of eta5_values (after broadcasting):", eta5_values.shape)

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

        # print(f"Dataset saved to '{output_file}'")

        # Check min and max of the eta values
        # print("eta5 min:", eta5_da.min().values, "eta5 max:", eta5_da.max().values)
        logger.info("Generating the SWE difference dataset successfully completed.")
        return output_file, new_ds

    def generate_surface_temp(self, ds):
        logger.info("Generating surface temperature dataset.")
        # Extract the SWE variable
        swe = ds["SWE_tavg"]

        # Extract the surface temperature variable (AvgSurfT_tavg) and convert from Kelvin to Celsius
        surface_temp = ds["AvgSurfT_tavg"] - 273.15

        # # Debugging: Print shape and range of surface temperature
        # print("Shape of Surface Temperature:", surface_temp.shape)
        # print(
        #     "Surface Temperature min:",
        #     surface_temp.min().values,
        #     "max:",
        #     surface_temp.max().values,
        # )

        # Mask NaN values (retain NaN outside valid regions)
        temp_masked = surface_temp.where(~np.isnan(surface_temp))

        # Define constants for eta0 calculation
        T = 0  # Reference temperature in Celsius
        k = 0.5  # Scaling factor for the logistic function

        # Define the eta0 calculation function
        def calculate_eta0_temp(temp_values, threshold=T, k_value=k):
            # Apply logistic function
            # return 1 / (1 + np.exp(-k_value * (temp_values - threshold)))
            return 1 / (1 + np.exp(k_value * (temp_values - threshold)))

        # Apply the eta0 calculation to the temperature values
        eta0_values = calculate_eta0_temp(temp_masked)

        # Retain NaN values from the original mask
        eta0_values = eta0_values.where(~np.isnan(surface_temp), np.nan)

        # Ensure eta0_values has the correct dimensions and coordinates
        eta0_values = eta0_values.broadcast_like(surface_temp)

        # # Debugging: Verify shape and range of eta0_values
        # print("Shape of eta0_values:", eta0_values.shape)
        # print(
        #     "Temperature eta0 min:",
        #     eta0_values.min().values,
        #     "Temperature eta0 max:",
        #     eta0_values.max().values,
        # )

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

        # print(f"Dataset saved to '{output_file}'")
        logger.info("Generating surface temperature dataset successfully completed.")

        return output_file, new_ds

    def generate_sensor_gcom(self, ds):
        logger.info("Generating GCOM efficiency dataset.")
        # Extract the SWE variable
        swe = ds["SWE_tavg"]

        # # Check the range of SWE values
        # print(
        #     "SWE min (combined):",
        #     swe.min().values,
        #     "SWE max (combined):",
        #     swe.max().values,
        # )

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

        # # Debugging: Verify shape and range of eta2_values
        # print("Shape of eta2_values:", eta2_values.shape)
        # print(
        #     "Eta2 min:", eta2_values.min().values, "Eta2 max:", eta2_values.max().values
        # )

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

        # print(f"Dataset saved to '{output_file}'")
        logger.info("Generating GCOM efficiency dataset successfully completed.")

        return output_file, new_ds

    def generate_sensor_capella(self, ds):
        logger.info("Generating Capella efficiency dataset.")
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

        # print(f"Capella dataset saved to '{output_file}'")
        logger.info("Generating Capella efficiency dataset successfully completed.")

        return output_file, capella_ds

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
        logger.info("Combining and multiplying the datasets.")
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
        # print(f"Combined dataset saved to '{output_file}'")
        logger.info("Combining and multiplying the datasets successfully completed.")

        return output_file, combined_ds

    # def multipolygon_to_polygon(self, mo_basin):
    #     # Assuming `multi_poly` is your MultiPolygon object
    #     multi_poly = mo_basin.iloc[0].geometry

    #     # Check if it's a MultiPolygon
    #     if isinstance(multi_poly, MultiPolygon):
    #         # Extract the first Polygon (or handle as needed)
    #         poly = multi_poly.geoms[0]
    #     else:
    #         poly = multi_poly

    #     return poly

    def process(self, gcom_ds, snowglobe_ds, mo_basin, start, end):
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
        logger.info("Generating final efficiency dataset.")
        # Define dates for ground track analysis
        # start = datetime(2019, 3, 9, tzinfo=timezone.utc)
        # end = datetime(2019, 3, 10, tzinfo=timezone.utc)
        duration = timedelta(days=1)
        frame_duration = timedelta(days=1)
        num_frames = int(1 + (end - start) / duration)

        roll_angle = (30 + 33.5) / 2
        roll_range = 33.5 - 30

        logger.info("Specifying constellation.")
        constellation = WalkerConstellation(
            name="SnowGlobe Ku",
            orbit=SunSynchronousOrbit(
                altitude=555e3,
                equator_crossing_time="06:00:30",
                equator_crossing_ascending=False,
                # epoch=datetime(2019, 3, 7, tzinfo=timezone.utc),  # start,
                epoch=datetime(2019, 3, 1, tzinfo=timezone.utc),
            ),
            number_planes=1,
            number_satellites=5,
            instruments=[
                PointedInstrument(
                    name="SnowGlobe Ku-SAR",
                    roll_angle=-roll_angle,
                    field_of_regard=2 * roll_angle
                    + swath_width_to_field_of_regard(555e3, 50e3),
                    along_track_field_of_view=swath_width_to_field_of_view(
                        555e3, 50e3, 0
                    ),
                    cross_track_field_of_view=roll_range
                    + swath_width_to_field_of_view(555e3, 50e3, roll_angle),
                    is_rectangular=True,
                )
            ],
        )
        logger.info("Specifying constellation successfully completed.")

        time_step = timedelta(seconds=5)
        # sim_times = pd.date_range(start, start + duration, freq=time_step)
        # sim_times = pd.date_range(start, end, freq=time_step)
        sim_times = pd.date_range(start, end + duration, freq=time_step)

        # Compute orbit tracks using vectorized operations
        logger.info("Computing orbit tracks.")
        orbit_tracks = pd.concat(
            [
                collect_orbit_track(
                    satellite=satellite,
                    times=sim_times,
                    mask=self.polygons[0],
                )
                for satellite in constellation.generate_members()
            ]
        )
        logger.info("Computing orbit tracks successfully completed.")

        # Compute ground tracks using vectorized operations
        logger.info("Computing ground tracks (P1).")
        ground_tracks = pd.concat(
            [
                compute_ground_track(
                    satellite=satellite,
                    times=sim_times,
                    mask=self.polygons[0],
                    crs="spice",
                )
                for satellite in constellation.generate_members()
            ],
            ignore_index=True,
        )
        logger.info("Computing ground tracks (P1) successfully completed.")

        # Define instrument
        amsr2 = Instrument(
            name="AMSR2",
            field_of_regard=utils.swath_width_to_field_of_regard(700e3, 1450e3),
        )

        # Define satellite using TLE
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

        satellite_instrument_pairs = [(gcom_w, amsr2)]

        # Function to compute ground tracks
        def get_ground_tracks(
            start, frame_duration, frame, satellite_instrument_pairs, clip_geo, mask
        ):
            return pd.concat(
                [
                    compute_ground_track(
                        gcom_w,  # satellite
                        pd.date_range(
                            start + frame * frame_duration,
                            start + (frame + 1) * frame_duration,
                            freq=timedelta(seconds=10),
                        ),
                        crs="EPSG:3857",
                        mask=mask,
                    )
                    for satellite_instrument_pair in satellite_instrument_pairs
                ]
            ).clip(clip_geo)

        # Compute ground tracks using vectorized operations
        logger.info("Computing ground tracks (P2).")
        gcom_tracks = pd.concat(
            [
                get_ground_tracks(
                    start,
                    frame_duration,
                    frame,
                    satellite_instrument_pairs,
                    mo_basin.envelope,
                    mask=self.polygons[0],
                )
                for frame in range(num_frames)
            ],
            ignore_index=True,
        )
        logger.info("Computing ground tracks (P2) successfully completed.")
        gcom_tracks["time"] = pd.to_datetime(gcom_tracks["time"]).dt.tz_localize(None)

        # Select the ground tracks for the specific date
        gcom_tracks = gcom_tracks[gcom_tracks["time"] == end]

        # Extract the second time step
        gcom_eta = gcom_ds["combined_eta"].isel(time=1).rio.write_crs("EPSG:4326")
        snowglobe_eta = (
            snowglobe_ds["combined_eta"].isel(time=1).rio.write_crs("EPSG:4326")
        )

        gcom_union = (
            gcom_tracks.union_all()  # .unary_union
        )  # or gcom_tracks.geometry.union_all() in newer versions
        snowglobe_union = ground_tracks.union_all()  # .unary_union

        gcom_geometries = (
            [gcom_union]
            if gcom_union.geom_type == "Polygon"
            else list(gcom_union.geoms)
        )
        snowglobe_geometries = (
            [snowglobe_union]
            if snowglobe_union.geom_type == "Polygon"
            else list(snowglobe_union.geoms)
        )

        gcom_mask = xr.full_like(snowglobe_eta, False, dtype=bool)
        snowglobe_mask = xr.full_like(snowglobe_eta, False, dtype=bool)

        gcom_mask.values = geometry_mask(
            geometries=gcom_geometries,
            out_shape=snowglobe_eta.shape,
            transform=snowglobe_eta.rio.transform(),
            invert=True,
        )

        snowglobe_mask.values = geometry_mask(
            geometries=snowglobe_geometries,
            out_shape=snowglobe_eta.shape,
            transform=snowglobe_eta.rio.transform(),
            invert=True,
        )
        logger.info("Creating masks for ground tracks successfully completed.")

        # Compute final_eta values
        logger.info("Computing final efficiency values.")
        # Perform exact spatial intersection (keeps only cells inside mo_basin)
        # final_eta_gdf = gpd.overlay(final_eta_gdf, mo_basin, how="intersection")
        final_eta = xr.full_like(snowglobe_eta, np.nan)
        final_eta = final_eta.where(snowglobe_mask)  # keep Capella area
        final_eta = final_eta.where(
            ~gcom_mask, snowglobe_eta - gcom_eta
        )  # subtract where overlap
        final_eta = final_eta.where(~snowglobe_mask, snowglobe_eta)  # keep Capella-only
        final_eta = final_eta.where(snowglobe_mask, np.nan)

        # Add time coords
        final_last_date = snowglobe_eta["time"].values
        final_eta = final_eta.assign_coords(time=final_last_date)

        # Convert final_eta to polygons (treat x,y as centers -> edges)

        final_eta_df = final_eta.to_dataframe(name="final_eta").reset_index()
        final_eta_df = final_eta_df.dropna(subset=["x", "y", "final_eta"])

        # Separate X & Y resolutions (assuming uniform spacing)
        x_res = abs(final_eta["x"].diff(dim="x").mean().values)
        y_res = abs(final_eta["y"].diff(dim="y").mean().values)

        # Build polygons from center coords
        polygons = []
        for row in final_eta_df.itertuples():
            x_center, y_center = row.x, row.y
            left = x_center - x_res / 2
            right = x_center + x_res / 2
            bottom = y_center - y_res / 2
            top = y_center + y_res / 2
            polygons.append(box(left, bottom, right, top))

        final_eta_gdf = gpd.GeoDataFrame(
            final_eta_df,
            geometry=polygons,
            crs="EPSG:4326",
        )

        final_eta_gdf["time"] = pd.Timestamp(final_last_date)

        output_file = (
            f"Reward_{pd.Timestamp(final_last_date).strftime('%Y%m%d')}.geojson"
        )
        final_eta_gdf.to_file(output_file, driver="GeoJSON")

        logger.info("Generating final efficiency dataset successfully completed.")

        return output_file, final_eta_gdf

    def find_optimal_solution(self, final_eta_gdf):

        unique_time = pd.Timestamp(final_eta_gdf["time"].iloc[0])

        # Define the number of cells to select
        N = 50  # Maximum number of cells to select

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
                            "time": unique_time,  # Assign the time value
                        }
                    )

            # Convert to GeoDataFrame
            selected_blocks_gdf = gpd.GeoDataFrame(selected_blocks, crs="EPSG:4326")

            # Define the output file name with the last date in YYYYMMDD format
            output_geojson = (
                f"Selected_Cells_Optimization_{unique_time.strftime('%Y%m%d')}.geojson"
            )

            # Save the selected cells to a GeoJSON file
            selected_blocks_gdf.to_file(output_geojson, driver="GeoJSON")

            print(
                f"Selected cells saved to '{output_geojson}' with time: {unique_time}"
            )
        else:
            print("No optimal solution found.")
        return output_geojson, selected_blocks_gdf

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
        logger.info("Loading Missouri Basin shapefile.")
        mo_basin = gpd.read_file(file_path)
        logger.info("Loading Missouri Basin shapefile successfully completed.")
        return gpd.GeoSeries(
            Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326"
        )

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

        # # logger.info('Encoding snow layer.')
        # polygons = self.open_polygons(geojson_path=geojson_path)

        raster_layer = dataset[variable]

        raster_layer = raster_layer.rio.write_crs("EPSG:4326")
        clipped_layer = raster_layer.rio.clip(self.polygons, all_touched=True)
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
            read_only (bool): If True, do not re-download the file if it already exists

        Returns:
            dataset (xarray.Dataset): The dataset
        """
        if not os.path.exists(filename):
            logger.info(f"Downloading file from S3: {filename}")
            config = TransferConfig(use_threads=False)
            s3.download_file(Bucket=bucket, Key=key, Filename=filename, Config=config)
        else:
            logger.info(f"File already exists: {filename}")

        # Open the file
        dataset = xr.open_dataset(filename, engine="h5netcdf")
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

    def upload_file(self, s3, bucket, key, filename):
        """
        Upload a file to an S3 bucket

        Args:
            s3: S3 client
            bucket: S3 bucket name
            key: S3 object key
            filename: Filename to upload
        """
        logger.info(f"Uploading file to S3.")
        config = TransferConfig(use_threads=False)
        s3.upload_file(Filename=filename, Bucket=bucket, Key=key, Config=config)
        logger.info(f"Uploading file to S3 successfully completed.")

    def detect_level_change(self, new_value, old_value, level):
        """
        Detect a change in the level of the time value (day, week, or month).

        Args:
            new_value (datetime): New time value
            old_value (datetime): Old time value
            level (str): Level of time value to detect changes ('day', 'week', or 'month')

        Returns:
            bool: True if the level has changed, False otherwise
        """
        # Check if the level is 'day', 'week', or 'month'. Otherwise, raise an error. If change in 'day', 'week', or 'month' is detected, return True. Otherwise, return False.
        if level == "day":
            return new_value.date() != old_value.date()
        elif level == "week":
            return new_value.isocalendar()[1] != old_value.isocalendar()[1]
        elif level == "month":
            return new_value.month != old_value.month
        else:
            raise ValueError("Invalid level. Choose from 'day', 'week', or 'month'.")

    def multipolygon_to_polygon(self, geometry):
        if geometry.geom_type == "MultiPolygon":
            # Combine all polygons into a single polygon
            return Polygon(
                [
                    coord
                    for polygon in geometry.geoms
                    for coord in polygon.exterior.coords
                ]
            )
        return geometry

    def process_geojson(self, mo_basin):

        mo_basin.at[0, "geometry"] = self.multipolygon_to_polygon(
            mo_basin.at[0, "geometry"]
        )
        return gpd.GeoSeries(
            Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326"
        )

    def on_change2(self, source, property_name, old_value, new_value):
        if property_name == "time":

            # Determine if day has changed
            change = self.detect_level_change(new_value, old_value, "day")

            # Publish message if day, week, or month has changed
            if change:
                old_value_reformat = str(old_value.date()).replace("-", "")
                new_value_reformat = str(new_value.date()).replace("-", "")

                # All Tracks
                all_cells_geojson_path = f"Reward_{new_value_reformat}.geojson"
                all_cells_gdf = gpd.read_file(all_cells_geojson_path)
                all_cells_gdf["time"] = all_cells_gdf["time"].astype(str)
                all_json_data = all_cells_gdf.to_json()
                self.app.send_message(
                    self.app.app_name,
                    "all",
                    VectorLayer(vector_layer=all_json_data).json(),
                )
                logger.info("(ALL) Publishing message successfully completed.")
                time.sleep(15)

                # Selected Cells
                selected_cells_geojson_path = (
                    f"Selected_Cells_Optimization_{new_value_reformat}.geojson"
                )
                selected_cells_gdf = gpd.read_file(selected_cells_geojson_path)

                # Convert the 'time' column to string format
                selected_cells_gdf["time"] = selected_cells_gdf["time"].astype(str)
                selected_json_data = selected_cells_gdf.to_json()
                self.app.send_message(
                    self.app.app_name,
                    "selected_cells",
                    VectorLayer(vector_layer=selected_json_data).json(),
                )
                logger.info("(SELECTED) Publishing message successfully completed.")
                time.sleep(15)

    def on_change(self, source, property_name, old_value, new_value):
        if property_name == "time":

            # Determine if day has changed
            change = self.detect_level_change(new_value, old_value, "day")

            # Publish message if day, week, or month has changed
            if change:
                old_value_reformat = str(old_value.date()).replace("-", "")
                new_value_reformat = str(new_value.date()).replace("-", "")
                logger.info(f">>>OLD VALUE: {old_value_reformat}")
                logger.info(f">>>NEW VALUE: {new_value_reformat}")

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

                mo_basin = self.download_geojson(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key="vector/WBDHU2_4326.geojson",
                    filename="WBDHU2_4326.geojson",
                )
                mo_basin = self.process_geojson(mo_basin)
                self.polygons = mo_basin.geometry

                ###################
                # Combined dataset#
                ###################
                # Get first dataset
                dataset1 = self.download_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=f"Missouri Open Loop sample output/LIS_HIST_{old_value_reformat}0000.d01.nc",
                    filename=f"LIS_HIST_{old_value_reformat}0000.d01.nc",
                )
                # Get second dataset
                dataset2 = self.download_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=f"Missouri Open Loop sample output/LIS_HIST_{new_value_reformat}0000.d01.nc",
                    filename=f"LIS_HIST_{new_value_reformat}0000.d01.nc",
                )

                # Generate the combined dataset
                combined_output_file, combined_dataset = self.generate_combined_dataset(
                    dataset1, dataset2, mo_basin
                )

                # Upload dataset to S3
                self.upload_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=combined_output_file,
                    filename=combined_output_file,
                )

                # Select the SWE_tavg variable for a specific time step (e.g., first time step)
                swe_data = combined_dataset["SWE_tavg"].isel(time=0)  # SEND AS MESSAGE

                swe_layer_encoded, top_left, top_right, bottom_left, bottom_right = (
                    self.encode(
                        dataset=combined_dataset,
                        variable="SWE_tavg",
                        output_path=f"swe_data_{new_value_reformat}.png",
                        time_step=0,
                        scale="time",
                        geojson_path="WBD_10_HU2_4326.geojson",
                        rotate=True,
                    )
                )
                # del combined_dataset

                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=swe_layer_encoded,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                    ).json(),
                )
                logger.info("Publishing message successfully completed.")
                time.sleep(15)

                ###############
                # ETA5 dataset#
                ###############
                # Generate the SWE difference
                swe_output_file, eta5_file = self.generate_swe_difference(
                    ds=combined_dataset
                )

                # Upload dataset to S3
                self.upload_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=swe_output_file,
                    filename=swe_output_file,
                )

                # Select the eta5 variable for a specific time step (e.g., first time step)
                eta5_data = eta5_file["eta5"].isel(time=1)

                eta5_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta5_file,
                    variable="eta5",
                    output_path=f"eta5_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                # del eta5_file

                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=eta5_layer_encoded,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                    ).json(),
                )
                logger.info("Publishing message successfully completed.")
                time.sleep(15)

                ###############
                # ETA0 dataset#
                ###############
                # Generate the surface temperature dataset
                surfacetemp_output_file, eta0_file = self.generate_surface_temp(
                    ds=combined_dataset
                )

                # Upload dataset to S3
                self.upload_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=surfacetemp_output_file,
                    filename=surfacetemp_output_file,
                )

                # Select the eta0 variable for a specific time step (e.g., first time step)
                eta0_data = eta0_file["eta0"].isel(time=1)

                eta0_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta0_file,
                    variable="eta0",
                    output_path=f"eta0_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                # del eta0_file

                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=eta0_layer_encoded,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                    ).json(),
                )
                logger.info("Publishing message successfully completed.")
                time.sleep(15)

                ###################
                # ETA2 GCOM dataset#
                ###################
                # Generate the sensor GCOM dataset
                sensor_gcom_output_file, eta2_file_GCOM = self.generate_sensor_gcom(
                    ds=combined_dataset
                )

                # Upload dataset to S3
                self.upload_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=sensor_gcom_output_file,
                    filename=sensor_gcom_output_file,
                )

                # Select the eta2 variable for a specific time step (e.g., first time step)
                eta2_data_GCOM = eta2_file_GCOM["eta2"].isel(time=1)

                eta2_gcom_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta2_file_GCOM,
                    variable="eta2",
                    output_path=f"eta2_gcom_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                # del eta2_file_GCOM

                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=eta2_gcom_layer_encoded,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                    ).json(),
                )
                logger.info("Publishing message successfully completed.")
                time.sleep(15)

                ######################
                # ETA2 Capella dataset#
                ######################
                # Generate the sensor capella dataset
                sensor_capella_output_file, eta2_file_Capella = (
                    self.generate_sensor_capella(ds=combined_dataset)
                )

                # Upload dataset to S3
                self.upload_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=sensor_capella_output_file,
                    filename=sensor_capella_output_file,
                )

                # Select the eta2 variable for a specific time step (e.g., first time step)
                eta2_data_Capella = eta2_file_Capella["eta2"].isel(time=1)

                eta2_capella_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta2_file_Capella,
                    variable="eta2",
                    output_path=f"eta2_capella_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                # del eta2_file_Capella

                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=eta2_capella_layer_encoded,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                    ).json(),
                )
                logger.info("Publishing message successfully completed.")
                time.sleep(15)

                ###########
                # GCOM ETA#
                ###########
                # Define the weights for each dataset
                weights = {"eta5": 0.5, "eta0": 0.3, "eta2": 0.2}

                # Process GCOM datasets
                gcom_combine_multiply_output_file, gcom_dataset = (
                    self.combine_and_multiply_datasets(
                        ds=combined_dataset,
                        eta5_file=eta5_file,
                        eta0_file=eta0_file,
                        eta2_file=eta2_file_GCOM,
                        weights=weights,
                        output_file="Combined_Efficiency_Weighted_Product_GCOM",
                    )
                )

                # Upload dataset to S3
                self.upload_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=gcom_combine_multiply_output_file,
                    filename=gcom_combine_multiply_output_file,
                )

                # Select the combined_eta variable for a specific time step (e.g., first time step)
                gcom_eta = gcom_dataset["combined_eta"].isel(time=1)

                gcom_eta_layer_encoded, _, _, _, _ = self.encode(
                    dataset=gcom_dataset,
                    variable="combined_eta",
                    output_path=f"gcom_eta_combined_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                # del gcom_dataset

                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=gcom_eta_layer_encoded,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                    ).json(),
                )
                logger.info("Publishing message successfully completed.")
                time.sleep(15)

                #############
                # Capella ETA#
                #############
                # Process Capella datasets
                capella_combine_multiply_output_file, capella_dataset = (
                    self.combine_and_multiply_datasets(
                        ds=combined_dataset,
                        eta5_file=eta5_file,
                        eta0_file=eta0_file,
                        eta2_file=eta2_file_Capella,
                        weights=weights,
                        output_file="Combined_Efficiency_Weighted_Product_Capella",
                    )
                )

                # Upload dataset to S3
                self.upload_file(
                    s3=s3,
                    bucket="snow-observing-systems",
                    key=capella_combine_multiply_output_file,
                    filename=capella_combine_multiply_output_file,
                )

                # # Select the combined_eta variable for a specific time step (e.g., first time step)
                # capella_eta = capella_dataset["combined_eta"].isel(time=1)

                capella_eta_layer_encoded, _, _, _, _ = self.encode(
                    dataset=capella_dataset,
                    variable="combined_eta",
                    output_path=f"capella_eta_combined_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                # del capella_dataset

                self.app.send_message(
                    self.app.app_name,
                    "layer",
                    SWEChangeLayer(
                        swe_change_layer=capella_eta_layer_encoded,
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                    ).json(),
                )
                logger.info("Publishing message successfully completed.")
                time.sleep(15)

                ###########
                # Final ETA#
                ###########
                final_eta_output_file, final_eta_gdf = self.process(
                    gcom_ds=gcom_dataset,
                    snowglobe_ds=capella_dataset,
                    mo_basin=mo_basin,
                    start=old_value,  # 03-01
                    end=new_value,  # 03-02
                )

                # Clip Final Eta GDF and ground tracks to the Missouri Basin
                final_eta_gdf_clipped = gpd.clip(final_eta_gdf, mo_basin)

                # Convert the clipped GeoDataFrame to GeoJSON and send as message
                all_json_data = final_eta_gdf_clipped.drop(
                    "time", axis=1, errors="ignore"
                ).to_json()
                self.app.send_message(
                    self.app.app_name,
                    "all",
                    VectorLayer(vector_layer=all_json_data).json(),
                )
                logger.info("(ALL) Publishing message successfully completed.")
                time.sleep(15)

                self.app.send_message(
                    self.app.app_name,
                    "available",
                    VectorLayer(vector_layer=all_json_data).json(),
                )

                #######################
                # Find Optimal Solution#
                #######################
                output_geojson, selected_cells_gdf = self.find_optimal_solution(
                    final_eta_gdf=final_eta_gdf
                )

                # # Clip Final Eta GDF and ground tracks to the Missouri Basin
                # selected_cells_gdf_clipped = gpd.clip(selected_cells_gdf, mo_basin)

                # Convert the clipped GeoDataFrame to GeoJSON and send as message
                selected_json_data = selected_cells_gdf.drop(
                    "time", axis=1, errors="ignore"
                ).to_json()
                self.app.send_message(
                    self.app.app_name,
                    "selected",
                    VectorLayer(vector_layer=selected_json_data).json(),
                )
                logger.info("(SELECTED) Publishing message successfully completed.")
                time.sleep(15)

                # # # Ensure CRS matches for consistency
                # # selected_cells_gdf = selected_cells_gdf.to_crs(mo_basin.crs)
                # # final_eta_gdf = final_eta_gdf.to_crs(mo_basin.crs)
                # s3.close()


def main():
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")

    # Define the simulation parameters
    NAME = "planner"

    # create the managed application
    app = ManagedApplication(NAME)

    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app))

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix, config
    )


if __name__ == "__main__":
    main()
