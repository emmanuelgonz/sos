# FROM: https://github.com/code-lab-org/sos/blob/H-Update/src/SNODAS/SNODAS.py
from datetime import datetime
import gzip
import os
import requests
import shutil
import tarfile
import tempfile
# from osgeo import gdal 
import pandas as pd
from dotenv import dotenv_values, load_dotenv
import rioxarray as rxr
import earthaccess
import xarray as xr
import numpy as np
import glob
from typing import List, Tuple
from nost_tools.simulator import Simulator, Mode
from constellation_config_files.schemas import SNODASStatus
from constellation_config_files.config import PREFIX, NAME, SCALE

from datetime import timedelta, datetime, timezone
import logging
from dotenv import dotenv_values

from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.observer import Observer
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher

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

    def __init__(self, app): #, grounds):
        self.app = app
        self.path_hdf, self.path_nc, self.path_shp, self.path_preprocessed, self.path_efficiency, self.path_snodas = self.load_config()

    def load_config(self) -> Tuple[str, ...]:
        """Load environment variables and return paths"""
        load_dotenv()
        return (
            os.getenv('path_hdf'),
            os.getenv('path_nc'), 
            os.getenv('path_shp'),
            os.getenv('path_preprocessed'),
            os.getenv('path_efficiency'),
            os.getenv('path_snodas')
        )
    
    def ensure_directory_exists(self, directory):
        """
        Ensure that the specified directory exists. If it does not, create it.

        Args:
            directory (str): The directory to check for existence. If it does not exist, create it.
        """
        if not os.path.exists(directory):
            logger.info(f'Creating {directory} directory')
            os.makedirs(directory)
    
    def download_snow_data(self, path_hdf: str, start_date: str, end_date: str) -> List[str]:
        """Download snow cover data using EarthAccess
        
        Args:
            path_hdf (str): The path to save the downloaded HDF files
            start_date (str): The start date to download data
            end_date (str): The end date to download data
        
        Returns:
            List[str]: A list of downloaded files
        """
        logger.info("Downloading snow data.")
        earthaccess.login(strategy="environment")
        results = earthaccess.search_data(
            short_name='MOD10C1',
            temporal=(start_date, end_date)
        )
        logger.info("Downloading snow data successfully completed.")
        return earthaccess.download(results, path_hdf, threads=1)
    
    def process_snow_files(self, path_hdf: str, path_nc: str) -> Tuple[List[str], List[datetime]]:
        """
        Process snow files to NetCDF format.

        Args:
            path_hdf (str): The path to HDF files
            path_nc (str): The path to save NetCDF files
        
        Returns:
            Tuple[List[str], List[datetime]]: A tuple of list of files and list of datetime
        """
        logger.info("Processing snow files.")
        ctr = 0
        lon = np.linspace(-180,180,7200)
        lat = np.flip(np.linspace(-90,90,3600))
        files = []
        time_sc = []

        for filename in os.listdir(path_hdf):    
            year = filename[9:13]
            day = filename[13:16]
            name = filename[0:34]

            dates = pd.to_datetime(int(day)-1,unit = 'D', origin=year)     
            time_sc.append(dates)
            f_nc = xr.open_dataset(os.path.join(path_hdf, filename),engine = 'netcdf4')  
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
        files = [f for f in files if not f.endswith("snowcover-merged.nc")]
        
        return files, time_sc
    
    def merge_netcdf_files(self, files: List[str], time_sc: List[datetime], output_path: str) -> xr.Dataset:
        """
        Merge NetCDF files.

        Args:
            files (List[str]): A list of files
            time_sc (List[datetime]): A list of datetime
            output_path (str): The path to save the merged NetCDF file
        """
        logger.info("Merging NetCDF files.")
        ds = xr.combine_by_coords(
            [        
                rxr.open_rasterio(files[i]).drop_vars("band").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
                for i in range(len(time_sc))            
            ], 
            combine_attrs="drop_conflicts"
        )
        ds = ds.rio.write_crs("EPSG:4326")
        ds.to_netcdf(output_path)
        logger.info(f"Merging NetCDF files successfully completed.")


    def on_change(self, source, property_name, old_value, new_value):
        """
        *Standard on_change callback function format inherited from Observer object class*

        In this instance, the callback function checks when the **PROPERTY_MODE** switches to **EXECUTING** to send a :obj:`GroundLocation` message to the *PREFIX/ground/location* topic:

            .. literalinclude:: /../../NWISdemo/grounds/main_ground.py
                :lines: 56-67

        """
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            logger.info(f"Environment observer received EXECUTING mode from {source}")

            # Extract start and stop times
            start_time = self.app._sim_start_time.date()
            stop_time = self.app._sim_stop_time.date()
            dates = pd.date_range(start_time, stop_time)
            logger.info(f"Processing MOD10C1 data from {start_time} to {stop_time}")

            # Ensure download directory exists & download MOD10C1 data
            self.ensure_directory_exists(self.path_hdf)
            self.download_snow_data(self.path_hdf, start_time, stop_time)
            
            # Process MOD10C1 data
            logger.info("Processing MOD10C1 data.")
            files, time_sc = self.process_snow_files(self.path_hdf, self.path_nc)
            logger.info("Processing MOD10C1 data successfully completed.")

            # Merge MOD10C1 data
            logger.info("Merging MOD10C1 data.")
            output_path = os.path.join(self.path_nc, "snowcover-merged.nc")
            self.merge_netcdf_files(files, time_sc, output_path)
            logger.info("Merging MOD10C1 data successfully completed.")

            # Send DataStatus message
            self.app.send_message(
                self.app.app_name,
                "data",
                SNODASStatus(
                    file_path=output_path,
                    publish_time=datetime.now(timezone.utc),
                    start_date=start_time,
                    end_date=stop_time,
                ).json(),
            )

            logger.info(f"SNODASStatus message sent.")

def main():
    # Load credentials from a .env file in current working directory
    credentials = dotenv_values(".env")
    HOST, RABBITMQ_PORT, KEYCLOAK_PORT, KEYCLOAK_REALM = credentials["HOST"], int(credentials["RABBITMQ_PORT"]), int(credentials["KEYCLOAK_PORT"]), str(credentials["KEYCLOAK_REALM"])
    USERNAME, PASSWORD = credentials["USERNAME"], credentials["PASSWORD"]
    CLIENT_ID = credentials["CLIENT_ID"]
    CLIENT_SECRET_KEY = credentials["CLIENT_SECRET_KEY"]
    VIRTUAL_HOST = credentials["VIRTUAL_HOST"]
    IS_TLS = credentials["IS_TLS"].lower() == 'true'  # Convert to boolean

    # Set the client credentials from the config file
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

    # create the managed application
    app = ManagedApplication(NAME)

    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app))

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        PREFIX,
        config,
        True,
        time_status_step=timedelta(seconds=10) * SCALE,
        time_status_init=datetime(2024, 1, 7, tzinfo=timezone.utc),
        time_step=timedelta(seconds=1) * SCALE,
        # shut_down_when_terminated=True,
    )
    
    # while True:
    #     pass

if __name__ == "__main__":
    main()