# FROM: https://github.com/code-lab-org/sos/blob/H-Update/src/SNODAS/SNODAS.py
from datetime import datetime
import gzip
import os
import requests
import shutil
import tarfile
import tempfile
from osgeo import gdal 
import pandas as pd
from dotenv import dotenv_values, load_dotenv
import rioxarray as rxr
import xarray as xr
from typing import List, Tuple
from nost_tools.simulator import Simulator, Mode
from constellation_config_files.schemas import SNODASStatus

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




# def main():
#     dates = pd.date_range(datetime(2024, 1, 7), datetime(2024, 1, 21))

#     # Get the minimum and maximum dates
#     min_date = dates.min()
#     max_date = dates.max()
    
#     print(min_date, max_date)
#     snodas_dir = "SNODAS"
#     ensure_directory_exists(snodas_dir)
#     process_dates(dates, snodas_dir)
#     output_path = os.path.join('./data', "snodas-merged.nc")
#     print("Writing snodas-merged.nc")
#     merge_netcdf_files(dates, snodas_dir, output_path)

# if __name__ == "__main__":
#     main()

# define an observer to manage ground updates
class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*

    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """

    def __init__(self, app): #, grounds):
        self.app = app
        # self.grounds = grounds

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
        if not os.path.exists(directory):
            print(f'Creating {directory} directory')
            os.makedirs(directory)

    def download_and_extract_tar(self, date, tmp_dir, dir_label):
        url = (
            "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/" + 
            f"{date.strftime('%Y')}/{date.strftime('%m')}_{date.strftime('%b')}/" +
            dir_label + ".tar"
        )
        r = requests.get(url)
        tar_path = os.path.join(tmp_dir, dir_label + ".tar")
        with open(tar_path, "wb") as tar_file:
            tar_file.write(r.content)
        with tarfile.open(tar_path, "r") as tar_file:
            tar_file.extractall(tmp_dir)

    def process_swe_file(self, tmp_dir, snodas_dir, file_label, date):
        gz_path = os.path.join(tmp_dir, file_label + ".dat.gz")
        dat_path = os.path.join(tmp_dir, file_label + ".dat")
        with gzip.open(gz_path, "rb") as gz_in:
            with open(dat_path, "wb") as gz_out:
                shutil.copyfileobj(gz_in, gz_out)
        hdr_path = os.path.join(tmp_dir, file_label + ".hdr")
        with open(hdr_path, "w") as hdr_file:
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
        command = " ".join([
            "gdal_translate",
            "-of NetCDF",
            "-a_srs EPSG:4326",
            "-a_nodata -9999",
            "-a_ullr -124.73375000000000 52.87458333333333 -66.94208333333333 24.94958333333333" 
            if date < datetime(2013, 10, 1)
            else "-a_ullr -124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000",
            dat_path,
            os.path.join(snodas_dir, file_label + ".nc")
        ])
        if os.system(command) > 0:
            print(f"Error processing command `{command}`")

    def process_dates(self, dates, snodas_dir):
        for date in dates:
            file_label = f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001"
            print(file_label)
            if os.path.isfile(os.path.join(snodas_dir, file_label + ".nc")):
                print("Skipping " + file_label)
                continue
            print("Processing " + file_label)
            dir_label = f"SNODAS_{date.strftime('%Y%m%d')}"
            with tempfile.TemporaryDirectory() as tmp_dir:
                self.download_and_extract_tar(date, tmp_dir, dir_label)
                for filename in os.listdir(tmp_dir):
                    if os.path.isfile(os.path.join(tmp_dir, filename)) and filename == file_label + ".dat.gz":
                        self.process_swe_file(tmp_dir, snodas_dir, file_label, date)

    def merge_netcdf_files(self, dates, snodas_dir, output_path):
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
        ).to_netcdf(output_path)

    def on_change(self, source, property_name, old_value, new_value):
        """
        *Standard on_change callback function format inherited from Observer object class*

        In this instance, the callback function checks when the **PROPERTY_MODE** switches to **EXECUTING** to send a :obj:`GroundLocation` message to the *PREFIX/ground/location* topic:

            .. literalinclude:: /../../NWISdemo/grounds/main_ground.py
                :lines: 56-67

        """
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            logger.info(f"Environment observer received EXECUTING mode from {source}")

            start_time = self.app._sim_start_time.date()
            stop_time = self.app._sim_stop_time.date()
            logger.info(f"Start time: {start_time}")
            logger.info(f"Stop time: {stop_time}")

            dates = pd.date_range(start_time, stop_time)
            logger.info(f"Dates: {dates}")

            # dates = pd.date_range(datetime(2024, 1, 7), datetime(2024, 1, 21))
            # logger.info(f"OG Dates: {dates}")

            # Download SNODAS data
            # snodas_dir = "SNODAS"
            path_hdf, path_nc, path_shp, path_preprocessed, path_efficiency, path_snodas = self.load_config()
            self.ensure_directory_exists(path_snodas)
            
            # Process SNODAS data
            logger.info("Processing SNODAS data.")
            self.process_dates(dates, path_snodas)
            logger.info("Processing SNODAS data successfully completed.")

            # Merge SNODAS data
            logger.info("Merging SNODAS data.")
            output_path = os.path.join(path_snodas, "snodas-merged.nc")
            self.merge_netcdf_files(dates, path_snodas, output_path)
            logger.info("Merging SNODAS data successfully completed.")

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

            logger.info(f"Environment observer sent SNODASStatus message to {self.app.app_name}/data")
   
def main():
    # Load credentials from a .env file in current working directory
    credentials = dotenv_values(".env")
    HOST, RABBITMQ_PORT, KEYCLOAK_PORT, KEYCLOAK_REALM = credentials["HOST"], int(credentials["RABBITMQ_PORT"]), int(credentials["KEYCLOAK_PORT"]), str(credentials["KEYCLOAK_REALM"])
    USERNAME, PASSWORD = credentials["USERNAME"], credentials["PASSWORD"]
    CLIENT_ID = credentials["CLIENT_ID"]
    CLIENT_SECRET_KEY = credentials["CLIENT_SECRET_KEY"]
    VIRTUAL_HOST = credentials["VIRTUAL_HOST"]
    IS_TLS = credentials["IS_TLS"].lower() == 'true'  # Convert to boolean
    PREFIX = "sos"
    NAME = "snodas"
    SCALE = 1


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
    app = ManagedApplication(NAME) #, config=config)

    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app)) #, GROUND))

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