from datetime import datetime
import gzip
import os
import requests
import shutil
import tarfile
import tempfile
from osgeo import gdal 
import pandas as pd
from dotenv import dotenv_values
import rioxarray as rxr
import xarray as xr
from datetime import datetime
import pandas as pd
import os
from nost_tools.simulator import Simulator, Mode
from constellation_config_files.schemas import SNODASStatus

from nost_tools.application_utils import ConnectionConfig, ShutDownObserver
from nost_tools.entity import Entity
from nost_tools.observer import Observer
from nost_tools.managed_application import ManagedApplication
from nost_tools.publisher import WallclockTimeIntervalPublisher

# # define the date range over which to prepare data
# dates = pd.date_range(datetime(2024,1,20), datetime(2024,1,21))

# # define the local directory in which to work
# snodas_dir = "SNODAS"
# # ensure the directory exists
# if not os.path.exists(snodas_dir):
#     print('Creating SNODAS directory')
#     os.makedirs(snodas_dir)

# # iterate over each date
# for date in dates:
#     # prepare the SWE file label for this date
#     file_label = f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001"
#     print(file_label)
#     # check if file already exists
#     if os.path.isfile(os.path.join(snodas_dir, file_label + ".nc")):
#         print("Skipping " + file_label)
#         continue
#     print("Processing " + file_label)
#     # prepare the SNODAS directory label for this date
#     dir_label = f"SNODAS_{date.strftime('%Y%m%d')}"
#     # request the .tar file from NSIDC
#     r = requests.get(
#         "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/" + 
#         f"{date.strftime('%Y')}/{date.strftime('%m')}_{date.strftime('%b')}/" +
#         dir_label + ".tar"
#     )
#     # create a temporary directory in which to do work
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         # save the .tar file
#         with open(os.path.join(tmp_dir, dir_label + ".tar"), "wb") as tar_file:
#             tar_file.write(r.content)
#         # open and extract the .tar file
#         with tarfile.open(os.path.join(tmp_dir, dir_label + ".tar"), "r") as tar_file:
#             tar_file.extractall(tmp_dir)
#         # iterate through all extracted files
#         for filename in os.listdir(tmp_dir):
#             # check if the file matches the SWE file label
#             if os.path.isfile(os.path.join(tmp_dir, filename)) and filename == file_label + ".dat.gz":
#                 # unzip the SWE .gz file
#                 with gzip.open(os.path.join(tmp_dir, file_label + ".dat.gz"), "rb") as gz_in:
#                     with open(os.path.join(tmp_dir, file_label + ".dat"), "wb") as gz_out:
#                         shutil.copyfileobj(gz_in, gz_out)
#                 # write the SWE .hdr file
#                 with open(os.path.join(tmp_dir, file_label + ".hdr"), "w") as hdr_file:
#                     hdr_file.write(
#                         "ENVI\n"
#                         "samples = 6935\n" +
#                         "lines = 3351\n" +
#                         "bands = 1\n" +
#                         "header offset = 0\n" + 
#                         "file type = ENVI Standard\n" + 
#                         "data type = 2\n" +
#                         "interleave = bsq\n" +
#                         "byte order = 1"
#                     )
#                 # run the gdal translator using date-specific bounding box
#                 command = " ".join([
#                     "gdal_translate",
#                     "-of NetCDF",
#                     "-a_srs EPSG:4326",
#                     "-a_nodata -9999",
#                     "-a_ullr -124.73375000000000 52.87458333333333 -66.94208333333333 24.94958333333333" 
#                     if date < datetime(2013, 10, 1)
#                     else "-a_ullr -124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000",
#                     os.path.join(tmp_dir, file_label + ".dat"),
#                     os.path.join(snodas_dir, file_label + ".nc")
#                 ])
#                 if os.system(command) > 0: 
#                     print(f"Error processing command `{command}`")



# dates = pd.date_range(datetime(2024,1,20), datetime(2024,1,21))
# print("Writing snodas-merged.nc")
# ds = xr.combine_by_coords(
#     [
#         rxr.open_rasterio(
#             os.path.join(
#                 snodas_dir, 
#                 f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001.nc"
#             )
#         ).drop_vars("band").assign_coords(time=date).expand_dims(dim="time")
#         for date in dates
#     ], 
#     combine_attrs="drop_conflicts"
# ).to_netcdf(os.path.join('./data', "snodas-merged.nc"))

# define an observer to manage ground updates
class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*

    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """

    def __init__(self, app, grounds):
        self.app = app
        self.grounds = grounds

    def on_change(self, source, property_name, old_value, new_value):
        """
        *Standard on_change callback function format inherited from Observer object class*

        In this instance, the callback function checks when the **PROPERTY_MODE** switches to **EXECUTING** to send a :obj:`GroundLocation` message to the *PREFIX/ground/location* topic:

            .. literalinclude:: /../../NWISdemo/grounds/main_ground.py
                :lines: 56-67

        """
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            for index, ground in self.grounds.iterrows():
                self.app.send_message(
                    self.app.app_name,
                    "location",
                    SNODASStatus(
                        file_path=ground.groundId,
                        time=ground.latitude
                    ).json(),
                )

# # define an observer to send fire detection events
# class DataGeneratedObserver(Observer):
#     """
#     *This object class inherits properties from the Observer object class from the observer template in the NOS-T tools library*

#     Args:
#         app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
#     """
#     def __init__(self, app):
#         self.app = app

#     def on_change(self, source, property_name, old_value, new_value):
#         """
#         *Standard on_change callback function format inherited from Observer object class in NOS-T tools library*

#         In this instance, the callback function checks for notification of the "detected" property and publishes :obj:`FireDetected` message to *PREFIX/constellation/detected* topic:

#         """
#         if property_name == Constellation.PROPERTY_FIRE_DETECTED:
#             self.app.send_message(
#                 self.app.app_name,
#                 "detected",
#                 FireDetected(
#                     fireId=new_value["fireId"],
#                     detected=new_value["detected"],
#                     detected_by=new_value["detected_by"],
#                 ).json(),
#             )

def main():

    credentials = dotenv_values(".env")
    HOST, RABBITMQ_PORT, KEYCLOAK_PORT, KEYCLOAK_REALM = credentials["HOST"], int(credentials["RABBITMQ_PORT"]), int(credentials["KEYCLOAK_PORT"]), str(credentials["KEYCLOAK_REALM"])
    USERNAME, PASSWORD = credentials["USERNAME"], credentials["PASSWORD"]
    CLIENT_ID = credentials["CLIENT_ID"]
    CLIENT_SECRET_KEY = credentials["CLIENT_SECRET_KEY"]
    VIRTUAL_HOST = credentials["VIRTUAL_HOST"]
    IS_TLS = credentials["IS_TLS"].lower() == 'true'
    PREFIX = "sos"
    NAME = "snodas"

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
    app.simulator.add_observer(Environment(app, GROUND))

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        PREFIX,
        config,
        True,
        time_status_step=timedelta(seconds=10) * SCALE,
        time_status_init=datetime(2020, 1, 1, 7, 20, tzinfo=timezone.utc),
        time_step=timedelta(seconds=1) * SCALE,
        # shut_down_when_terminated=True,
    )
        
if __name__ == "__main__":
    main()