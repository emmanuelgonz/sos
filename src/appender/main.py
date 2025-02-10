import json
import logging

import geopandas as gpd
import pandas as pd
from constellation_config_files.schemas import VectorLayer
from nost_tools.application_utils import ShutDownObserver
from nost_tools.config import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import Observer
from nost_tools.simulator import Mode, Simulator

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*

    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """

    def __init__(self, app):  # , grounds):
        self.app = app
        self.counter = 0
        self.master_components = []
        self.master_gdf = gpd.GeoDataFrame()

    def add_prefix_to_columns(self, gdf, prefix):
        """
        Adds a prefix to each column name in the GeoDataFrame, except for the 'geometry' column.

        Inputs:
            gdf (GeoDataFrame): The GeoDataFrame whose columns will be prefixed.
            prefix (str): The prefix to add to each column name.

        Returns:
            GeoDataFrame: The GeoDataFrame with each column name prefixed
        """
        gdf.columns = [
            prefix + col if col != "geometry" else col for col in gdf.columns
        ]
        return gdf

    def add_columns(self, gdf):
        """
        Adds columns to the GeoDataFrame that will be filled by the simulator.

        Inputs:
            gdf (GeoDataFrame): The GeoDataFrame to which the columns will be added.

        Returns:
            GeoDataFrame: The GeoDataFrame with the additional columns added.
        """
        gdf["simulator_simulation_status"] = None
        gdf["simulator_completion_date"] = pd.NaT  # Setting it as a datetime type
        gdf["simulator_expiration_date"] = pd.NaT
        gdf["simulator_satellite"] = None
        gdf["simulator_polygon_groundtrack"] = None
        gdf["planner_latitude"] = gdf["planner_centroid"].y
        gdf["planner_longitude"] = gdf["planner_centroid"].x
        # gdf['planner_geometry'] = gdf['planner_geometry'].to_wkt()
        gdf["planner_centroid"] = gdf["planner_centroid"].to_wkt()

        return gdf

    def reorder_columns(self, gdf):
        """
        Reorders the columns of the GeoDataFrame to a specific order.

        Inputs:
            gdf (GeoDataFrame): The GeoDataFrame whose columns will be reordered.

        Returns:
            GeoDataFrame: The GeoDataFrame with the columns reordered.
        """
        gdf = gdf[
            [
                "simulator_id",
                "planner_time",
                "planner_final_eta",  #'planner_geometry',
                "planner_latitude",
                "planner_longitude",
                "simulator_expiration_date",
                "simulator_simulation_status",
                "simulator_completion_date",
                "simulator_satellite",
                "geometry",
            ]
        ]
        return gdf

    def process_component(self, component_gdf):
        """
        Inputs:
            component_gdf (GeoDataFrame): The GeoDataFrame of the component to process.
            counter (int): The counter to use for assigning unique IDs to the component.

        Returns:
            GeoDataFrame: The processed GeoDataFrame of the component.
        """
        logger.info("Processing component GeoJSON.")
        # Ensure the geometry is in a projected CRS before calculating the centroid
        if component_gdf.crs.is_geographic:
            component_gdf = component_gdf.to_crs(
                epsg=3395
            )  # Use EPSG:3395 for projection

        # Add the centroid of the geometry
        component_gdf["centroid"] = component_gdf.centroid

        # Add prefix for organizational purposes
        component_gdf = self.add_prefix_to_columns(component_gdf, "planner_")

        # # Set geometry
        # component_gdf = component_gdf.set_geometry("planner_geometry")

        # Add columns that will be filled by the simulator
        component_gdf = self.add_columns(component_gdf)

        # Add unique ID to each component GDF using counter
        component_gdf["simulator_id"] = range(
            self.counter, self.counter + len(component_gdf)
        )

        # Reorder columns
        component_gdf = self.reorder_columns(component_gdf)

        # Reproject the GeoDataFrame to EPSG:4326
        component_gdf = component_gdf.to_crs(epsg=4326)

        logger.info("Processing component GeoJSON successfully completed.")

        return component_gdf

    def remove_duplicates(self):

        # Separate rows with 'simulator_simulation_status' as 'Completed'
        completed_rows = self.master_gdf[
            self.master_gdf["simulator_simulation_status"] == "Completed"
        ]

        # Separate rows with 'simulator_simulation_status' as None
        none_rows = self.master_gdf[
            self.master_gdf["simulator_simulation_status"].isna()
        ]

        # Find the most recent row for each geometry where 'simulator_simulation_status' is None
        most_recent_none_rows = none_rows.loc[
            none_rows.groupby("geometry")["planner_time"].idxmax()
        ]

        # Combine the completed rows with the most recent none rows
        self.master_gdf = pd.concat(
            [completed_rows, most_recent_none_rows], ignore_index=True
        )

    # def on_simulator(self, ch, method, properties, body):
    #     body.new_data > "true" master
    #     print(body)

    def on_planner(self, ch, method, properties, body):

        body = body.decode("utf-8")
        data = VectorLayer.parse_raw(body)

        # Convert the string back to a GeoJSON object
        geojson_obj = json.loads(data.vector_layer)

        # Create a GeoDataFrame from the GeoJSON object
        component_gdf = gpd.GeoDataFrame.from_features(
            geojson_obj["features"], crs="EPSG:4326"
        )
        # component_gdf["time"] = pd.to_datetime(component_gdf["time"])

        # Process component
        component_gdf = self.process_component(component_gdf)

        # Append processed component to master component list
        self.master_components.append(component_gdf)

        # Update counter for the next component GDF
        self.counter += len(component_gdf)

        # Assuming component_gdf is your GeoDataFrame
        min_value = component_gdf["simulator_id"].min()
        max_value = component_gdf["simulator_id"].max()

        logger.info(f"Range of values in 'simulator_id': {min_value} to {max_value}")

        # Concatenate all components into a single GeoDataFrame
        self.master_gdf = pd.concat(self.master_components, ignore_index=True)
        self.remove_duplicates()

        # Convert the clipped GeoDataFrame to GeoJSON and send as message
        selected_json_data = self.master_gdf[
            ["planner_final_eta", "planner_time", "geometry"]
        ].to_json()
        self.app.send_message(
            "swe_change",
            "selected",
            VectorLayer(vector_layer=selected_json_data).json(),
        )
        logger.info("(SELECTED) Publishing message successfully completed.")
        self.master_gdf.to_file("final_master.geojson", driver="GeoJSON")

    def on_change(self, source, property_name, old_value, new_value):
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            logger.info(
                "Switched to EXECUTING mode. Counter and master components list initialized."
            )
            # self.first_run = True
        # if property_name == Simulator.PROPERTY_MODE and new_value == Mode.TERMINATING:
        #     logger.info(
        #         "Switched to TERMINATING mode. Shutting down the application and saving data."
        #     )
        #     # master_gdf = pd.concat(self.master_components, ignore_index=True)
        #     self.master_gdf.to_file("final_master.geojson", driver="GeoJSON")
        #     # self.app.shut_down()


def main():
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")

    # Define the simulation parameters
    NAME = "appender"

    # create the managed application
    app = ManagedApplication(NAME)

    # initialize the Environment object class
    environment = Environment(app)

    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app))

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix, config
    )

    app.add_message_callback("swe_change", "selected_cells", environment.on_planner)


if __name__ == "__main__":
    main()
