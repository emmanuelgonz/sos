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

    def __init__(self, app):
        self.app = app
        self.counter = 0
        self.master_components = []
        self.master_gdf = gpd.GeoDataFrame()
        self.visualize_selected = False

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
        gdf["simulator_completion_date"] = pd.NaT
        gdf["simulator_expiration_date"] = pd.NaT
        gdf["simulator_satellite"] = None
        gdf["simulator_polygon_groundtrack"] = None
        gdf["planner_latitude"] = gdf["planner_centroid"].y
        gdf["planner_longitude"] = gdf["planner_centroid"].x
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
                "planner_final_eta",
                "planner_latitude",
                "planner_longitude",
                "simulator_expiration_date",
                "simulator_simulation_status",
                "simulator_completion_date",
                "simulator_satellite",
                "simulator_polygon_groundtrack",
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
        component_gdf["centroid"] = component_gdf.centroid
        component_gdf = self.add_prefix_to_columns(component_gdf, "planner_")
        component_gdf = self.add_columns(component_gdf)
        component_gdf["simulator_id"] = range(
            self.counter, self.counter + len(component_gdf)
        )
        component_gdf = self.reorder_columns(component_gdf)
        component_gdf = component_gdf.to_crs(epsg=4326)
        logger.info("Processing component GeoJSON successfully completed.")
        return component_gdf

    def remove_duplicates(self):
        """
        Removes duplicate rows from the master GeoDataFrame.

        Returns:
            GeoDataFrame: The master GeoDataFrame with duplicates
        """
        completed_rows = self.master_gdf[
            self.master_gdf["simulator_simulation_status"] == "Completed"
        ]
        none_rows = self.master_gdf[
            self.master_gdf["simulator_simulation_status"].isna()
        ]
        most_recent_none_rows = none_rows.loc[
            none_rows.groupby("geometry")["planner_time"].idxmax()
        ]
        self.master_gdf = pd.concat(
            [completed_rows, most_recent_none_rows], ignore_index=True
        )

    def message_to_geojson(self, body):
        """
        Converts a message body to a GeoDataFrame.

        Inputs:
            body (bytes): The message body to convert.

        Returns:
            GeoDataFrame: The GeoDataFrame created from the message
        """
        body = body.decode("utf-8")
        data = VectorLayer.model_validate_json(body)
        return gpd.GeoDataFrame.from_features(
            json.loads(data.vector_layer)["features"], crs="EPSG:4326"
        )

    # def on_simulator(self, ch, method, properties, body):

    def on_planner(self, ch, method, properties, body):
        """
        Responds to messages from planner application

        Inputs:
            ch (Channel): The channel on which the message was received.
            method (Method): The method used to receive the message.
            properties (Properties): The properties of the message.
            body (bytes): The body of the message.
        """
        component_gdf = self.message_to_geojson(body)
        component_gdf = self.process_component(component_gdf)
        self.master_components.append(component_gdf)
        self.counter += len(component_gdf)
        min_value = component_gdf["simulator_id"].min()
        max_value = component_gdf["simulator_id"].max()
        self.master_gdf = pd.concat(self.master_components, ignore_index=True)
        self.remove_duplicates()
        date = self.app.simulator._time
        date = str(date.date()).replace("-", "")
        self.master_gdf.to_file(f"master_{date}.geojson", driver="GeoJSON")
        selected_json_data = self.master_gdf.to_json()
        self.app.send_message(
            self.app.app_name,
            "master",  # ["master", "selected"],
            VectorLayer(vector_layer=selected_json_data).model_dump_json(),
        )
        if self.visualize_selected:
            self.app.send_message(
                "planner",
                "selected",
                VectorLayer(vector_layer=selected_json_data).model_dump_json(),
            )
        logger.info(f"{self.app.app_name} sent message.")

    def on_change(self, source, property_name, old_value, new_value):
        """
        Responds to changes in the simulator mode.

        Inputs:
            source (Simulator): The simulator that changed mode.
            property_name (str): The name of the property that changed.
            old_value (Mode): The old value of the property.
            new_value (Mode): The new value of the property.
        """
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            logger.info(
                "Switched to EXECUTING mode. Counter and master components list initialized."
            )
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.TERMINATING:
            logger.info(
                "Switched to TERMINATING mode. Shutting down the application and saving data."
            )


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

    app.add_message_callback("planner", "selected_cells", environment.on_planner)


if __name__ == "__main__":
    main()
