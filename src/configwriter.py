
# This code writes the configuration file in the standar format

from configparser import ConfigParser

config = ConfigParser()

config["DEFAULT"] = {
    "Threshold": 50,
    "Coefficient": 0.003
}

config["Resolution"] = {
    "Threshold": 40,
    "Coefficient": 0.003
}

config["Snow_cover"] = {
    "Threshold": 30,
    "Coefficient": 0.5
}

with open("Input_parameters.ini", "w") as f:
    config.write(f)