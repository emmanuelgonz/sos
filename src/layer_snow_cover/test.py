import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sos_tools.efficiency import efficiency


from configparser import ConfigParser

config = ConfigParser()
config.read("Input_parameters.ini")
print(config.sections())

# config_data = config['Snow_cover'] 


# "../Input_parameters.ini"