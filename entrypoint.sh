#!/bin/bash

if [ "$1" = "snow_cover" ]; then
    exec /opt/miniconda/envs/sos_test/bin/python /opt/src/layer_snow_cover/main.py
elif [ "$1" = "resolution" ]; then
    exec /opt/miniconda/envs/sos_test/bin/python /opt/src/layer_resolution/main.py
else
    echo "Invalid argument. Use 'snow_cover' or 'resolution'."
    exit 1
fi