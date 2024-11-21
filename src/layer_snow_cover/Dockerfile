FROM emmanuelgonzalez/sos:latest

WORKDIR /opt
COPY . /opt

USER root

# Set the entrypoint
ENTRYPOINT [ "/opt/miniconda/envs/sos_test/bin/python", "/opt/main.py" ]