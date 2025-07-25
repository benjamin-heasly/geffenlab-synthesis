# This runs in an Ubuntu environment.
FROM ubuntu:22.04

# Install wget and https certificate info, which we'll use to install things, below.
# Install git so we can clone our population-analysis repo.
# Clean up cache and temp files along the way.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt install --no-install-recommends --yes wget ca-certificates git \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install conda.
RUN mkdir -p /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh -O /opt/conda/miniconda.sh \
    && bash /opt/conda/miniconda.sh -b -p /opt/miniconda

# Create our conda environment from environment.yml.
COPY environment/environment.yml /opt/environment/environment.yml
RUN /opt/miniconda/bin/conda env create -f /opt/environment/environment.yml

# Include all of our code/* in /opt/code
COPY code /opt/code
RUN mv /opt/code/population-analysis /opt/code/population_analysis
COPY code/conda_run /usr/local/bin/conda_run

# Give Matplotlib a writeable, user-independent cache directory.
# https://matplotlib.org/stable/install/environment_variables_faq.html
ENV MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir $MPLCONFIGDIR && chmod a+w $MPLCONFIGDIR
