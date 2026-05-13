FROM ubuntu:22.04
MAINTAINER Mingxun Wang "mwang87@gmail.com"

RUN apt-get update && apt-get install -y build-essential libarchive-dev wget vim git-core

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Adding to bashrc
RUN echo "export PATH=$CONDA_DIR:$PATH" >> ~/.bashrc

RUN mamba install -y -n base -c conda-forge \
	python=3.10 \
	numpy \
	scikit-bio \
	&& mamba clean -afy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pip-only packages and the GNPS git package with pip to avoid long solver time
RUN pip install --no-cache-dir git+https://github.com/Wang-Bioinformatics-Lab/GNPSDataPackage.git
	
COPY . /app
WORKDIR /app
