FROM ubuntu:22.04
LABEL maintainer="Mingxun Wang <mwang87@gmail.com>"

RUN apt-get update && apt-get install -y build-essential libarchive-dev wget vim git-core

# System libraries needed by the headless Chrome that Kaleido uses for PNG/PDF figure export
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
	libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libcups2 \
	libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxkbcommon0 libgbm1 libdrm2 \
	libcairo2 libpango-1.0-0 libasound2 fonts-liberation \
	&& rm -rf /var/lib/apt/lists/*

# Install Mamba
ENV CONDA_DIR=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Adding to bashrc
RUN echo "export PATH=$CONDA_DIR/bin:$PATH" >> ~/.bashrc

RUN mamba install -y -n base -c conda-forge \
	python=3.12 \
	numpy \
	scikit-bio \
	&& mamba clean -afy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kaleido 1.x does not bundle a browser; download Chrome for figure export
RUN plotly_get_chrome -y

# Install pip-only packages and the GNPS git package with pip to avoid long solver time
RUN pip install --no-cache-dir git+https://github.com/Wang-Bioinformatics-Lab/GNPSDataPackage.git

COPY . /app
WORKDIR /app
