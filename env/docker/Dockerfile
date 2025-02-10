FROM debian:bullseye-slim

LABEL maintainer="Sergei Papulin <papulin.study@yandex.ru>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ARG SHA256SUM="c536ddb7b4ba738bddbd4e581b29308cb332fa12ae3fa2cd66814bd735dff231"
ARG ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh"

# setup environment

RUN set -x \
    && apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# setup anaconda

 RUN set -x \
    && wget ${ANACONDA_URL} -O anaconda.sh -q \
    && echo "${SHA256SUM} anaconda.sh" > shasum \
    && sha256sum --check --status shasum \
    && /bin/bash anaconda.sh -b -p /opt/conda \
    && rm anaconda.sh shasum \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \ 
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \ 
    && echo "conda activate base" >> ~/.bashrc \ 
    && find /opt/conda/ -follow -type f -name '*.a' -delete \ 
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \ 
    && /opt/conda/bin/conda clean -afy

# Copy .env

COPY --chmod=644 .env /usr/local/env/practice.env

# Copy the entrypoint script

COPY --chmod=500 entrypoint.sh /usr/local/bin/

# Copy .yml with dependencies

COPY mlmethods.yml /usr/local/etc/mlmethods.yml

# Create conda and setup environment

RUN conda config --set ssl_verify false

RUN conda config --set solver classic && conda env create -f /usr/local/etc/mlmethods.yml --debug

RUN echo "conda activate mlmethods" >> ~/.bashrc 

RUN /bin/bash -c "source ~/.bashrc && conda activate mlmethods && pip install --upgrade typing_extensions && conda install jupyter"

# Default command on start
ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]
