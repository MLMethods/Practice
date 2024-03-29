FROM debian:bullseye-slim

LABEL maintainer="Sergei Papulin <papulin.study@yandex.ru>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ARG SHA256SUM="e7ecbccbc197ebd7e1f211c59df2e37bc6959d081f2235d387e08c9026666acd"
ARG ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh"

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

# COPY ./Sources/Anaconda3-2022.10-Linux-x86_64.sh /usr/local/bin/

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

# Copy the entrypoint script

COPY --chmod=500 entrypoint.sh /usr/local/bin/

# Default command on start
ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]
