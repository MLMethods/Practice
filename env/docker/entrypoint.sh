#!/bin/bash

set -e

ENV_FILE="/usr/local/env/practice.env"

if [ -f "$ENV_FILE" ]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate mlmethods

jupyter notebook --ip='*' --notebook-dir "$BASE_DIR" --allow-root

