#!/bin/bash

set -e

ENV_FILE="/usr/local/env/practice.env"

if [ -f "$ENV_FILE" ]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

# echo "Running entrypoint script..."

# if [[ ! -d "$PRACTICE_REPO" || ! -d "$ASSIGNMENTS_REPO" ]]; then
#     echo "Set the path where repo will be created in .env"
# fi

# echo "Cloning Practice repository..."
# git clone https://github.com/MLMethods/Practice "$PRACTICE_REPO"

# echo "Cloning Assignments repository..."
# git clone https://github.com/MLMethods/Assignments "$ASSIGNMENTS_REPO"

source /opt/conda/etc/profile.d/conda.sh
conda activate mlmethods

jupyter notebook --ip='*' --notebook-dir "$BASE_DIR" --allow-root

