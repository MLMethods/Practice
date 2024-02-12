#!/bin/bash

set -e

echo "Running entrypoint script..."

if [ ! -d /home/ubuntu/Repos/mlmethods/Practice ]; then
    echo "Cloning Practice repository..."
	git clone https://github.com/MLMethods/Practice /home/ubuntu/Repos/mlmethods/Practice
fi

if [ ! -d /home/ubuntu/Repos/mlmethods/Assignments ]; then
    echo "Cloning Assignments repository..."
    git clone https://github.com/MLMethods/Assignments /home/ubuntu/Repos/mlmethods/Assignments
fi

jupyter notebook --ip='*' --notebook-dir /home/ubuntu/Repos --allow-root