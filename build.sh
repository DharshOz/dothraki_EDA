#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating static directory..."
mkdir -p static