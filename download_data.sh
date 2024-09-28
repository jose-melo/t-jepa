#!/bin/bash

# Function to check if pip is installed
check_pip_installed() {
    if ! command -v pip &> /dev/null; then
        echo "pip is not installed. Please install pip first."
        exit 1
    fi
}

# Function to install gdown if it's not already installed
install_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "gdown is not installed. Installing gdown..."
        check_pip_installed
        pip install gdown
        if [ $? -ne 0 ]; then
            echo "Failed to install gdown. Please check your pip installation."
            exit 1
        fi
    fi
}

# Function to download a Google Drive file using gdown
download_file() {
    FILE_ID=$1
    FILE_NAME=$2

    # Use gdown to download the file
    gdown --id "$FILE_ID" -O "$FILE_NAME"
}

# Install gdown if it's not installed
install_gdown

# Download the file
download_file "16ZyP1Mvjbu9iNesxXH7zomjnEFM7u4mw" "dataset.zip"
