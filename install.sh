#!/bin/bash

echo "Updating package list..."
sudo apt update -y

echo "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git

# Set up Python virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Setting up virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Navigate to pyQSC directory and install as editable
if [ -d "pyQSC" ]; then
    echo "Navigating to pyQSC directory and installing package..."
    cd pyQSC
    pip install -e .
else
    echo "Directory 'pyQSC' not found. Skipping editable installation."
fi

echo "Installation complete. Activate your virtual environment with 'source venv/bin/activate'"
