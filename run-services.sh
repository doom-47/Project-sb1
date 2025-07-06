#!/bin/bash

# --- Start: Robust .env loading ---
# Check if .env file exists and load it
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  # Use 'set -a' to automatically export variables defined in the sourced file
  set -a
  . ./.env # Source the .env file
  set +a
  echo "Environment variables loaded."
else
  echo "Warning: .env file not found. Ensure TOKEN is set in your environment."
fi
# --- End: Robust .env loading ---

# Check if TOKEN is now set
if [ -z "$TOKEN" ]; then
  echo "Error: TOKEN environment variable is not set. Please ensure it's in your .env file."
  exit 1
fi

# Run SearXNG
# Note: If you are using docker-compose.yml for your FastAPI app,
# you might manage SearXNG and Browserless through it as well.
# This script is for manual docker run commands.
echo "Starting SearXNG container..."
docker run -d --name searxng -p 8080:8080 -v ./searxng:/etc/searxng:rw searxng/searxng

# Run Browserless
# Pass the TOKEN loaded from the .env file to the browserless container
echo "Starting Browserless container with TOKEN: $TOKEN..."
docker run -d --name browserless -p 3000:3000 -e "TOKEN=$TOKEN" ghcr.io/browserless/chromium

echo "SearXNG is running at http://localhost:8080"
echo "Browserless is running at http://localhost:3000"