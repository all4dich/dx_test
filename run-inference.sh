#!/bin/bash

# Get the current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the absolute path of the dxnn file. If not defined use the default path
if [ -z "$1" ]; then
  echo "INFO: dxnn file path not provided, using default path."
  DXNN_FILE_PATH="${HOME}/workspace/simulator/examples/compiled_results/yolov5s.dxnn"
else
  DXNN_FILE_PATH=$(realpath "$1")
fi
# Get the absolute path of the mp4 file
# IF NOT DEFINED USE THE DEFAULT PATH
if [ -z "$2" ]; then
  echo "INFO: mp4 file path not provided, using default path."
  MP4_FILE_PATH="${HOME}/Airshow.mp4"
else
  MP4_FILE_PATH=$(realpath "$2")
fi
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Error: Python virtual environment is not activated."
  echo "INFO: Activating virtual environment..."
  if [ -d ".venv" ]; then
      source .venv/bin/activate
  else
      source ${HOME}/program/venv/bin/activate
  fi
fi
set -x
python3 ./main.py ${DXNN_FILE_PATH} ${MP4_FILE_PATH}
set +x