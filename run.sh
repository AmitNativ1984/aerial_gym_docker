#!/bin/bash

# Run aerial_gym Docker container with proper GPU and graphics support

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "Error: DISPLAY environment variable is not set"
    echo "Please ensure X server is running and DISPLAY is configured"
    exit 1
fi

# Allow docker containers to connect to X server
echo "Allowing X11 forwarding..."
xhost +local:docker > /dev/null 2>&1

# Cleanup function
cleanup() {
    xhost -local:docker > /dev/null 2>&1
}
trap cleanup EXIT

echo "Starting aerial_gym container with GUI support (DISPLAY=$DISPLAY)..."

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/dri \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  --network host \
  aerial-gym:ig "$@"
