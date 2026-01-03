#!/bin/bash
set -e

echo "IsaacGym GUI Test"
echo "=================="

if [ -z "$DISPLAY" ]; then
    echo "Error: DISPLAY not set"
    exit 1
fi

echo "Building image..."
docker build -t aerial-gym:ig .

echo "Running test (1080_balls_of_solitude.py)..."
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
  -w /opt/isaacgym/python/examples \
  aerial-gym:ig \
  python 1080_balls_of_solitude.py

xhost -local:docker
echo "Test complete!"
