# Mico Obstacle Avoidance
## Dependencies
- [realsense-ros](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy)
- OpenCV
- PCL

## Hardware
- Intel Realsense D435 camera
- Mico
- chessboard printout

## Dockerfile
To use mico_obstacle_avoidance within Docker container, follow the [docker instructions](https://github.com/argallab/mico_base#using-mico_base-with-docker) for mico_base, but create the docker container using the `argallab/stephanie_video:base` image
```
sudo docker run -it --privileged \
-v /dev:/dev \
-v /dev/bus/usb:/dev/bus/usb \
-v /dev/input/by-id:/dev/input/by-id \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-v /home/stephanie/mico_ws/src:/home/mico_ws/src \
-e DISPLAY \
-e QT_X11_NO_MITSHM=1 \
--name UNIQUE_CONTAINER_NAME \
--net=host \
argallab/stephanie_video:base
```
Note: this docker image contains most of the dependencies, but you will likely still need to build `realsense-ros` from source.
## Usage
To launch obstacle avoidance, run
```
roslaunch mico_obstacle_avoidance camera.launch
```
Make sure that the chessboard printout is within the camera's view. You can check what the camera is seeing by viewing the `/camera/depth/color/points` point cloud in Rviz.

## Chessboard Placement
You will need to hardcode the transform from the bottom right corner of the chessboard to the center of mico's base. The translation is hardcoded as the variable `CHESS` at the top of [calibration.py](src/calibration.py).
