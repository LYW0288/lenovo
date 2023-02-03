#!/bin/bash
NOW_PATH=$(cd $(dirname $0); pwd)
PARANT_PATH="${NOW_PATH%/*}"
BASE_PATH="${PARANT_PATH%/*}"
IMAGE_NAME="lenovo-test"
CONTAINER_NAME="docker-lenovo-test"
# docker build -t $IMAGE_NAME  --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$http_proxy --build-arg NO_PROXY="$NO_PROXY" --build-arg http_proxy=$http_proxy --build-arg https_proxy=$http_proxy --build-arg no_proxy="$NO_PROXY" .

docker build -t $IMAGE_NAME  .

# nvidia-docker run -d -it --name $CONTAINER_NAME -v ${BASE_PATH}/output/model/darknet/:/opt/program/history/ -v ${BASE_PATH}/working_dir/config/:/opt/ml/input/config/ -v ${BASE_PATH}/output/model/tensorflow/:/opt/ml/model/ -v ${BASE_PATH}/input/training_zip/:/opt/ml/input/data/train/   $IMAGE_NAME /bin/bash 
