#! /usr/bin/env bash

REPO="tanmaniac"
IMAGE="cs231n"
TAG="cpu"
CONTAINER_NAME="${IMAGE}-${TAG}"

if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${CONTAINER_NAME})" ]; then
        # cleanup
        docker rm ${CONTAINER_NAME}
    fi
    docker run -it \
    --runtime=nvidia \
    --privileged \
    --name ${CONTAINER_NAME} \
    -p 8888:8888 -p 6006:6006 \
    -v $(pwd)/../assignment1:/home/$(id -un)/assignment1 \
    -v $(pwd)/../assignment2:/home/$(id -un)/assignment2 \
    ${REPO}/${IMAGE}:${TAG} \
    /bin/bash
fi
