# check if DOCKER_IMAGE exists and if not ask the user to export it
if [ -z ${DOCKER_IMAGE} ]; then
    echo "Please export DOCKER_IMAGE envionrment variable first as follows:"
    echo "    export DOCKER_IMAGE=<name-of-your-docker-image>"
    echo "To get the names of available docker images run:"
    echo "    docker images"
    exit 1
fi

# if PROCTHOR_PATH isn't provided, use the current directory
if [ -z ${PROCTHOR_PATH} ]; then
    PROCTHOR_PATH=$(pwd)
    echo "Mounting the current directory ${PROCTHOR_PATH} at /root/procthor-rl"
    echo "If you wish to mount a different directory, export the environment variable SPOC_PATH as follows:"
    echo "    export PROCTHOR_PATH=/path/to/procthor-rl/directory"
else
    echo "Mounting ${PROCTHOR_PATH} at /root/procthor-rl"
fi

# launch an interactive session with the docker image
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${PROCTHOR_PATH},target=/root/procthor-rl \
    --shm-size 50G \
    -it ${DOCKER_IMAGE}:latest