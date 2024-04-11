# Using ProcTHOR-RL with Docker

## Building the Docker image

These instructions assume you are building and running this Docker image on a Linux machine with an nvidia graphics
card. You'll need to have the `docker` installed along with the `nvidia-container-toolkit`, you can install the 
`nvidia-container-toolkit` by running:
```bash
# Installing nvidia-container-toolkit
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit-base nvidia-container-toolkit

# Need to restart docker to get the `nvidia-container-toolkit` to be usable
sudo systemctl restart docker
```

Change your current working directory to this directory:
```bash
cd /path/to/procthor-rl/docker
```

You can now build the docker image by running
```bash
create_image.sh
```
This will print out the name of the docker image, this will look something like
```
procthor-rl-base-20240301
```

## Running the Docker image

Let's assume you've built the docker image using the above instructions and this has created the docker image
```bash
DOCKER_IMAGE=procthor-rl-base-20240301
```
You can now start this Docker image mounting your local procthor-rl repository to `/root/procthor-rl` in the Docker image by running:
```bash
procthor-rl_PATH=/path/to/procthor-rl
DATA_PATH=/path/to/data
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${PROCTHOR_PATH},target=/root/procthor-rl \
    --shm-size 250G \
    -it ${DOCKER_IMAGE}:latest
```
You should set `--shm-size 250G` to be a reasonable size for your system, you will need a large shared memory when
running training as the dataloader will load a lot of data into shared memory.

Alternatively, you could use our helper script to start the docker interactive session by running the following command from the topmost directory of the procthor-rl repository:
```bash
bash create_image.sh
```