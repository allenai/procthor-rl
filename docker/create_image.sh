BASE_IMAGE="procthor-rl-$(date +"%Y%m%d")" \
  && DOCKER_BUILDKIT=1 docker build -t \
   $BASE_IMAGE:latest \
   --file Dockerfile \
   .

echo "Docker image name: ${BASE_IMAGE}"