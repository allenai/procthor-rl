FROM ghcr.io/allenai/cuda:11.8-ubuntu20.04-v1.2.27 AS conda_env_builder

ENV APP_HOME /root/procthor-rl
WORKDIR $APP_HOME

RUN conda create -n procthor-rl -y python=3.9

RUN /opt/miniconda3/bin/conda install -n procthor-rl -y -c conda-forge jsonnet ipython

FROM conda_env_builder AS requirements_installer

ENV PYTHON=/opt/miniconda3/envs/procthor-rl/bin/python
ENV PIP=/opt/miniconda3/envs/procthor-rl/bin/pip

COPY ./requirements.txt /root/requirements.txt

RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/procthor-rl/pipsrc; \
    $PIP install --no-cache-dir setuptools==66.0.0 \
    && $PIP install --no-cache-dir -r /root/requirements.txt \
)

RUN $PIP install --no-cache-dir --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+ca10d107fb46cb051dba99af484181fda9947a28
RUN $PIP install --no-cache-dir torch==2.0.1 torchvision open_clip_torch objaverse objathor

# ffmpeg from conda is broken. We'll install it at a system level
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 nano vulkan-tools libvulkan1 vulkan-utils -y

# Clean up conda to remove unused packages
RUN /opt/miniconda3/bin/conda clean -ya 

RUN touch /root/.git-credentials

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
ENTRYPOINT ["bash", "-l"]