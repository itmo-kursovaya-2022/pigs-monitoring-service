FROM nvidia/cuda:11.2.0-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*


# Install python3.8
RUN sudo apt-get -y update \
 && sudo apt-get -y install software-properties-common \
 && sudo add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && sudo apt-get -y install python3.8 \
 && sudo apt-get -y install python3.8-dev \
 && sudo apt-get -y install python3-pip \
 && python3.8 -m pip install --upgrade pip \
 && sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN apt-get update -y
RUN apt-get install -y libsm6 libxrender1 libfontconfig1
RUN apt-get install -y libxext6 libgl1-mesa-glx

ARG APP_DIR=/app
WORKDIR "$APP_DIR"

COPY requirements.txt $APP_DIR/

RUN pip install -U pip
RUN pip install -U setuptools
RUN pip install -r requirements.txt

COPY . $APP_DIR/
ENTRYPOINT ["python3", "app.py"]