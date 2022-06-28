# PIGS MONITORING SERVICE

## Installation

Install requirements:
`pip install -r requirements.txt`

## Docker
You can also build docker image and run the inference in the container.

### Usage:
To build use:

`docker build -t pigs/service .`

To use the GPU, you need `--gpus=all` option. See the guide for more details.

Example:
`docker run --gpus=all -it --rm pigs/service`

## Demo

![App demo](static/demo.gif)