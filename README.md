# PIGS MONITORING SERVICE

## Installation

Install requirements:
`pip install -r requirements.txt`
Download `mars-small128.pb`, `yolov5m_pig_4_classes.pt` and paste into `weights` folder 

## Endpoints

### GET /video/
Get all videos from server

```json
Response:
[
    {
        "id": 14,
        "source": "static/uploads/demo.mp4",
        "subtitle": "not demo",
        "thumb": "static/uploads/demo.jpg",
        "title": "Uploaded video"
    }
]
```
### GET /video/`<id>`
Process video with `id`. 
```json
Response:
{
    "id": 14,
    "source": "static/uploads/demo.mp4",
    "statistic": [ ## All statistic of pig actions on video 
        {
            "eat": 41,
            "id": 1,
            "lying": 0,
            "run": 3,
            "standing": 42,
            "walk": 14
        },
    ],
    "subtitle": "not demo",
    "thumb": "static/uploads/demo.jpg",
    "title": "Uploaded video",
    "tracked_boxes": [
        [],
        [],
        [
            {
                "class_name": "standing",
                "pig_id": 1,
                "score": 0.7158662676811218,
                "xywhn": [
                    0.293359375,
                    0.49236111111111114,
                    0.12890625,
                    0.17083333333333334
                ]
            },
            {
                "class_name": "standing",
                "pig_id": 1,
                "score": 0.7158662676811218,
                "xywhn": [
                    0.393359375,
                    0.59236111111111114,
                    0.22890625,
                    0.27083333333333334
                ]
            },
        ]
    ]
}

```
### POST /video
Upload video to server. Required `file` in request
```json
Response:
{"status":"success"}
```


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
