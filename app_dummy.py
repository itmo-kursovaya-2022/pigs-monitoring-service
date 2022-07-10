import sys
import os
import json
from dataclasses import asdict
from statistics import mean
from typing import List
from collections import Counter

from flask import Flask, flash, request, redirect, url_for, render_template, abort, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import warnings
from videos_db import VideosJsonDB, VideoData

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, './yolo5')

from dataclasses import dataclass

UPLOAD_FOLDER = 'static/uploads/'
VIDEO_DB_PATH = 'data/videos.json'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'debug'
db = VideosJsonDB(os.path.join('data', 'videos.json'))

ACTIONS = ['eat', 'walk', 'standing', 'lying', 'run']

@app.route('/video/', methods=['get'])
def get_videos():
    output = [
    {
        "id": 15,
        "source": "static/uploads/demo.mp4",
        "subtitle": "not demo",
        "thumb": "static/uploads/demo.jpg",
        "title": "Uploaded video"
    },
    {
        "id": 14,
        "source": "static/uploads/demo2.mp4",
        "subtitle": "not demo",
        "thumb": "static/uploads/demo.jpg",
        "title": "Uploaded video"
    },
    {
        "id": 16,
        "source": "static/uploads/demo3.mp4",
        "subtitle": "not demo",
        "thumb": "static/uploads/demo.jpg",
        "title": "Uploaded video"
    }
]
    return jsonify(output)


@app.route('/video/<_id>', methods=['get'])
def process_video(_id):
    output = {
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
    return output


@app.route('/video/', methods=['post'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return abort(400)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return abort(400)
    else:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        request_json = request.get_json()
        if request_json is None:
            request_json = {}
        video = VideoData(id=request_json.get('id', len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1),
                          title=request_json.get('title', 'Uploaded video'),
                          thumb=create_thumb(file_path),
                          source=file_path,
                          subtitle=request_json.get('subtitle', 'not demo'),
                          )
        db.add_video(video)
        flash('Video successfully uploaded and displayed below')
        return {"status": "success"}, 200

@dataclass()
class PigStatistic:
    id: int
    run: float = 0
    walk: float = 0
    eat: float = 0
    lying: float = 0
    standing: float = 0


def create_thumb(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2))
    ret, frame = cap.read()
    thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{video_path.split("/")[-1].split(".")[0]}.jpg')
    cv2.imwrite(thumb_path, frame)
    return thumb_path


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8889)
    # process_video('color.mp4', 'mask.jpg')
