import sys
import os
import json
from dataclasses import asdict
from statistics import mean
from typing import List
from collections import Counter

import imageio
from flask import Flask, flash, request, redirect, url_for, render_template, abort, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import warnings
from videos_db import VideosJsonDB, VideoData

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, './yolo5')

from deep_sort.generate_detections import create_box_encoder
from deep_sort.inference import DeepsortTracker, TrackedBox
from pigs_services import PigsMonitoringService
from yolo5.inference import Yolo
# from yolo5.utils.plots import Annotator, colors
from dataclasses import dataclass

UPLOAD_FOLDER = 'static/uploads/'
VIDEO_DB_PATH = 'data/videos.json'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'debug'
db = VideosJsonDB(os.path.join('data', 'videos.json'))

ACTIONS = ['eat', 'walk', 'standing', 'lying', 'run']
yolo = Yolo(weights="/home/user/Luxoft/animals_recognition/pigs-monitoring-service/weights/yolov5m_pig_4_classes.pt",
            img_size=(640, 640))
encoder_path = 'weights/mars-small128.pb'
feature_extractor = create_box_encoder(encoder_path)
tracker = DeepsortTracker(model_path=encoder_path)
service = PigsMonitoringService(detector=yolo, tracker=tracker)


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/video/', methods=['get'])
def get_videos():
    videos = db.get_videos()
    return jsonify([asdict(video) for video in videos])


@app.route('/video/<_id>', methods=['get'])
def process_video(_id):
    video = db.get_video_by_id(_id)
    if video is None:
        abort(404)
    if video.subtitle == "demo":
        mask = os.path.join('static', 'mask.jpg')
    else:
        mask = None
    tracked_boxes = process_video(video.source, mask)
    statistic = _get_statistic_per_pig(tracked_boxes)
    output = {
        **asdict(video),
        'statistic': [asdict(pig) for pig in statistic],
        'tracked_boxes': [[{"pig_id": box.tracking_id,
                            'xywhn': box.xywhn,
                            'score': box.score,
                            'class_name': box.class_name}
                           for box in frame] for frame in tracked_boxes]
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


def process_video(video_path, paddock_mask_path=None) -> List[List[TrackedBox]]:
    cap = cv2.VideoCapture(video_path)
    if paddock_mask_path is not None:
        paddock_mask = cv2.imread(paddock_mask_path, cv2.IMREAD_GRAYSCALE) > 100
        paddock_mask = np.repeat(paddock_mask[:, :, np.newaxis], 3, axis=2)
    pigs_movements = {}
    pigs_last_position = {}
    result_tracked_boxes = []
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        # orig_frame = frame.copy()
        if paddock_mask_path is not None:
            frame = (frame * paddock_mask).astype(np.uint8)
        tracked_boxes = service.process_image(frame)
        # annotator = Annotator(orig_frame, line_width=2)
        for box in tracked_boxes:
            box.class_name = _get_activity_statistic(box, pigs_movements, pigs_last_position)
            # annotator.box_label((box.x_min, box.y_min, box.x_max, box.y_max),
            #                     f"{box.tracking_id}: {box.class_name}, {box.score:.2}",
            #                     color=colors(box.tracking_id, True))
        # result_frame = annotator.result()
        result_tracked_boxes.append(tracked_boxes)
    cap.release()
    return result_tracked_boxes


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



def _get_statistic_per_pig(statistic) -> List[PigStatistic]:
    pigs_boxes = {}
    for frame in statistic:
        for item in frame:
            pig_boxes = pigs_boxes.get(item.tracking_id, [])
            pig_boxes.append(item)
            pigs_boxes[item.tracking_id] = pig_boxes
    processed_statistic = []
    for pig_id, boxes in pigs_boxes.items():
        pig_statistic = Counter([box.class_name for box in boxes])
        pig_statistic = {key: round((pig_statistic.get(key, 0) / len(boxes)) * 100) for key in ACTIONS}
        processed_statistic.append(PigStatistic(id=pig_id, **pig_statistic))
    return processed_statistic


def _get_activity_statistic(box: TrackedBox, pigs_movements: dict, pigs_last_position: dict):
    pig_movements = pigs_movements.get(box.tracking_id, [])
    position = ((box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2)
    if box.tracking_id in pigs_last_position:
        last_position = pigs_last_position[box.tracking_id]
        movement = ((last_position[0] - position[0]) ** 2 + (last_position[1] - position[1]) ** 2) ** 0.5
        pig_movements.append(movement)
    pigs_last_position[box.tracking_id] = position
    pigs_movements[box.tracking_id] = pig_movements
    if len(pig_movements) > 3:
        activity = mean(pig_movements[-3:])
    else:
        activity = 0
    pig_size = ((box.x_max - box.x_min) + (box.y_max - box.y_min)) / 2
    if activity > 0.1 * pig_size:
        return "run"
    elif activity > 0.05 * pig_size:
        return 'walk'
    return box.class_name


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8889)
    # process_video('color.mp4', 'mask.jpg')
