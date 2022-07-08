import sys
import os
from statistics import mean
from typing import List
from collections import Counter

import imageio
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, './yolo5')

from deep_sort.generate_detections import create_box_encoder
from deep_sort.inference import DeepsortTracker, TrackedBox
from pigs_services import PigsMonitoringService
from yolo5.inference import Yolo
from yolo5.utils.plots import Annotator, colors

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'debug'

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


@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        result_filename = f'{filename.split(".")[0]}_result.gif'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        file.save(file_path)
        flash('Video successfully uploaded and displayed below')
        maks = os.path.join('static', 'mask.jpg')
        mask = None
        statistic = process_video(file_path, result_file_path, mask)
        processed_statistic = _get_statistic_per_pig(statistic)
        print(processed_statistic)
        return render_template('upload.html', filename=result_filename, statistic=processed_statistic)


@app.route('/display/<filename>')
def display_video(filename):
    # print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def process_video(video_path, result_path, paddock_mask_path=None) -> List[List[TrackedBox]]:
    cap = cv2.VideoCapture(video_path)
    with imageio.get_writer(result_path, mode='I') as writer:
        if paddock_mask_path is not None:
            paddock_mask = cv2.imread(paddock_mask_path, cv2.IMREAD_GRAYSCALE) > 100
            paddock_mask = np.repeat(paddock_mask[:, :, np.newaxis], 3, axis=2)
        pigs_movements = {}
        pigs_last_position = {}
        result_tracked_boxes = []
        for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            orig_frame = frame.copy()
            if paddock_mask_path is not None:
                frame = (frame * paddock_mask).astype(np.uint8)
            tracked_boxes = service.process_image(frame)
            annotator = Annotator(orig_frame, line_width=2)
            for box in tracked_boxes:
                box.class_name = _get_activity_statistic(box, pigs_movements, pigs_last_position)
                annotator.box_label((box.x_min, box.y_min, box.x_max, box.y_max),
                                    f"{box.tracking_id}: {box.class_name}, {box.score:.2}",
                                    color=colors(box.tracking_id, True))
            result_frame = annotator.result()
            writer.append_data(result_frame)
            result_tracked_boxes.append(tracked_boxes)
        cap.release()
    return result_tracked_boxes


def _get_statistic_per_pig(statistic):
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
        processed_statistic.append({'id': pig_id, **pig_statistic})
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
