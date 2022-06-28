import sys

from yolo5.utils.general import xywh2xyxy

sys.path.insert(0, './yolo5')

from yolo5.inference import Yolo
from deep_sort.inference import DeepsortTracker, TrackedBox
from deep_sort.generate_detections import create_box_encoder
from typing import List
import cv2
import numpy as np


class PigsMonitoringService:
    def __init__(self, detector: Yolo, tracker: DeepsortTracker) -> None:
        self.detector = detector
        self.tracker = tracker

    def re_init_tracker(self):
        self.tracker.re_init()

    def process_image(self, image: np.array) -> List[TrackedBox]:
        image_boxes = self.detector.detect(image)
        # tracked_boxes = [TrackedBox(*xywh2xyxy(np.array([box.bbox]))[0], score=box.score, tracking_id=0, class_name=box.class_name) for box in image_boxes]
        tracked_boxes = self.tracker.track_boxes(frame=image, boxes=image_boxes)
        return tracked_boxes


# class VideoMonitoringService:
#     def __init__(self, detecotr: Yolo, tracker: DeepsortTracker) -> None:
#         self.image_monitoring_service = ImageMonitoringService(detecotr=detecotr, tracker=tracker)
#
#     def process_video(self, cap: cv2.VideoCapture) :
#
#         tracked_boxes = image_boxes = self.detecotr.detect(image)
#         boxes = [i.bbox for i in image_boxes]
#         tracked_boxes = self.image_monitoring_service.process_image()
#
#         return tracked_boxes