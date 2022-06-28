from collections import namedtuple
from typing import List

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
import numpy as np
import torch



YoloDetection = namedtuple('YoloDetection', 'bbox score class_name')



class Yolo:

    def __init__(self, weights, img_size, device='0', dataset_config_path=None, conf_thres=0.25, iou_thres=0.45,
                 max_det=1000):
        """"""

        self.device = select_device(device)
        self.img_size = img_size
        self.model = DetectMultiBackend(weights, data=dataset_config_path, device=self.device,
                                        dnn=False,  fp16=False)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def detect(self, image: np.ndarray) -> List[YoloDetection]:
        """

        :param image: BGR np.ndarray
        :return:
        """
        im = self._preprocess_img(image)
        im = torch.from_numpy(im).to(self.device, dtype=torch.float)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)
        pred_boxes = []
        # Process predictions
        for det in pred:  # per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = torch.tensor(xyxy).tolist()
                    xywh[2] = xywh[2] - xywh[0]
                    xywh[3] = xywh[3] - xywh[1]
                    pred_boxes.append(YoloDetection(xywh, conf.item(), self.model.names[int(cls)]))
        return pred_boxes

    def _preprocess_img(self, image):
        # Padded resize
        image = letterbox(image, self.img_size, stride=32, auto=True)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        return np.ascontiguousarray(image)