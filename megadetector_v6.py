import logging
import numpy as np
import torch
from ultralytics import YOLO


class MegaDetectorLabel:
    ANIMAL = 0
    PERSON = 1
    VEHICLE = 2


class MegaDetectorV6:
    """
    MegaDetector wrapper that correctly loads the **YOLOv10-based** MegaDetector weights.

    Important:
    - MDV6-yolov10-e-1280.pt is a YOLOv10 model, NOT YOLOv8.
    - Therefore we must use torch.hub with the YOLOv10 repo.

    This version is:
    ✔ Compatible with official MegaDetector weights
    ✔ GPU aware
    ✔ Works on Windows + HPC
    ✔ Same return format as before
    """

    def __init__(self, weights_path: str = "weights/MDV6-yolov10-e-1280.pt", conf: float = 0.1):
        self.weights_path = weights_path
        self.conf = conf
        self._model_loaded = False
        self.device = None
        self.model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
            if self._model_loaded:
                return
            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.weights_path)
            self._model_loaded = True

    def __call__(self, img: np.ndarray):
        self._load_model()

        # pass OpenCV BGR image directly
        r = self.model.predict(
            source=img,
            conf=self.conf,
            imgsz=1280,   # matches MDV6-yolov10-*-1280 naming
            device=self.device,
            verbose=False
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32),
            )

        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = r.boxes.cls.cpu().numpy().astype(np.int32)
        return scores, class_ids, boxes

