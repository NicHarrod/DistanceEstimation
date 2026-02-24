import logging
import numpy as np
import torch


class MegaDetectorLabel:
    ANIMAL = 0
    PERSON = 1
    VEHICLE = 2


class MegaDetector:
    """
    MegaDetector wrapper that correctly loads the **YOLOv5-based** MegaDetector weights.

    Important:
    - md_v5a.0.0.pt is a YOLOv5 model, NOT YOLOv8.
    - Therefore we must use torch.hub with the YOLOv5 repo.

    This version is:
    ✔ Compatible with official MegaDetector weights
    ✔ GPU aware
    ✔ Works on Windows + HPC
    ✔ Same return format as before
    """

    def __init__(self, weights_path: str = "weights/md_v5a.0.0.pt", conf: float = 0.1):
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

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Correct way to load YOLOv5 MegaDetector
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=self.weights_path,
                source="github",
            )

            self.model.to(self.device)
            self.model.conf = self.conf

            self._model_loaded = True
            logging.info(f"MegaDetector (YOLOv5) loaded on {self.device}")

        except Exception as e:
            logging.error(f"Failed to load MegaDetector: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def __call__(self, img: np.ndarray):
        """
        Runs MegaDetector on a BGR OpenCV image.

        Returns:
            scores (N,)
            class_ids (N,)
            boxes (N, 4) in xyxy format
        """

        self._load_model()

        if self.model is None:
            raise RuntimeError("MegaDetector model failed to load.")

        # YOLOv5 expects RGB
        img_rgb = img[..., ::-1]

        results = self.model(img_rgb, size=640)

        # No detections
        if len(results.xyxy[0]) == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32),
            )

        detections = results.xyxy[0].cpu().numpy()

        boxes = detections[:, :4].astype(np.float32)
        scores = detections[:, 4].astype(np.float32)
        class_ids = detections[:, 5].astype(np.int32)

        return scores, class_ids, boxes
