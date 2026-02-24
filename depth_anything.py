import logging
import json
import numpy as np
import cv2
import torch
from transformers import AutoModelForDepthEstimation
from de_utils import DownloadableWeights


class DepthAnything(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False
        self.device = None

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        # ---- SAME MODEL FAMILY AS ONNX (V1 BASE) ----
        model_name = "LiheYoung/depth-anything-base-hf"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.eval().to(self.device)

        # ---- HARDCODE values from ONNX metadata ----
        # These must match code 1 exactly
        self.net_w, self.net_h = 518, 518
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.prediction_factor = 1000.0  # typical for metric model

        logging.info(f"Depth Anything V1 loaded on {self.device}")

    def __call__(self, img):
        self._load_model()

        # ---- EXACT SAME PREPROCESSING AS CODE 1 ----

        # BGR → RGB
        img = img[..., ::-1]

        # 0..1 scaling
        img = img / 255.0

        # Resize to network size
        img_input = cv2.resize(img, (self.net_w, self.net_h), cv2.INTER_AREA)

        # Normalize
        img_input = (img_input - self.mean) / self.std

        # HWC → CHW
        img_input = img_input.transpose(2, 0, 1)

        # Add batch dim
        img_input = torch.from_numpy(img_input).unsqueeze(0).float().to(self.device)

        # ---- INFERENCE ----
        with torch.inference_mode():
            prediction = self.model(pixel_values=img_input).predicted_depth

        # Remove batch/channel dims
        prediction = prediction[0, 0].cpu().numpy()

        # Resize back to original
        prediction = cv2.resize(
            prediction,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        print(prediction.min(), prediction.max(), prediction.mean())
        # Apply same scaling as ONNX
        prediction *= self.prediction_factor
        print(prediction.min(), prediction.max(), prediction.mean())

        return prediction
