import sys
import os
import logging
import numpy as np
import cv2
import torch
from de_utils import DownloadableWeights, condition_disparity


class Metric3D(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False
        self.device = None
        self.model = None

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        try:
            from metric3d.apis import SingleImageInferencer
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load Metric3D ViT-S model
            self.model = SingleImageInferencer(
                ckpt_path='https://huggingface.co/JUGGHM/Metric3D-vit-small/resolve/main/metric_depth_vit_small.pth',
                device=self.device
            )
            
            logging.info(f"Metric3D loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load Metric3D: {str(e)}")
            raise

    def __call__(self, img):
        # ensure model is loaded
        self._load_model()

        # BGR to RGB
        img_rgb = img[..., ::-1]

        # Run inference
        with torch.inference_mode():
            # The SingleImageInferencer handles preprocessing and returns metric depth
            output = self.model.infer_cv2(img_rgb)
            
            # output is a dict with 'metric_depth' and other keys
            prediction = output['metric_depth']

        # Convert metric depth to disparity (inverse)
        # Clip to avoid division by zero
        prediction = np.clip(prediction, 1e-6, np.inf) ** -1
        prediction = prediction.astype(np.float32)
        prediction = condition_disparity(prediction)
        
        return prediction