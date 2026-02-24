import sys
import os
import logging
import numpy as np
import cv2
import torch
from de_utils import DownloadableWeights


class SAM(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False
        self.device = None
        self.model = None
        self.predictor = None

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True

        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Download and load SAM ViT-B model
            weights_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            weights_md5 = "01ec64d29a2fca3f0661936605ae66f8"
            weights_path = self.get_weights(weights_url, weights_md5)
            
            sam = sam_model_registry["vit_b"](checkpoint=weights_path)
            sam = sam.to(self.device)
            
            self.predictor = SamPredictor(sam)
            
            logging.info(f"SAM loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load SAM: {str(e)}")
            raise

        self.image_size = (1024, 1024)

    def __call__(self, img, boxes):
        # ensure model is loaded
        self._load_model()

        img_rgb = img[..., ::-1]
        original_size = img_rgb.shape[:2]
        
        # Set the image for the predictor
        self.predictor.set_image(img_rgb)

        mask_list = []
        for box in boxes:
            # Convert box format [x1, y1, x2, y2] to what SAM expects
            box_input = np.array(box, dtype=np.float32)
            
            # Get masks for this box
            masks, scores, logits = self.predictor.predict(
                box=box_input,
                multimask_output=False
            )
            
            # Take the mask with highest confidence
            mask = masks[0] > 0.0
            mask_list.append(mask)

        return np.array(mask_list)