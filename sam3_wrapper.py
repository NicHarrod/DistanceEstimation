import logging
import numpy as np
import torch
from PIL import Image
from de_utils import DownloadableWeights


class SAM3(DownloadableWeights):
    def __init__(self):
        self._model_loaded = False
        self.device = None
        self.model = None
        self.processor = None

    def _load_model(self):
        if self._model_loaded:
            return

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = build_sam3_image_model(
                device=str(self.device),
                enable_inst_interactivity=True,
                load_from_HF=True,
            )

            self.processor = Sam3Processor(self.model, device=str(self.device))

            self._model_loaded = True
            logging.info(f"SAM3 loaded on {self.device}")

        except Exception as e:
            logging.error(f"Failed to load SAM3: {str(e)}")
            raise

        self.image_size = (1024, 1024)

    def __call__(self, img, boxes):
        self._load_model()

        img_rgb = np.ascontiguousarray(img[..., ::-1])
        # Sam3Processor handles PIL image size metadata correctly for coordinate normalization.
        inference_state = self.processor.set_image(Image.fromarray(img_rgb))
        h, w = img.shape[:2]

        mask_list = []
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = np.clip(x1, 0.0, float(w - 1))
            x2 = np.clip(x2, 0.0, float(w - 1))
            y1 = np.clip(y1, 0.0, float(h - 1))
            y2 = np.clip(y2, 0.0, float(h - 1))
            if x2 <= x1 or y2 <= y1:
                mask_list.append(np.zeros((h, w), dtype=bool))
                continue

            # SAM3 expects box prompts in XYXY format.
            box_input = np.array([x1, y1, x2, y2], dtype=np.float32)[None, :]

            masks, scores, logits = self.model.predict_inst(
                inference_state,
                box=box_input,
                multimask_output=False,
                normalize_coords=True,
            )

            mask = masks[0] > 0
            mask_list.append(mask)

        return np.array(mask_list)