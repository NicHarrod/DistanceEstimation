import numpy as np
import torch
import cv2

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAMDetectorLabel:
    ANIMAL = 0
    PERSON = 1
    VEHICLE = 2


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_boxes(boxes):
    boxes = _to_numpy(boxes)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes.reshape(-1, boxes.shape[-1])
    return boxes[:, :4].astype(np.float32)


def _normalize_scores(scores):
    return _to_numpy(scores).reshape(-1).astype(np.float32)


class SAMDetector:
    """
    SAM3 text-prompt detector that matches MegaDetector call signature.

    Input:
    img (np.ndarray): OpenCV BGR image.

    Returns:
    scores (N,) float32
    class_ids (N,) int32
    boxes (N, 4) float32 in xyxy format
    """

    def __init__(self, prompt: str = "Sign Being Held", conf: float = 0.1):
        self.prompt = "Sign Being Held"
        self.set_prompt(prompt)
        self.conf = conf
        self._model_loaded = False
        self.device = None
        self.model = None
        self.processor = None

    def set_prompt(self, prompt: str):
        sanitized_prompt = "" if prompt is None else str(prompt).strip()
        self.prompt = sanitized_prompt if sanitized_prompt else "Sign Being Held"

    def _load_model(self):
        if self._model_loaded:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading SAM3 model on device: {self.device}")

        self.model = build_sam3_image_model(
            device=self.device,
            # Text-prompt detection does not need SAM1-style interaction.
            # Keeping this off avoids initializing tracker code that enables
            # a global bf16 autocast context.
            enable_inst_interactivity=False,
            load_from_HF=True,
        )
        self.processor = Sam3Processor(self.model, device=self.device)
        self._model_loaded = True

    def __call__(self, img: np.ndarray):
        self._load_model()

        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Expected a valid OpenCV image array, got None/non-array input")
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError(
                "Expected a non-empty OpenCV image. This usually means the frame became empty after cropping."
            )

        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unsupported image shape for SAMDetector: {img.shape}")

        image = Image.fromarray(img_rgb)

        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=self.prompt)

        boxes = _normalize_boxes(output.get("boxes", np.zeros((0, 4), dtype=np.float32)))
        scores = _normalize_scores(output.get("scores", np.zeros((0,), dtype=np.float32)))

        if boxes.shape[0] == 0 or scores.shape[0] == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32),
            )

        keep = np.nonzero(scores >= self.conf)[0]       
        if keep.size == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32),
            )

        scores = scores[keep]
        boxes = boxes[keep]
        class_ids = np.full(scores.shape[0], SAMDetectorLabel.ANIMAL, dtype=np.int32)

        return scores, class_ids, boxes