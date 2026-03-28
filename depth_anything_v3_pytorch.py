import os
import cv2
import torch
import numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3
from pathlib import Path

class Depth_Anything_3:
    def __init__(
        self,
        checkpoint_dir=None,
        device=None,
        process_res=504,
        process_res_method="upper_bound_resize",
    ):
        # Resolve from this file's location, not current working directory
        if checkpoint_dir is None:
            repo_root = Path(__file__).resolve().parent.parent
            checkpoint_dir = repo_root / "depth-anything-3" / "checkpoints" / "da3_mono_large"

        checkpoint_dir = str(Path(checkpoint_dir).resolve())

        if not os.path.isdir(checkpoint_dir):
            raise RuntimeError(f"Checkpoint directory not found: {checkpoint_dir}")

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading Depth Anything 3 from {checkpoint_dir}")
        self.model = DepthAnything3.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.process_res = process_res
        self.process_res_method = process_res_method

        print(f"Depth Anything V3 loaded on {self.device}")

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: OpenCV image (H, W, 3) BGR uint8
        returns: depth map (H, W) float32
        """

        if img_bgr is None:
            raise ValueError("Input image is None")

        # Store original shape for resizing output
        original_shape = img_bgr.shape[:2]

        # Convert BGR → RGB
        img_rgb = img_bgr[..., ::-1]

        # Convert to PIL
        pil_img = Image.fromarray(img_rgb)

        # Run inference using official pipeline
        prediction = self.model.inference(
            [pil_img],
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )

        depth = prediction.depth[0]
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        
        # Resize depth output back to original input image size
        depth = cv2.resize(depth, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        
        return depth.astype(np.float32)