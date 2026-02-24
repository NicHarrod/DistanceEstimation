import torch
import numpy as np
import cv2
import os
from depth_anything_3.api import DepthAnything3
from de_utils import DownloadableWeights


class Depth_Anything_3(DownloadableWeights):
    def __init__(self, checkpoint_path=None, model_name="da3mono-large", device=None):
        self._model_loaded = False
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_checkpoint_path(self):
        if self.checkpoint_path is not None:
            if os.path.isfile(self.checkpoint_path):
                return self.checkpoint_path
            raise RuntimeError(f"Checkpoint file not found: '{self.checkpoint_path}'")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "weights", "da3_mono_large.safetensors"),
            os.path.join(base_dir, "..", "depth-anything-3", "checkpoints", "da3_mono_large.safetensors"),
            os.path.join(base_dir, "..", "da3_mono_large.safetensors"),
        ]
        for candidate in candidates:
            path = os.path.normpath(candidate)
            if os.path.isfile(path):
                return path

        checked_paths = "\n".join([os.path.normpath(path) for path in candidates])
        raise RuntimeError(
            "Local DA3 checkpoint not found. Place 'da3_mono_large.safetensors' in one of these paths:\n"
            f"{checked_paths}"
        )

    def _load_model(self):
        if self._model_loaded:
            return

        checkpoint_path = self._resolve_checkpoint_path()

        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'safetensors'. Install it with: pip install safetensors"
            ) from e

        state_dict = load_safetensors_file(checkpoint_path, device="cpu")

        self.model = DepthAnything3(model_name=self.model_name)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys while loading DA3 checkpoint '{checkpoint_path}'. "
                "This usually means checkpoint/model mismatch. "
                f"model_name='{self.model_name}', unexpected_keys_count={len(unexpected_keys)}"
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        self._model_loaded = True
        print(f"Depth Anything V3 loaded on {self.device}")

    def __call__(self, img):
        """
        img: BGR uint8 numpy image (H, W, 3)
        returns: depth map (H, W) float32 numpy
        """
        self._load_model()

        # BGR → RGB
        img = img[..., ::-1]

        # Save original size
        orig_h, orig_w = img.shape[:2]

        # Convert to float and normalize to 0..1
        img = img.astype(np.float32) / 255.0

        # Convert to tensor (HWC → CHW)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            depth = self.model(img_tensor)

        # Some DA3 models return dict
        if isinstance(depth, dict):
            depth = depth["depth"]

        depth = depth.squeeze().cpu().numpy()

        # Resize back to original resolution
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return depth.astype(np.float32)

