"""Optional MONAI 3D U-Net inference for the MSD Task06_Lung NIfTI dataset."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SegmentationService:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = "not-loaded"
        self.torch = None
        self.model = None
        self.load_error: str | None = None
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            self.load_error = f"3D segmentation checkpoint not found at {self.checkpoint_path}. Train ml/train_segmentation.py first."
            return
        try:
            import torch
            from monai.networks.nets import UNet
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet(
                spatial_dims=3, in_channels=1, out_channels=2,
                channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
            )
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            self.model = model.to(self.device).eval()
            self.load_error = None
        except Exception as exc:
            self.load_error = f"Could not load the MONAI segmentation model: {exc}"
            logger.exception(self.load_error)

    def segment_nifti(self, content: bytes, filename: str) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError(self.load_error or "3D segmentation model is not ready.")
        if not (filename.lower().endswith(".nii") or filename.lower().endswith(".nii.gz")):
            raise ValueError("Upload a NIfTI CT volume (.nii or .nii.gz).")
        try:
            from monai.inferers import sliding_window_inference
            from monai.transforms import (
                EnsureChannelFirst, LoadImage, Orientation, ScaleIntensityRanged, Spacing,
            )
        except ImportError as exc:
            raise RuntimeError("MONAI and nibabel are required for 3D segmentation.") from exc

        suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            volume = LoadImage(image_only=True)(temp_file.name)

        volume = EnsureChannelFirst(channel_dim="no_channel")(volume)
        volume = Orientation(axcodes="RAS")(volume)
        volume = Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear")(volume)
        volume = ScaleIntensityRanged(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True)(volume)
        torch = self.torch
        tensor = torch.as_tensor(np.asarray(volume).copy(), dtype=torch.float32).unsqueeze(0).to(self.device)
        if tensor.ndim != 5:
            raise ValueError(f"Expected a 3D CT volume, got tensor with shape {tuple(tensor.shape)}")

        roi_size = (96, 96, 96)
        with torch.no_grad():
            logits = sliding_window_inference(tensor, roi_size, sw_batch_size=1, predictor=self.model, overlap=0.25)
            mask = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        image_array = tensor[0, 0].detach().cpu().numpy()
        depth_index = image_array.shape[-1] // 2
        source_slice = image_array[:, :, depth_index]
        mask_slice = mask[:, :, depth_index]
        source_u8 = (np.clip(source_slice, 0, 1) * 255).astype(np.uint8)
        source_img = Image.fromarray(source_u8).convert("RGB")
        mask_rgb = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)
        mask_rgb[mask_slice > 0] = np.array([255, 70, 85], dtype=np.uint8)
        mask_img = Image.fromarray(mask_rgb, mode="RGB")
        overlay_img = Image.blend(source_img, mask_img, alpha=0.52)
        return {
            "original_slice": source_img,
            "mask_slice": mask_img,
            "overlay_slice": overlay_img,
            "mask_voxels": int(mask.sum()),
            "total_voxels": int(mask.size),
            "volume_shape": [int(v) for v in mask.shape],
            "slice_index": int(depth_index),
            "device": str(self.device),
            "model_name": "MONAI-3D-UNet-MSD-Lung",
            "notice": "Experimental segmentation mask for education only; not a diagnosis.",
        }
