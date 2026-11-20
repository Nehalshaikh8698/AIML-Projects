from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


def image_from_bytes(content: bytes, filename: str) -> Image.Image:
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload PNG, JPG, JPEG, or DICOM.")

    if suffix in {".dcm", ".dicom"}:
        try:
            import pydicom
            try:
                from pydicom.pixels import apply_voi_lut
            except ImportError:
                from pydicom.pixel_data_handlers.util import apply_voi_lut
        except ImportError as exc:
            raise ValueError("DICOM support is unavailable. Install pydicom.") from exc
        try:
            ds = pydicom.dcmread(BytesIO(content), force=True)
            pixels = ds.pixel_array
            try:
                pixels = apply_voi_lut(pixels, ds)
            except Exception:
                pass
            pixels = pixels.astype(np.float32)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixels = pixels.max() - pixels
            lo, hi = np.percentile(pixels, [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            pixels = np.clip((pixels - lo) / (hi - lo), 0, 1)
            pixels = (pixels * 255).astype(np.uint8)
            if pixels.ndim == 3:
                # Multi-frame DICOM: use the first frame for this MVP.
                pixels = pixels[0]
            return Image.fromarray(pixels).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not decode DICOM pixels: {exc}") from exc

    try:
        image = Image.open(BytesIO(content))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        raise ValueError("The uploaded image could not be decoded.") from exc


def save_preview(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
