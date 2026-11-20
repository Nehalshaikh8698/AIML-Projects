from io import BytesIO

import pytest
from PIL import Image

from app.services.image_service import image_from_bytes


def test_png_image_is_decoded():
    buffer = BytesIO()
    Image.new("L", (32, 24), color=128).save(buffer, format="PNG")
    result = image_from_bytes(buffer.getvalue(), "sample.png")
    assert result.mode == "RGB"
    assert result.size == (32, 24)


def test_unsupported_extension_is_rejected():
    with pytest.raises(ValueError, match="Unsupported file type"):
        image_from_bytes(b"not an image", "sample.exe")
