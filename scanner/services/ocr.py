from pathlib import Path
from io import BytesIO

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils.text import get_valid_filename

from .time_parser import parse_attendance_from_text


class OCRUnavailable(Exception):
    pass


def image_to_text(image_path):
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise OCRUnavailable(
            "OCR dependencies are not installed. Install Pillow, pytesseract, and the Tesseract OCR engine."
        ) from exc

    processed_path = preprocess_image(image_path)
    full_path = Path(default_storage.path(processed_path))
    try:
        with Image.open(full_path) as image:
            return pytesseract.image_to_string(image)
    except Exception as exc:
        raise OCRUnavailable("OCR could not process this image. Please try a clearer image.") from exc


def process_logbook_image(image_path, person):
    text = image_to_text(image_path)
    parsed = parse_attendance_from_text(text, person)
    return parsed, text


def preprocess_image(image_path):
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError as exc:
        raise OCRUnavailable("Image preprocessing requires Pillow.") from exc

    source_path = Path(default_storage.path(image_path))
    processed_name = f"logbook_scans/processed/{get_valid_filename(source_path.stem)}_ocr.png"

    try:
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("L")

            width, height = image.size
            shortest = min(width, height)
            if shortest < 1200:
                scale = 1200 / max(shortest, 1)
                image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

            image = ImageOps.autocontrast(image)
            image = ImageEnhance.Contrast(image).enhance(1.6)
            image = image.filter(ImageFilter.MedianFilter(size=3))
            image = image.filter(ImageFilter.SHARPEN)

            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            processed_name = default_storage.save(processed_name, ContentFile(buffer.getvalue()))
    except Exception as exc:
        raise OCRUnavailable("Image preprocessing failed. Please try a clearer image.") from exc

    return processed_name
