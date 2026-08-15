import os
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from PIL import Image

from .attendance import calculate_total_minutes
from .handwriting import HandwritingOCRUnavailable, TrOCRHandwritingRecognizer
from .time_parser import normalize, parse_date, parse_logbook_time


class TableDetectionError(Exception):
    pass


FIELD_COLUMNS = {
    "date": 0,
    "name": 1,
    "am_in": 2,
    "am_out": 4,
    "pm_in": 6,
    "pm_out": 8,
}

HIGH_NAME_CONFIDENCE = 0.82
MEDIUM_NAME_CONFIDENCE = 0.60
MAX_NAME_CANDIDATES = 3
NAME_MAX_NEW_TOKENS = 16
TIME_MAX_NEW_TOKENS = 8
DATE_MAX_NEW_TOKENS = 16

# Content-based cell extraction tuning (handwritten logbooks often have
# faint or missing vertical grid lines, so time cells are located from the
# ink structure of the matched row instead of relying on grid columns).
SIGNATURE_MIN_WIDTH = 60
GROUP_GAP = 30
TIME_GROUP_MIN_WIDTH = 20
TIME_GROUP_MAX_WIDTH = 170
NAME_GROUP_MIN_WIDTH = 140
NAME_SKIP_LEFT_FRACTION = 0.28
TIME_CROP_LEFT_FRACTION = 0.68
TIME_TARGET_HEIGHT = 128
NAME_TARGET_HEIGHT = 96

TIME_FIELDS = ("am_in", "am_out", "pm_in", "pm_out")
TIME_PERIODS = {"am_in": "am", "am_out": "am", "pm_in": "pm", "pm_out": "pm"}


@dataclass
class CellCrop:
    field: str
    image: Image.Image
    box: tuple[int, int, int, int]


@dataclass
class DetectedRow:
    index: int
    box: tuple[int, int, int, int]
    cells: dict[str, CellCrop]
    image: Image.Image = None
    table_origin: tuple[int, int] = (0, 0)
    full_size: tuple[int, int] = (0, 0)


@dataclass
class RecognizedRow:
    index: int
    date_text: str
    name_text: str
    am_in_text: str
    am_out_text: str
    pm_in_text: str
    pm_out_text: str
    date: object
    time_in_1: object
    time_out_1: object
    time_in_2: object
    time_out_2: object
    total_minutes: int
    name_similarity: float
    ocr_confidence: float
    field_confidences: dict[str, float]
    needs_review: list[str]
    warnings: list[str]
    box: tuple[int, int, int, int]


@dataclass
class TableScanResult:
    selected: RecognizedRow | None
    rows: list[RecognizedRow]
    warnings: list[str]
    confidence: float
    debug: dict


def pil_from_cv(image):
    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_for_table_detection(path):
    try:
        import cv2
    except ImportError as exc:
        raise TableDetectionError("OpenCV is required for table and row detection.") from exc

    image = cv2.imread(str(path))
    if image is None:
        raise TableDetectionError("The uploaded image could not be opened.")
    image = resize_if_needed(image)
    return image


def resize_if_needed(image):
    import cv2

    h, w = image.shape[:2]
    shortest = min(h, w)
    if shortest >= 1200:
        return image
    scale = 1200 / max(shortest, 1)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def correct_perspective(image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        area = cv2.contourArea(approx)
        if len(approx) != 4 or area < w * h * 0.25:
            continue
        points = approx.reshape(4, 2).astype("float32")
        ordered = order_points(points)
        target_w = int(max(
            np.linalg.norm(ordered[2] - ordered[3]),
            np.linalg.norm(ordered[1] - ordered[0]),
        ))
        target_h = int(max(
            np.linalg.norm(ordered[1] - ordered[2]),
            np.linalg.norm(ordered[0] - ordered[3]),
        ))
        if target_w < 400 or target_h < 400:
            continue
        destination = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(ordered, destination)
        return cv2.warpPerspective(image, matrix, (target_w, target_h))
    return image


def order_points(points):
    import numpy as np

    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def detect_table_bounds(image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8)
    h, w = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, w // 45), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    line_runs = []
    for y in range(h):
        xs = np.where(horizontal[y] > 0)[0]
        if xs.size >= w * 0.08:
            line_runs.append((y, int(xs.min()), int(xs.max()), int(xs.size)))
    groups = group_line_runs(line_runs, axis_index=0, max_gap=10)
    if len(groups) >= 3:
        candidates = []
        for group in groups:
            y = int(sum(item[0] for item in group) / len(group))
            x1 = min(item[1] for item in group)
            x2 = max(item[2] for item in group)
            length = max(item[3] for item in group)
            if length >= w * 0.18:
                candidates.append((y, x1, x2, length))
        if len(candidates) >= 3:
            ys = [item[0] for item in candidates]
            left = min(item[1] for item in candidates)
            right = max(item[2] for item in candidates)
            top = max(0, min(ys) - 12)
            bottom = min(h, max(ys) + int(median_spacing(ys) * 1.5))
            if max(ys) > h * 0.45:
                bottom = h
            if right - left > w * 0.45 and bottom - top > h * 0.18:
                return (max(0, int(left) - 10), int(top), min(w, int(right) + 10), int(bottom))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return (0, 0, int(w), int(h))

    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area_ratio = (cw * ch) / float(w * h)
        if area_ratio > 0.25 and cw > w * 0.45 and ch > h * 0.25:
            candidates.append((int(x), int(y), int(cw), int(ch), area_ratio))
    if not candidates:
        return (0, 0, int(w), int(h))
    x, y, cw, ch, _ = max(candidates, key=lambda item: item[4])
    pad = 8
    return (max(0, x - pad), max(0, y - pad), min(int(w), x + cw + pad), min(int(h), y + ch + pad))


def group_line_runs(items, axis_index=0, max_gap=10):
    groups = []
    for item in items:
        value = item[axis_index]
        if not groups or value - groups[-1][-1][axis_index] > max_gap:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def median_spacing(values):
    import statistics

    ordered = sorted(values)
    gaps = [right - left for left, right in zip(ordered, ordered[1:]) if 20 <= right - left <= 180]
    if not gaps:
        return 56
    return statistics.median(gaps)


def line_positions(projection, min_gap=12, threshold_ratio=0.42):
    if projection.size == 0:
        return []
    threshold = projection.max() * threshold_ratio
    indexes = [idx for idx, value in enumerate(projection) if value >= threshold]
    groups = []
    for idx in indexes:
        if not groups or idx - groups[-1][-1] > min_gap:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return [int(sum(group) / len(group)) for group in groups]


def detect_grid(table_image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8)
    h, w = binary.shape

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(70, w // 45), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(55, h // 24)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    y_projection = horizontal.sum(axis=1) / 255
    x_projection = vertical.sum(axis=0) / 255
    y_lines = line_positions(y_projection, min_gap=10, threshold_ratio=0.08)
    x_lines = line_positions(x_projection, min_gap=10, threshold_ratio=0.20)
    return x_lines, y_lines


def fallback_columns(width):
    ratios = [0.0, 0.12, 0.34, 0.41, 0.49, 0.56, 0.65, 0.72, 0.82, 0.89, 1.0]
    return [int(width * ratio) for ratio in ratios]


def normalized_columns(x_lines, width):
    if len(x_lines) >= 10:
        columns = sorted(x_lines)
        if columns[0] > 8:
            columns.insert(0, 0)
        if columns[-1] < width - 8:
            columns.append(width)
        if len(columns) >= 11:
            return columns[:11]
    return fallback_columns(width)


def normalized_rows(y_lines, height):
    rows = sorted(value for value in y_lines if 0 <= value <= height)
    if len(rows) < 3:
        row_height = max(42, height // 12)
        return list(range(0, height + row_height, row_height))
    if rows[0] > 8:
        rows.insert(0, 0)
    spacing = median_spacing(rows)
    expanded = rows[:]
    while expanded[-1] + spacing <= height + 8:
        expanded.append(int(expanded[-1] + spacing))
    if expanded[-1] < height - 8:
        expanded.append(height)
    return dedupe_lines(expanded, min_gap=max(18, int(spacing * 0.45)))


def dedupe_lines(lines, min_gap=18):
    result = []
    for line in sorted(lines):
        if not result or line - result[-1] >= min_gap:
            result.append(int(line))
        else:
            result[-1] = int((result[-1] + line) / 2)
    return result


def prepare_cell_image(image, target_height=TIME_TARGET_HEIGHT):
    """Preprocess a handwriting cell for TrOCR.

    Crops are small, so they are upscaled with LANCZOS and binarized with
    Otsu. Binarization is applied only after upscaling so faint handwriting
    strokes (common for tiny digit entries) are preserved and clarified.
    """
    import cv2
    import numpy as np

    cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    if gray.size == 0 or gray.shape[0] < 4 or gray.shape[1] < 4:
        return image
    scale = target_height / gray.shape[0]
    new_w = max(1, int(gray.shape[1] * scale))
    up = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
    _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary).convert("RGB")


def save_debug_image(path, image, name):
    try:
        from django.conf import settings

        if not getattr(settings, "DEBUG", False):
            return ""
        debug_dir = os.path.join(settings.MEDIA_ROOT, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        target = os.path.join(debug_dir, name)
        if hasattr(image, "save"):
            image.save(target)
        else:
            import cv2

            cv2.imwrite(target, image)
        return os.path.relpath(target, settings.MEDIA_ROOT).replace("\\", "/")
    except Exception:
        return ""


def detect_rows_and_cells(original_path, crop_box=None):
    print("[OCR] START image loading")
    start = time.perf_counter()
    image = load_for_table_detection(original_path)
    print(f"[OCR] END image loading: {time.perf_counter() - start:.2f}s")

    print("[OCR] START table detection")
    start = time.perf_counter()
    image = correct_perspective(image)
    full_h, full_w = (int(value) for value in image.shape[:2])
    print(f"[OCR] Full image size: {full_w}x{full_h}")
    table_box = None
    if crop_box:
        x, y, w, h = crop_box
        image = image[y : y + h, x : x + w]
        table_box = (x, y, x + w, y + h)
    else:
        x1, y1, x2, y2 = detect_table_bounds(image)
        image = image[y1:y2, x1:x2]
        table_box = (x1, y1, x2, y2)
    table_origin = (table_box[0], table_box[1])

    x_lines, y_lines = detect_grid(image)
    print(f"[OCR] END table detection: {time.perf_counter() - start:.2f}s")

    print("[OCR] START row detection")
    start = time.perf_counter()
    h, w = (int(value) for value in image.shape[:2])
    columns = normalized_columns(x_lines, w)
    row_lines = normalized_rows(y_lines, h)
    rows = []

    for index, (top, bottom) in enumerate(zip(row_lines, row_lines[1:])):
        row_height = bottom - top
        if row_height < 34:
            continue
        if index == 0 or top < 55:
            continue
        cells = {}
        for field, column_index in FIELD_COLUMNS.items():
            left = columns[column_index]
            right = columns[column_index + 1]
            pad_ratio = 0.10 if field in {"am_in", "am_out", "pm_in", "pm_out"} else 0.04
            pad_x = max(2, int((right - left) * pad_ratio))
            pad_y = max(2, int(row_height * 0.12))
            crop = image[top + pad_y : bottom - pad_y, left + pad_x : right - pad_x]
            if crop.size == 0:
                continue
            cells[field] = CellCrop(
                field=field,
                image=pil_from_cv(crop),
                box=(left + pad_x, top + pad_y, right - pad_x, bottom - pad_y),
            )
        if {"date", "name", "am_in", "am_out", "pm_in", "pm_out"}.issubset(cells):
            rows.append(DetectedRow(
                index=index,
                box=(0, top, w, bottom),
                cells=cells,
                image=pil_from_cv(image[top:bottom, :, :]),
                table_origin=table_origin,
                full_size=(full_w, full_h),
            ))

    if not rows:
        raise TableDetectionError("Your image was uploaded successfully, but no logbook rows were detected.")
    debug = {
        "image_dimensions": {"width": w, "height": h},
        "full_image_dimensions": {"width": full_w, "height": full_h},
        "table_box": table_box,
        "table_origin": table_origin,
        "detected_table_dimensions": {"width": w, "height": h},
        "row_lines": row_lines,
        "column_lines": columns,
        "detected_rows": [{"index": row.index, "box": row.box} for row in rows],
        "debug_table_image": save_debug_image(original_path, image, "detected_table.png"),
    }
    print(f"[OCR] END row detection: {time.perf_counter() - start:.2f}s")
    return rows, debug


def similarity(a, b):
    left = normalize(a)
    right = normalize(b)
    if not left or not right:
        return 0.0
    direct = SequenceMatcher(None, left, right).ratio()
    left_tokens = " ".join(sorted(left.split()))
    right_tokens = " ".join(sorted(right.split()))
    token_score = SequenceMatcher(None, left_tokens, right_tokens).ratio()
    left_parts = left.split()
    right_parts = right.split()
    best_token_scores = []
    for target in right_parts:
        if len(target) <= 1:
            continue
        best_token_scores.append(max(SequenceMatcher(None, target, seen).ratio() for seen in left_parts))
    coverage_score = sum(best_token_scores) / len(best_token_scores) if best_token_scores else 0.0
    return round(max(direct, token_score, coverage_score), 2)


# ---------------------------------------------------------------------------
# Content-based row cell extraction
# ---------------------------------------------------------------------------

def row_content_groups(row_image):
    """Return the non-signature ink groups of a row, sorted left to right.

    Signature scribbles are wide/dense connected components; they are dropped
    so that time digits (and the name) remain as isolated groups. Grouping
    merges nearby blobs that belong to the same value.
    """
    import cv2
    import numpy as np

    gray = np.array(row_image.convert("L"))
    row_h, row_w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if ww < 2 or hh < 2 or area < 4:
            continue
        is_signature = ww >= SIGNATURE_MIN_WIDTH or (hh >= row_h * 0.9 and ww >= 30 and area > 200)
        if is_signature:
            continue
        blobs.append((x, y, ww, hh))
    blobs.sort(key=lambda b: b[0])
    groups = []
    for blob in blobs:
        x0, x1 = blob[0], blob[0] + blob[2]
        if not groups or x0 - groups[-1]["x1"] > GROUP_GAP:
            groups.append({"x0": x0, "x1": x1, "y0": blob[1], "y1": blob[1] + blob[3]})
        else:
            last = groups[-1]
            last["x1"] = max(last["x1"], x1)
            last["y0"] = min(last["y0"], blob[1])
            last["y1"] = max(last["y1"], blob[1] + blob[3])
    for group in groups:
        group["width"] = group["x1"] - group["x0"]
        group["height"] = group["y1"] - group["y0"]
    return groups


def name_group_from(groups):
    """The name is the widest non-signature group on the row."""
    if not groups:
        return None
    return max(groups, key=lambda g: g["width"])


def crop_group_portion(row_image, group, left_fraction=1.0, right_fraction=1.0, pad_x=4, pad_y=3):
    """Crop a horizontal portion of a group's bounding box from the row image.

    left_fraction: keep [x0, x0 + width*left_fraction)  (times are left-aligned)
    right_fraction: keep [x1 - width*right_fraction, x1)
    """
    x0 = group["x0"]
    width = group["width"]
    if left_fraction < 1.0:
        x1 = x0 + int(width * left_fraction)
    else:
        x1 = group["x1"]
    if right_fraction < 1.0:
        x0 = max(x0, x1 - int(width * right_fraction))
    y0 = max(0, group["y0"] - pad_y)
    y1 = min(row_image.size[1], group["y1"] + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(row_image.size[0], x1 + pad_x)
    box = (int(x0), int(y0), int(x1), int(y1))
    return row_image.crop(box), box


def extract_name_crop(row, recognizer):
    """OCR the name cell of a row using content-based extraction."""
    groups = row_content_groups(row.image)
    name_group = name_group_from(groups)
    if name_group is None:
        return "", 0.0, None
    crop, _ = crop_group_portion(row.image, name_group, right_fraction=1.0 - NAME_SKIP_LEFT_FRACTION)
    return crop, name_group["width"], groups


def recognize_name(row, recognizer, person):
    crop, _, _ = extract_name_crop(row, recognizer)
    result = recognizer.read(
        prepare_cell_image(crop, target_height=NAME_TARGET_HEIGHT),
        max_new_tokens=NAME_MAX_NEW_TOKENS,
        cache_key=f"name:{row.index}:{row.box}",
    )
    print(f"[OCR] NAME RAW: {result.text!r} (conf {result.confidence})")
    return {
        "index": row.index,
        "name_text": result.text,
        "name_similarity": similarity(result.text, person.name),
        "confidence": result.confidence,
        "box": row.box,
    }


def name_candidate_rows(rows, limit=MAX_NAME_CANDIDATES):
    if not rows:
        return []
    table_top = min(row.box[1] for row in rows)
    table_bottom = max(row.box[3] for row in rows)
    target_y = table_top + ((table_bottom - table_top) * 0.56)
    return sorted(
        rows,
        key=lambda row: abs(((row.box[1] + row.box[3]) / 2) - target_y),
    )[:limit]


def time_groups_for_row(row):
    """Return the four time-value groups (am_in, am_out, pm_in, pm_out)."""
    groups = row_content_groups(row.image)
    name_group = name_group_from(groups)
    if name_group is None:
        return []
    time_groups = [
        group for group in groups
        if group["x0"] >= name_group["x0"] + 1
        and TIME_GROUP_MIN_WIDTH <= group["width"] <= TIME_GROUP_MAX_WIDTH
    ]
    return time_groups[:4]


def recognize_row(row, recognizer, person, name_reading=None):
    readings = {}
    confidences = {}
    needs_review = []
    warnings = []

    # --- Name ---
    if name_reading:
        readings["name"] = name_reading["name_text"]
        confidences["name"] = name_reading["confidence"]
    else:
        crop, _, _ = extract_name_crop(row, recognizer)
        result = recognizer.read(
            prepare_cell_image(crop, target_height=NAME_TARGET_HEIGHT),
            max_new_tokens=NAME_MAX_NEW_TOKENS,
            cache_key=f"name:{row.index}:{row.box}",
        )
        readings["name"] = result.text
        confidences["name"] = result.confidence

    # --- Time cells (content-based, signature-free) ---
    # Coordinate systems: group/crop boxes are LOCAL to the row image (which
    # starts at row.box[1] within the detected table image). GLOBAL coordinates
    # add the row origin, and full-image coordinates add the table origin.
    time_groups = time_groups_for_row(row)
    row_origin = (row.box[0], row.box[1])
    print(f"[OCR] Full image size: {row.full_size[0]}x{row.full_size[1]}")
    print(f"[OCR] Matched row GLOBAL: x1={row.box[0]} y1={row.box[1]} x2={row.box[2]} y2={row.box[3]}")
    print(f"[OCR] Table origin (offset from full image): x={row.table_origin[0]} y={row.table_origin[1]}")
    row_gray = row.image

    def to_global(local_box):
        return (
            row_origin[0] + local_box[0],
            row_origin[1] + local_box[1],
            row_origin[0] + local_box[2],
            row_origin[1] + local_box[3],
        )

    batch_items = []
    field_boxes = {}
    for field, group in zip(TIME_FIELDS, time_groups):
        crop, box = crop_group_portion(row_gray, group, left_fraction=TIME_CROP_LEFT_FRACTION)
        field_boxes[field] = box
        gbox = to_global(box)
        print(f"[OCR] {field.upper()} LOCAL: x1={box[0]} y1={box[1]} x2={box[2]} y2={box[3]}")
        print(f"[OCR] {field.upper()} GLOBAL: x1={gbox[0]} y1={gbox[1]} x2={gbox[2]} y2={gbox[3]}")
        batch_items.append((f"{field}:{row.index}:{box}", prepare_cell_image(crop), TIME_MAX_NEW_TOKENS))

    if batch_items:
        batch_results = recognizer.read_batch(batch_items)
        for field, box in field_boxes.items():
            key = f"{field}:{row.index}:{box}"
            readings[field] = batch_results[key].text
            confidences[field] = batch_results[key].confidence
    else:
        for field in TIME_FIELDS:
            readings[field] = ""
            confidences[field] = 0.0
    for field in TIME_FIELDS:
        print(f"[OCR] {field.upper()} RAW: {readings[field]!r} (conf {confidences[field]})")

    # --- Date (attempted; never invented when unreadable) ---
    groups = row_content_groups(row_gray)
    name_group = name_group_from(groups)
    date_crop = None
    date_box = None
    if name_group is not None:
        if name_group["x0"] > 8:
            date_region = {"x0": 0, "x1": name_group["x0"], "y0": name_group["y0"], "y1": name_group["y1"], "width": name_group["x0"]}
            date_crop, date_box = crop_group_portion(row_gray, date_region, pad_x=2, pad_y=3)
        else:
            date_crop, date_box = crop_group_portion(row_gray, name_group, left_fraction=NAME_SKIP_LEFT_FRACTION, pad_x=2)
    if date_crop is not None and date_crop.size[0] > 4 and date_crop.size[1] > 4:
        result = recognizer.read(
            prepare_cell_image(date_crop, target_height=NAME_TARGET_HEIGHT),
            max_new_tokens=DATE_MAX_NEW_TOKENS,
            cache_key=f"date:{row.index}:{date_box}",
        )
        readings["date"] = result.text
        confidences["date"] = result.confidence
    else:
        readings["date"] = ""
        confidences["date"] = 0.0

    # --- Parse + per-field validation ---
    parsed_date = parse_date(readings["date"])
    time_in_1 = parse_logbook_time(readings["am_in"], period="am")
    time_out_1 = parse_logbook_time(readings["am_out"], period="am")
    time_in_2 = parse_logbook_time(readings["pm_in"], period="pm")
    time_out_2 = parse_logbook_time(readings["pm_out"], period="pm")

    print(f"[OCR] NAME NORMALIZED: {readings['name']!r}")
    print(f"[OCR] DATE RAW: {readings['date']!r} -> NORMALIZED: {parsed_date.isoformat() if parsed_date else None}")
    for field, value in (("am_in", time_in_1), ("am_out", time_out_1), ("pm_in", time_in_2), ("pm_out", time_out_2)):
        print(f"[OCR] {field.upper()} NORMALIZED: {value.strftime('%H:%M') if value else None} (raw {readings[field]!r})")

    if not parsed_date:
        confidences["date"] = min(confidences["date"], 0.2)
        needs_review.append("date")
        warnings.append("Date needs review.")
    for field, value in (("am_in", time_in_1), ("am_out", time_out_1), ("pm_in", time_in_2), ("pm_out", time_out_2)):
        if value is None:
            confidences[field] = min(confidences[field], 0.2)
            needs_review.append(field)
    if not (time_in_1 and time_out_1):
        warnings.append("AM time needs review.")
    if not (time_in_2 and time_out_2):
        warnings.append("PM time needs review.")

    total = calculate_total_minutes(time_in_1, time_out_1, time_in_2, time_out_2)
    name_sim = similarity(readings["name"], person.name)

    return RecognizedRow(
        index=row.index,
        date_text=readings["date"],
        name_text=readings["name"],
        am_in_text=readings["am_in"],
        am_out_text=readings["am_out"],
        pm_in_text=readings["pm_in"],
        pm_out_text=readings["pm_out"],
        date=parsed_date,
        time_in_1=time_in_1,
        time_out_1=time_out_1,
        time_in_2=time_in_2,
        time_out_2=time_out_2,
        total_minutes=total,
        name_similarity=name_sim,
        ocr_confidence=round(sum(confidences.values()) / max(len(confidences), 1), 2),
        field_confidences=confidences,
        needs_review=needs_review,
        warnings=warnings,
        box=row.box,
    )


def save_row_debug_images(row, recognized, debug):
    """Save debug crops of the matched row for visual inspection.

    Files are written with fixed names (matched_row.jpg, name_crop.jpg,
    am_in.jpg, ...) so they can be inspected after a scan. The coordinates
    in debug["debug_row_crops"] are LOCAL to the row image.
    """
    import cv2
    import numpy as np

    debug["debug_row_crops"] = {}
    try:
        from django.conf import settings

        if not getattr(settings, "DEBUG", False):
            return
        debug_dir = os.path.join(settings.MEDIA_ROOT, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        row_image = row.image
        matched = cv2.cvtColor(np.array(row_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.line(matched, (0, 4), (row_image.size[0], 4), (0, 0, 255), 3)
        cv2.line(matched, (0, row_image.size[1] - 4), (row_image.size[0], row_image.size[1] - 4), (0, 0, 255), 3)
        debug["debug_row_crops"]["matched_row"] = save_debug_image("", matched, "matched_row.jpg")

        groups = row_content_groups(row_image)
        name_group = name_group_from(groups)
        name_crop, name_box = crop_group_portion(row_image, name_group, right_fraction=1.0 - NAME_SKIP_LEFT_FRACTION)
        debug["debug_row_crops"]["name"] = save_debug_image("", name_crop, "name_crop.jpg")
        debug["debug_row_crops"]["name_box"] = list(name_box)

        for field, group in zip(TIME_FIELDS, time_groups_for_row(row)):
            crop, box = crop_group_portion(row_image, group, left_fraction=TIME_CROP_LEFT_FRACTION)
            debug["debug_row_crops"][field] = save_debug_image("", crop, f"{field}.jpg")
            debug["debug_row_crops"][f"{field}_box"] = list(box)
        if name_group is not None:
            if name_group["x0"] > 8:
                date_region = {"x0": 0, "x1": name_group["x0"], "y0": name_group["y0"], "y1": name_group["y1"], "width": name_group["x0"]}
                date_crop, _ = crop_group_portion(row_image, date_region, pad_x=2, pad_y=3)
            else:
                date_crop, _ = crop_group_portion(row_image, name_group, left_fraction=NAME_SKIP_LEFT_FRACTION, pad_x=2)
            debug["debug_row_crops"]["date"] = save_debug_image("", date_crop, "date.jpg")
    except Exception:
        debug["debug_row_crops"] = {}


def scan_handwritten_logbook(original_path, person, crop_box=None):
    total_start = time.perf_counter()
    TrOCRHandwritingRecognizer.reset_counter()
    rows, debug = detect_rows_and_cells(original_path, crop_box=crop_box)
    try:
        recognizer = TrOCRHandwritingRecognizer()
    except HandwritingOCRUnavailable:
        raise

    print("[OCR] START name matching")
    start = time.perf_counter()
    name_readings = []
    selected_name = None
    candidates = name_candidate_rows(rows)
    for candidate_number, row in enumerate(candidates, start=1):
        print(f"[OCR] Name candidate {candidate_number}")
        reading = recognize_name(row, recognizer, person)
        name_readings.append(reading)
        if reading["name_similarity"] >= HIGH_NAME_CONFIDENCE:
            selected_name = reading
            break
    name_readings.sort(key=lambda item: item["name_similarity"], reverse=True)
    if selected_name is None and name_readings:
        selected_name = name_readings[0]
    print(f"[OCR] END name matching: {time.perf_counter() - start:.2f}s")
    debug["detected_names"] = [
        {
            "row_index": item["index"],
            "text": item["name_text"],
            "similarity": round(item["name_similarity"], 2),
            "confidence": item["confidence"],
            "box": item["box"],
        }
        for item in name_readings
    ]
    debug["name_matching"] = {
        "recognized_rows": len(name_readings),
        "detected_rows": len(rows),
        "max_name_candidates": MAX_NAME_CANDIDATES,
        "candidate_order": "geometry_lower_middle",
        "candidate_rows": [row.index for row in candidates],
        "reason": "Checked at most three geometry-ranked name cells and stopped after a high-confidence stored-name match.",
    }

    row_lookup = {row.index: row for row in rows}
    print("[OCR] START time extraction")
    start = time.perf_counter()
    recognized = [blank_row_from_name(row_lookup[reading["index"]], person, reading) for reading in name_readings[:5]]
    if selected_name and selected_name["name_similarity"] >= HIGH_NAME_CONFIDENCE:
        selected_row = recognize_row(row_lookup[selected_name["index"]], recognizer, person, name_reading=selected_name)
        recognized = [selected_row] + [
            item for item in recognized
            if item.index != selected_row.index
        ]
        save_row_debug_images(row_lookup[selected_name["index"]], selected_row, debug)
    recognized.sort(key=lambda item: item.name_similarity, reverse=True)
    selected = recognized[0] if recognized else None
    print(f"[OCR] END time extraction: {time.perf_counter() - start:.2f}s")

    print("[OCR] START result preparation")
    start = time.perf_counter()
    warnings = []
    if selected is None:
        warnings.append("Could not confidently identify your attendance record.")
    elif selected.name_similarity < HIGH_NAME_CONFIDENCE:
        warnings.append("Could not confidently identify your attendance record.")
    elif selected.warnings:
        warnings.extend(selected.warnings)

    confidence = 0.0
    if selected:
        confidence = round((selected.name_similarity * 0.65) + (selected.ocr_confidence * 0.35), 2)

    debug["selected_row"] = selected.index if selected else None
    debug["time_cells"] = {
        "date": selected.date_text if selected else "",
        "am_in": selected.am_in_text if selected else "",
        "am_out": selected.am_out_text if selected else "",
        "pm_in": selected.pm_in_text if selected else "",
        "pm_out": selected.pm_out_text if selected else "",
    }
    debug["parsed_times"] = {
        "time_in_1": str(selected.time_in_1) if selected and selected.time_in_1 else "",
        "time_out_1": str(selected.time_out_1) if selected and selected.time_out_1 else "",
        "time_in_2": str(selected.time_in_2) if selected and selected.time_in_2 else "",
        "time_out_2": str(selected.time_out_2) if selected and selected.time_out_2 else "",
    }
    debug["field_confidences"] = selected.field_confidences if selected else {}
    debug["needs_review"] = selected.needs_review if selected else []
    debug["trocr_inference_count"] = TrOCRHandwritingRecognizer.inference_count()
    print(f"[OCR] TOTAL TrOCR inference calls: {TrOCRHandwritingRecognizer.inference_count()}")
    print(f"[OCR] END result preparation: {time.perf_counter() - start:.2f}s")
    print(f"[OCR] Total OCR processing time: {time.perf_counter() - total_start:.2f}s")
    print("[OCR] Result prepared")

    return TableScanResult(
        selected=selected if selected and selected.name_similarity >= HIGH_NAME_CONFIDENCE else None,
        rows=recognized[:5],
        warnings=warnings,
        confidence=confidence,
        debug=debug,
    )


def blank_row_from_name(row, person, reading):
    return RecognizedRow(
        index=row.index,
        date_text="",
        name_text=reading["name_text"],
        am_in_text="",
        am_out_text="",
        pm_in_text="",
        pm_out_text="",
        date=None,
        time_in_1=None,
        time_out_1=None,
        time_in_2=None,
        time_out_2=None,
        total_minutes=0,
        name_similarity=reading["name_similarity"],
        ocr_confidence=reading["confidence"],
        field_confidences={"name": reading["confidence"]},
        needs_review=[],
        warnings=["Select this row to read its time cells."],
        box=row.box,
    )


def scan_row_by_index(original_path, person, row_index, crop_box=None):
    total_start = time.perf_counter()
    TrOCRHandwritingRecognizer.reset_counter()
    rows, debug = detect_rows_and_cells(original_path, crop_box=crop_box)
    row_lookup = {row.index: row for row in rows}
    if row_index not in row_lookup:
        raise TableDetectionError("The selected row could not be found in this image.")
    recognizer = TrOCRHandwritingRecognizer()
    print("[OCR] START time extraction")
    start = time.perf_counter()
    recognized = recognize_row(row_lookup[row_index], recognizer, person)
    print(f"[OCR] END time extraction: {time.perf_counter() - start:.2f}s")
    print("[OCR] START result preparation")
    start = time.perf_counter()
    debug["selected_row"] = recognized.index
    debug["time_cells"] = {
        "date": recognized.date_text,
        "am_in": recognized.am_in_text,
        "am_out": recognized.am_out_text,
        "pm_in": recognized.pm_in_text,
        "pm_out": recognized.pm_out_text,
    }
    debug["parsed_times"] = {
        "time_in_1": str(recognized.time_in_1) if recognized.time_in_1 else "",
        "time_out_1": str(recognized.time_out_1) if recognized.time_out_1 else "",
        "time_in_2": str(recognized.time_in_2) if recognized.time_in_2 else "",
        "time_out_2": str(recognized.time_out_2) if recognized.time_out_2 else "",
    }
    debug["field_confidences"] = recognized.field_confidences
    debug["needs_review"] = recognized.needs_review
    debug["trocr_inference_count"] = TrOCRHandwritingRecognizer.inference_count()
    print(f"[OCR] TOTAL TrOCR inference calls: {TrOCRHandwritingRecognizer.inference_count()}")
    print(f"[OCR] END result preparation: {time.perf_counter() - start:.2f}s")
    print(f"[OCR] Total OCR processing time: {time.perf_counter() - total_start:.2f}s")
    print("[OCR] Result prepared")
    return TableScanResult(
        selected=recognized,
        rows=[recognized],
        warnings=recognized.warnings,
        confidence=round((recognized.name_similarity * 0.65) + (recognized.ocr_confidence * 0.35), 2),
        debug=debug,
    )
