import os
from dataclasses import dataclass
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
                return (max(0, left - 10), top, min(w, right + 10), bottom)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return (0, 0, w, h)

    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area_ratio = (cw * ch) / float(w * h)
        if area_ratio > 0.25 and cw > w * 0.45 and ch > h * 0.25:
            candidates.append((x, y, cw, ch, area_ratio))
    if not candidates:
        return (0, 0, w, h)
    x, y, cw, ch, _ = max(candidates, key=lambda item: item[4])
    pad = 8
    return (max(0, x - pad), max(0, y - pad), min(w, x + cw + pad), min(h, y + ch + pad))


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


def prepare_cell_image(image):
    import cv2
    import numpy as np

    cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.45, blur, -0.45, 0)
    return Image.fromarray(sharp).convert("RGB")


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
    image = correct_perspective(load_for_table_detection(original_path))
    table_box = None
    if crop_box:
        x, y, w, h = crop_box
        image = image[y : y + h, x : x + w]
        table_box = (x, y, x + w, y + h)
    else:
        x1, y1, x2, y2 = detect_table_bounds(image)
        image = image[y1:y2, x1:x2]
        table_box = (x1, y1, x2, y2)

    x_lines, y_lines = detect_grid(image)
    h, w = image.shape[:2]
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
                image=prepare_cell_image(pil_from_cv(crop)),
                box=(left + pad_x, top + pad_y, right - pad_x, bottom - pad_y),
            )
        if {"date", "name", "am_in", "am_out", "pm_in", "pm_out"}.issubset(cells):
            rows.append(DetectedRow(index=index, box=(0, top, w, bottom), cells=cells))

    if not rows:
        raise TableDetectionError("Your image was uploaded successfully, but no logbook rows were detected.")
    debug = {
        "image_dimensions": {"width": w, "height": h},
        "table_box": table_box,
        "detected_table_dimensions": {"width": w, "height": h},
        "row_lines": row_lines,
        "column_lines": columns,
        "detected_rows": [{"index": row.index, "box": row.box} for row in rows],
        "debug_table_image": save_debug_image(original_path, image, "detected_table.png"),
    }
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


def recognize_name(row, recognizer, person):
    result = recognizer.read(row.cells["name"].image)
    return {
        "index": row.index,
        "name_text": result.text,
        "name_similarity": similarity(result.text, person.name),
        "confidence": result.confidence,
        "box": row.box,
    }


def recognize_row(row, recognizer, person, name_reading=None):
    readings = {}
    confidences = []
    if name_reading:
        readings["name"] = name_reading["name_text"]
        confidences.append(name_reading["confidence"])
    for field in ("date", "am_in", "am_out", "pm_in", "pm_out"):
        result = recognizer.read(row.cells[field].image)
        readings[field] = result.text
        confidences.append(result.confidence)
    if "name" not in readings:
        result = recognizer.read(row.cells["name"].image)
        readings["name"] = result.text
        confidences.append(result.confidence)

    parsed_date = parse_date(readings["date"])
    time_in_1 = parse_logbook_time(readings["am_in"], period="am")
    time_out_1 = parse_logbook_time(readings["am_out"], period="am")
    time_in_2 = parse_logbook_time(readings["pm_in"], period="pm")
    time_out_2 = parse_logbook_time(readings["pm_out"], period="pm")
    total = calculate_total_minutes(time_in_1, time_out_1, time_in_2, time_out_2)
    warnings = []
    if not parsed_date:
        warnings.append("Date needs review.")
    if not (time_in_1 and time_out_1):
        warnings.append("AM time needs review.")
    if not (time_in_2 and time_out_2):
        warnings.append("PM time needs review.")

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
        name_similarity=similarity(readings["name"], person.name),
        ocr_confidence=round(sum(confidences) / max(len(confidences), 1), 2),
        warnings=warnings,
        box=row.box,
    )


def scan_handwritten_logbook(original_path, person, crop_box=None):
    rows, debug = detect_rows_and_cells(original_path, crop_box=crop_box)
    try:
        recognizer = TrOCRHandwritingRecognizer()
    except HandwritingOCRUnavailable:
        raise

    name_readings = [recognize_name(row, recognizer, person) for row in rows]
    name_readings.sort(key=lambda item: item["name_similarity"], reverse=True)
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

    candidate_indexes = {item["index"] for item in name_readings[:5]}
    selected_name = name_readings[0] if name_readings else None
    if selected_name and selected_name["name_similarity"] >= HIGH_NAME_CONFIDENCE:
        candidate_indexes.add(selected_name["index"])

    row_lookup = {row.index: row for row in rows}
    recognized = []
    for reading in name_readings:
        if reading["index"] not in candidate_indexes:
            recognized.append(blank_row_from_name(row_lookup[reading["index"]], person, reading))
            continue
        recognized.append(recognize_row(row_lookup[reading["index"]], recognizer, person, name_reading=reading))
    recognized.sort(key=lambda item: item.name_similarity, reverse=True)
    selected = recognized[0] if recognized else None
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
        warnings=["Select this row to read its time cells."],
        box=row.box,
    )


def scan_row_by_index(original_path, person, row_index, crop_box=None):
    rows, debug = detect_rows_and_cells(original_path, crop_box=crop_box)
    row_lookup = {row.index: row for row in rows}
    if row_index not in row_lookup:
        raise TableDetectionError("The selected row could not be found in this image.")
    recognizer = TrOCRHandwritingRecognizer()
    recognized = recognize_row(row_lookup[row_index], recognizer, person)
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
    return TableScanResult(
        selected=recognized,
        rows=[recognized],
        warnings=recognized.warnings,
        confidence=round((recognized.name_similarity * 0.65) + (recognized.ocr_confidence * 0.35), 2),
        debug=debug,
    )
