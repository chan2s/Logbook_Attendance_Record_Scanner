import re
from dataclasses import dataclass
from datetime import date, datetime, time

from django.utils import timezone

from .attendance import calculate_total_minutes


MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

DATE_PATTERNS = [
    re.compile(r"\b(?P<month>[A-Za-z]{3,9})\.?\s+(?P<day>\d{1,2})(?:,\s*(?P<year>\d{4}))?\b"),
    re.compile(r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})(?:[/-](?P<year>\d{2,4}))?\b"),
]
TIME_RE = re.compile(r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>[AaPp]\.?[Mm]\.?)?\b")


@dataclass
class ParsedAttendance:
    date: date | None
    time_in_1: time | None
    time_out_1: time | None
    time_in_2: time | None
    time_out_2: time | None
    total_minutes: int
    confidence: float
    warnings: list[str]
    source_line: str


def parse_date(text, default_year=None):
    default_year = default_year or timezone.localdate().year
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw_month = match.group("month").lower()
        month = MONTHS.get(raw_month)
        if month is None and raw_month.isdigit():
            month = int(raw_month)
        day = int(match.group("day"))
        year_text = match.group("year")
        year = int(year_text) if year_text else default_year
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def parse_time(text):
    cleaned = text.strip().replace(".", "")
    for fmt in ("%I:%M %p", "%I %p", "%H:%M"):
        try:
            return datetime.strptime(cleaned.upper(), fmt).time()
        except ValueError:
            pass
    return None


def parse_logbook_time(text, period=None):
    cleaned = re.sub(r"[^0-9:apmAPM]+", " ", text).strip()
    match = TIME_RE.search(cleaned)
    if not match:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or "").lower().replace(".", "")
    if hour > 23 or minute > 59:
        return None

    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif not ampm and period == "pm" and 1 <= hour <= 7:
        hour += 12
    elif not ampm and period == "am" and hour == 12:
        hour = 12

    try:
        return time(hour, minute)
    except ValueError:
        return None


def extract_times(text):
    results = []
    for match in TIME_RE.finditer(text):
        token = match.group(0)
        parsed = parse_time(token)
        if parsed and parsed not in results:
            results.append(parsed)
    return results[:4]


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def find_matching_lines(text, person):
    student_id = normalize(person.student_id)
    name = normalize(person.name)
    name_parts = [part for part in name.split() if len(part) > 1]
    matches = []
    for line in text.splitlines():
        normalized = normalize(line)
        if not normalized:
            continue
        id_match = student_id and student_id in normalized
        name_score = sum(1 for part in name_parts if part in normalized)
        if id_match or name_score >= max(1, min(2, len(name_parts))):
            matches.append((line.strip(), id_match, name_score))
    return matches


def parse_attendance_from_text(text, person):
    warnings = []
    matches = find_matching_lines(text, person)
    if not text.strip():
        warnings.append("OCR could not read text from the image.")
    if not matches:
        warnings.append("Person not found in OCR text.")
        source_line = ""
    else:
        source_line = matches[0][0]
        if len(matches) > 1:
            warnings.append("Multiple matching rows were found. Please verify the selected row.")

    search_text = source_line or text
    parsed_date = parse_date(search_text) or parse_date(text)
    if not parsed_date:
        warnings.append("Date could not be detected or is invalid.")

    times = extract_times(search_text)
    if len(times) < 2:
        times = extract_times(text)
    if len(times) < 2:
        warnings.append("Time In and Time Out could not be confidently detected.")
    if len(times) % 2 == 1:
        warnings.append("A Time Out value appears to be missing.")

    time_in_1 = times[0] if len(times) > 0 else None
    time_out_1 = times[1] if len(times) > 1 else None
    time_in_2 = times[2] if len(times) > 2 else None
    time_out_2 = times[3] if len(times) > 3 else None
    total = calculate_total_minutes(time_in_1, time_out_1, time_in_2, time_out_2)

    confidence = 0.95
    if warnings:
        confidence -= min(0.65, len(warnings) * 0.18)
    if not matches:
        confidence = min(confidence, 0.35)
    if total == 0:
        confidence = min(confidence, 0.45)

    return ParsedAttendance(
        date=parsed_date,
        time_in_1=time_in_1,
        time_out_1=time_out_1,
        time_in_2=time_in_2,
        time_out_2=time_out_2,
        total_minutes=total,
        confidence=round(max(0, confidence), 2),
        warnings=warnings,
        source_line=source_line,
    )
