from datetime import datetime
import logging

from django.contrib import messages
from django.core.files.storage import default_storage
from django.db import IntegrityError, transaction
from django.db.models import Sum
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .forms import IdentifyForm, ReviewAttendanceForm, ScanUploadForm
from .models import AttendanceRecord, Person
from .services.attendance import calculate_total_minutes, format_minutes, progress_percent, record_time_range
from .services.handwriting import HandwritingOCRUnavailable
from .services.ocr import OCRUnavailable
from .services.table_ocr import TableDetectionError, scan_handwritten_logbook, scan_row_by_index


logger = logging.getLogger(__name__)


def session_key(student_id):
    return f"scan_session_{student_id}"


def pending_key(student_id):
    return f"pending_scan_{student_id}"


def serialize_time(value):
    return value.strftime("%H:%M") if value else ""


def serialize_date(value):
    return value.isoformat() if value else ""


def decimal_hours(minutes):
    return f"{minutes / 60:.2f}"


def decorate_record(record):
    item = record.copy()
    item["total_display"] = format_minutes(item.get("total_minutes", 0))
    item["morning"] = time_range_text(item.get("time_in_1"), item.get("time_out_1"))
    item["afternoon"] = time_range_text(item.get("time_in_2"), item.get("time_out_2"))
    return item


def time_range_text(start, end):
    if not start or not end:
        return ""
    return f"{display_time(start)} -> {display_time(end)}"


def display_time(value):
    if not value:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%I:%M %p").lstrip("0")
    parsed = datetime.strptime(value, "%H:%M").time()
    return parsed.strftime("%I:%M %p").lstrip("0")


def get_session_records(request, student_id):
    return request.session.get(session_key(student_id), [])


def save_session_records(request, student_id, records):
    request.session[session_key(student_id)] = records
    request.session.modified = True


def attendance_duplicate_exists(person, data):
    return AttendanceRecord.objects.filter(
        person=person,
        date=data["date"],
        time_in_1=data.get("time_in_1") or None,
        time_out_1=data.get("time_out_1") or None,
        time_in_2=data.get("time_in_2") or None,
        time_out_2=data.get("time_out_2") or None,
    ).exists()


def identify(request):
    if request.method == "POST":
        form = IdentifyForm(request.POST)
        if form.is_valid():
            student_id = form.cleaned_data["student_id"].strip()
            name = form.cleaned_data["name"].strip()
            person, created = Person.objects.get_or_create(
                student_id=student_id,
                defaults={"name": name},
            )
            if not created and name and person.name != name:
                person.name = name
                person.save(update_fields=["name", "updated_at"])
            return render(request, "identify_success.html", {"person": person})
    else:
        form = IdentifyForm()
    return render(request, "identify.html", {"form": form})


def person_lookup(request, student_id):
    person = get_object_or_404(Person, student_id=student_id)
    return JsonResponse({"student_id": person.student_id, "name": person.name})


def switch_person(request):
    for key in list(request.session.keys()):
        if key.startswith("scan_session_") or key.startswith("pending_scan_"):
            del request.session[key]
    request.session.modified = True
    return render(request, "switch.html")


def dashboard(request, student_id):
    person = get_object_or_404(Person, student_id=student_id)
    records = person.attendance_records.all()
    completed = records.aggregate(total=Sum("total_minutes"))["total"] or 0
    remaining = max(person.target_minutes - completed, 0)
    history = [
        {
            "date": record.date,
            "time": record_time_range(record),
            "total": format_minutes(record.total_minutes),
        }
        for record in records[:30]
    ]
    context = {
        "person": person,
        "completed": format_minutes(completed),
        "remaining": format_minutes(remaining),
        "target": format_minutes(person.target_minutes),
        "progress": progress_percent(completed, person.target_minutes),
        "history": history,
    }
    return render(request, "dashboard.html", context)


def scan(request, student_id):
    person = get_object_or_404(Person, student_id=student_id)
    form = ScanUploadForm()
    return render(request, "scanner.html", {"form": form, "person": person})


def upload_scan_api(request):
    if request.method != "POST":
        return api_error("request_method", "Only POST uploads are supported.", status=405)

    student_id = request.POST.get("person_id", "").strip()
    if not student_id:
        return api_error("invalid_upload", "Missing selected person.", status=400)
    person = Person.objects.filter(student_id=student_id).first()
    if person is None:
        return api_error("person_lookup", "Selected person was not found.", status=404)
    if "image" not in request.FILES:
        return api_error("image_validation", "Missing image. Please choose a logbook image.", status=400)

    form = ScanUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        error = form.errors.get("image", ["Invalid file type. Please upload a JPG, JPEG, PNG, or WEBP image."])[0]
        return api_error("image_validation", str(error), status=400)

    image = form.cleaned_data["image"]
    image_path = default_storage.save(f"logbook_scans/originals/{student_id}/{image.name}", image)
    crop_box = parse_crop_box(request)

    try:
        scan_result = scan_handwritten_logbook(default_storage.path(image_path), person, crop_box=crop_box)
    except TableDetectionError as exc:
        logger.exception("Table detection failed for uploaded scan.")
        request.session[pending_key(student_id)] = pending_failure(person, image_path, str(exc), crop_box=crop_box)
        request.session.modified = True
        return api_error(
            "table_detection",
            "No attendance table or rows were detected clearly enough.",
            detail=str(exc),
            status=422,
            actions=["try_again", "select_row_manually", "enter_attendance_manually"],
        )
    except (HandwritingOCRUnavailable, OCRUnavailable) as exc:
        logger.exception("OCR processing is unavailable.")
        request.session[pending_key(student_id)] = pending_failure(person, image_path, str(exc), crop_box=crop_box)
        request.session.modified = True
        return api_error(
            "ocr_processing",
            "The handwriting OCR service is not available.",
            detail=str(exc),
            status=503,
            actions=["try_again", "enter_attendance_manually"],
        )
    except Exception:
        logger.exception("Unexpected scanner upload failure.")
        return api_error(
            "server_error",
            "A server error occurred while processing the image.",
            status=500,
        )

    possible_rows = [serialize_recognized_row(row, person.name) for row in scan_result.rows]
    selected = scan_result.selected
    pending = pending_from_row(selected, person, image_path, scan_result, possible_rows) if selected else {
        "name": person.name,
        "date": "",
        "time_in_1": "",
        "time_out_1": "",
        "time_in_2": "",
        "time_out_2": "",
        "total_minutes": 0,
        "confidence": scan_result.confidence,
        "warnings": scan_result.warnings,
        "source_line": "",
        "source_image": image_path,
        "ocr_text": "",
        "possible_rows": possible_rows,
        "crop_box": crop_box,
        "ocr_debug": scan_result.debug,
    }
    request.session[pending_key(student_id)] = pending
    request.session.modified = True

    if selected is None:
        return api_error(
            "name_matching",
            "Could not confidently identify the selected person.",
            status=422,
            warnings=scan_result.warnings,
            possible_rows=possible_rows,
            candidates=possible_rows,
            confidence=scan_result.confidence,
            ocr_debug=scan_result.debug,
            actions=["try_again", "select_row_manually", "enter_attendance_manually"],
        )

    if not selected.date or selected.total_minutes == 0:
        return api_error(
            "time_extraction",
            "The row was found, but one or more date/time values need review.",
            status=422,
            warnings=scan_result.warnings,
            possible_rows=possible_rows,
            pending=pending,
            confidence=scan_result.confidence,
            ocr_debug=scan_result.debug,
            actions=["edit_result", "select_row_manually", "enter_attendance_manually"],
        )

    return JsonResponse(
        {
            "success": True,
            "person": person.name,
            "date": pending["date"],
            "time_in_1": pending["time_in_1"],
            "time_out_1": pending["time_out_1"],
            "time_in_2": pending["time_in_2"],
            "time_out_2": pending["time_out_2"],
            "total_hours": decimal_hours(selected.total_minutes),
            "total_minutes": selected.total_minutes,
            "total_display": format_minutes(selected.total_minutes),
            "confidence": scan_result.confidence,
            "warnings": scan_result.warnings,
            "source_line": selected.name_text,
            "possible_rows": possible_rows,
            "ocr_debug": scan_result.debug,
            "source_image_url": default_storage.url(image_path),
        }
    )


def select_scan_row_api(request):
    if request.method != "POST":
        return api_error("request_method", "Only POST row selection is supported.", status=405)

    student_id = request.POST.get("person_id", "").strip()
    person = Person.objects.filter(student_id=student_id).first()
    if person is None:
        return api_error("person_lookup", "Selected person was not found.", status=404)
    try:
        row_index = int(request.POST.get("row_index", ""))
    except ValueError:
        return api_error("attendance_row_detection", "Invalid row selection.", status=400)

    pending = request.session.get(pending_key(student_id), {})
    image_path = pending.get("source_image")
    if not image_path:
        return api_error("invalid_upload", "No uploaded image is available for row selection.", status=400)

    try:
        scan_result = scan_row_by_index(
            default_storage.path(image_path),
            person,
            row_index,
            crop_box=pending.get("crop_box"),
        )
    except TableDetectionError as exc:
        logger.exception("Manual row selection failed.")
        return api_error("attendance_row_detection", str(exc), status=422)
    except (HandwritingOCRUnavailable, OCRUnavailable) as exc:
        logger.exception("OCR processing is unavailable during manual row selection.")
        return api_error("ocr_processing", "The handwriting OCR service is not available.", detail=str(exc), status=503)
    except Exception:
        logger.exception("Unexpected manual row selection failure.")
        return api_error("server_error", "A server error occurred while processing the selected row.", status=500)

    row = scan_result.selected
    possible_rows = [serialize_recognized_row(item, person.name) for item in scan_result.rows]
    pending = pending_from_row(row, person, image_path, scan_result, possible_rows)
    pending["crop_box"] = request.session.get(pending_key(student_id), {}).get("crop_box")
    pending["ocr_debug"] = scan_result.debug
    request.session[pending_key(student_id)] = pending
    request.session.modified = True

    if not row.date or row.total_minutes == 0:
        return api_error(
            "time_extraction",
            "The selected row was read, but one or more date/time values need review.",
            status=422,
            pending=pending,
            possible_rows=possible_rows,
            confidence=scan_result.confidence,
            ocr_debug=scan_result.debug,
            actions=["edit_result", "enter_attendance_manually"],
        )

    return JsonResponse(
        {
            "success": True,
            "person": person.name,
            "date": pending["date"],
            "time_in_1": pending["time_in_1"],
            "time_out_1": pending["time_out_1"],
            "time_in_2": pending["time_in_2"],
            "time_out_2": pending["time_out_2"],
            "total_hours": decimal_hours(row.total_minutes),
            "total_minutes": row.total_minutes,
            "total_display": format_minutes(row.total_minutes),
            "confidence": scan_result.confidence,
            "warnings": scan_result.warnings,
            "source_line": row.name_text,
            "possible_rows": possible_rows,
            "ocr_debug": scan_result.debug,
        }
    )


def api_error(stage, error, status, **extra):
    payload = {"success": False, "stage": stage, "error": error}
    payload.update(extra)
    return JsonResponse(payload, status=status)


def pending_failure(person, image_path, warning, crop_box=None):
    return {
        "name": person.name,
        "date": "",
        "time_in_1": "",
        "time_out_1": "",
        "time_in_2": "",
        "time_out_2": "",
        "total_minutes": 0,
        "confidence": 0,
        "warnings": [warning],
        "source_line": "",
        "source_image": image_path,
        "ocr_text": "",
        "possible_rows": [],
        "crop_box": crop_box,
    }


def parse_crop_box(request):
    keys = ["crop_x", "crop_y", "crop_w", "crop_h"]
    if not all(request.POST.get(key) for key in keys):
        return None
    try:
        values = [int(float(request.POST[key])) for key in keys]
    except ValueError:
        return None
    if values[2] < 20 or values[3] < 20:
        return None
    return tuple(values)


def serialize_recognized_row(row, target_name):
    return {
        "index": row.index,
        "box": row.box,
        "name": row.name_text,
        "target_name": target_name,
        "similarity": round(row.name_similarity * 100),
        "date": serialize_date(row.date),
        "date_text": row.date_text,
        "time_in_1": serialize_time(row.time_in_1),
        "time_out_1": serialize_time(row.time_out_1),
        "time_in_2": serialize_time(row.time_in_2),
        "time_out_2": serialize_time(row.time_out_2),
        "am_in_text": row.am_in_text,
        "am_out_text": row.am_out_text,
        "pm_in_text": row.pm_in_text,
        "pm_out_text": row.pm_out_text,
        "total_minutes": row.total_minutes,
        "total_display": format_minutes(row.total_minutes),
        "confidence": round(row.ocr_confidence * 100),
        "warnings": row.warnings,
    }


def pending_from_row(row, person, image_path, scan_result, possible_rows):
    return {
        "name": person.name,
        "date": serialize_date(row.date),
        "time_in_1": serialize_time(row.time_in_1),
        "time_out_1": serialize_time(row.time_out_1),
        "time_in_2": serialize_time(row.time_in_2),
        "time_out_2": serialize_time(row.time_out_2),
        "total_minutes": row.total_minutes,
        "confidence": scan_result.confidence,
        "warnings": scan_result.warnings,
        "source_line": row.name_text,
        "source_image": image_path,
        "ocr_text": "",
        "possible_rows": possible_rows,
        "ocr_debug": scan_result.debug,
    }


def review(request, student_id):
    person = get_object_or_404(Person, student_id=student_id)
    pending = request.session.get(pending_key(student_id), {})
    if request.method == "POST":
        if request.POST.get("action") == "review":
            form = ReviewAttendanceForm(initial=pending)
        else:
            form = ReviewAttendanceForm(request.POST)
            if form.is_valid():
                data = form.cleaned_data
                total = calculate_total_minutes(
                    data["time_in_1"],
                    data["time_out_1"],
                    data.get("time_in_2"),
                    data.get("time_out_2"),
                )
                record = {
                    "name": data["name"],
                    "date": data["date"].isoformat(),
                    "time_in_1": serialize_time(data["time_in_1"]),
                    "time_out_1": serialize_time(data["time_out_1"]),
                    "time_in_2": serialize_time(data.get("time_in_2")),
                    "time_out_2": serialize_time(data.get("time_out_2")),
                    "total_minutes": total,
                    "source_image": pending.get("source_image", ""),
                }
                duplicate_data = {
                    "date": data["date"],
                    "time_in_1": data["time_in_1"],
                    "time_out_1": data["time_out_1"],
                    "time_in_2": data.get("time_in_2"),
                    "time_out_2": data.get("time_out_2"),
                }
                if attendance_duplicate_exists(person, duplicate_data):
                    request.session[pending_key(student_id)] = record
                    request.session.modified = True
                    return render(request, "duplicate.html", {"person": person, "record": decorate_record(record)})

                records = get_session_records(request, student_id)
                session_duplicate = any(
                    item["date"] == record["date"]
                    and item["time_in_1"] == record["time_in_1"]
                    and item["time_out_1"] == record["time_out_1"]
                    and item["time_in_2"] == record["time_in_2"]
                    and item["time_out_2"] == record["time_out_2"]
                    for item in records
                )
                if session_duplicate:
                    messages.warning(request, "This attendance record has already been recorded.")
                else:
                    records.append(record)
                    save_session_records(request, student_id, records)
                request.session.pop(pending_key(student_id), None)
                request.session.modified = True
                return redirect("scanner:scan_session", student_id=student_id)
    else:
        form = ReviewAttendanceForm(initial=pending or {"name": person.name, "date": timezone.localdate()})

    total = 0
    if pending:
        total = pending.get("total_minutes", 0)
    return render(
        request,
        "review.html",
        {
            "person": person,
            "form": form,
            "pending": pending,
            "total_display": format_minutes(total),
        },
    )


def scan_session(request, student_id):
    person = get_object_or_404(Person, student_id=student_id)
    records = [decorate_record(record) for record in get_session_records(request, student_id)]
    session_total = sum(record.get("total_minutes", 0) for record in records)
    return render(
        request,
        "session.html",
        {
            "person": person,
            "records": records,
            "session_total": format_minutes(session_total),
        },
    )


def summary(request, student_id):
    person = get_object_or_404(Person, student_id=student_id)
    records = get_session_records(request, student_id)
    previous_total = person.attendance_records.aggregate(total=Sum("total_minutes"))["total"] or 0
    current_total = 0
    skipped_duplicates = 0

    if request.method == "POST":
        with transaction.atomic():
            for item in records:
                data = {
                    "date": item["date"],
                    "time_in_1": item.get("time_in_1") or None,
                    "time_out_1": item.get("time_out_1") or None,
                    "time_in_2": item.get("time_in_2") or None,
                    "time_out_2": item.get("time_out_2") or None,
                }
                if AttendanceRecord.objects.filter(person=person, **data).exists():
                    skipped_duplicates += 1
                    continue
                try:
                    AttendanceRecord.objects.create(
                        person=person,
                        total_minutes=item["total_minutes"],
                        source_image=item.get("source_image") or None,
                        **data,
                    )
                    current_total += item["total_minutes"]
                except IntegrityError:
                    skipped_duplicates += 1
        save_session_records(request, student_id, [])
    else:
        current_total = sum(item.get("total_minutes", 0) for item in records)

    completed = previous_total + current_total
    remaining = max(person.target_minutes - completed, 0)
    context = {
        "person": person,
        "previous_total": format_minutes(previous_total),
        "current_total": format_minutes(current_total),
        "completed": format_minutes(completed),
        "target": format_minutes(person.target_minutes),
        "remaining": format_minutes(remaining),
        "skipped_duplicates": skipped_duplicates,
        "saved": request.method == "POST",
    }
    return render(request, "summary.html", context)
