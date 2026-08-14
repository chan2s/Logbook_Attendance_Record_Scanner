from datetime import datetime, timedelta


def minutes_between(start, end):
    if not start or not end:
        return 0

    base = datetime(2000, 1, 1)
    start_dt = datetime.combine(base.date(), start)
    end_dt = datetime.combine(base.date(), end)
    if end_dt < start_dt:
        end_dt += timedelta(days=1)
    return int((end_dt - start_dt).total_seconds() // 60)


def calculate_total_minutes(time_in_1, time_out_1, time_in_2=None, time_out_2=None):
    return minutes_between(time_in_1, time_out_1) + minutes_between(time_in_2, time_out_2)


def format_minutes(minutes):
    minutes = max(0, int(minutes or 0))
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m"


def progress_percent(completed_minutes, target_minutes):
    if not target_minutes:
        return 0
    return min(100, round((completed_minutes / target_minutes) * 100, 2))


def record_time_range(record):
    parts = []
    if record.time_in_1 and record.time_out_1:
        parts.append(f"{record.time_in_1.strftime('%I:%M %p').lstrip('0')} - {record.time_out_1.strftime('%I:%M %p').lstrip('0')}")
    if record.time_in_2 and record.time_out_2:
        parts.append(f"{record.time_in_2.strftime('%I:%M %p').lstrip('0')} - {record.time_out_2.strftime('%I:%M %p').lstrip('0')}")
    return " / ".join(parts) or "Incomplete"
