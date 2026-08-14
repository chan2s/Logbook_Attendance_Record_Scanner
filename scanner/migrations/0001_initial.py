# Generated for the OCR Logbook Time Scanner.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Person",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("student_id", models.CharField(max_length=50, unique=True)),
                ("name", models.CharField(max_length=255)),
                ("target_minutes", models.PositiveIntegerField(default=18000)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name="AttendanceRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField()),
                ("time_in_1", models.TimeField(blank=True, null=True)),
                ("time_out_1", models.TimeField(blank=True, null=True)),
                ("time_in_2", models.TimeField(blank=True, null=True)),
                ("time_out_2", models.TimeField(blank=True, null=True)),
                ("total_minutes", models.PositiveIntegerField(default=0)),
                ("source_image", models.ImageField(blank=True, null=True, upload_to="logbook_scans/")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "person",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="attendance_records",
                        to="scanner.person",
                    ),
                ),
            ],
            options={"ordering": ["-date", "-time_in_1", "-created_at"]},
        ),
        migrations.AddConstraint(
            model_name="attendancerecord",
            constraint=models.UniqueConstraint(
                fields=("person", "date", "time_in_1", "time_out_1", "time_in_2", "time_out_2"),
                name="unique_attendance_exact_times",
            ),
        ),
    ]
