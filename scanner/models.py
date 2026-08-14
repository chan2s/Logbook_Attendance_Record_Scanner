from django.db import models


class Person(models.Model):
    student_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    target_minutes = models.PositiveIntegerField(default=300 * 60)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.student_id} - {self.name}"


class AttendanceRecord(models.Model):
    person = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name="attendance_records",
    )
    date = models.DateField()
    time_in_1 = models.TimeField(null=True, blank=True)
    time_out_1 = models.TimeField(null=True, blank=True)
    time_in_2 = models.TimeField(null=True, blank=True)
    time_out_2 = models.TimeField(null=True, blank=True)
    total_minutes = models.PositiveIntegerField(default=0)
    source_image = models.ImageField(upload_to="logbook_scans/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "-time_in_1", "-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "person",
                    "date",
                    "time_in_1",
                    "time_out_1",
                    "time_in_2",
                    "time_out_2",
                ],
                name="unique_attendance_exact_times",
            )
        ]

    def __str__(self):
        return f"{self.person.student_id} {self.date} {self.total_minutes}m"
