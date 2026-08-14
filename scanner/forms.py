from django import forms

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024


class IdentifyForm(forms.Form):
    student_id = forms.CharField(label="Student/Employee ID", max_length=50)
    name = forms.CharField(label="Full Name", max_length=255)


class ScanUploadForm(forms.Form):
    image = forms.FileField(label="Logbook image")

    def clean_image(self):
        image = self.cleaned_data["image"]
        extension = "." + image.name.rsplit(".", 1)[-1].lower() if "." in image.name else ""
        if extension not in ALLOWED_IMAGE_EXTENSIONS or image.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise forms.ValidationError("Invalid file type. Please upload a JPG, JPEG, PNG, or WEBP image.")
        if image.size > MAX_IMAGE_SIZE:
            raise forms.ValidationError("Image is too large. Please upload an image up to 10 MB.")
        try:
            from PIL import Image

            image.seek(0)
            with Image.open(image) as opened:
                opened.verify()
            image.seek(0)
        except Exception as exc:
            raise forms.ValidationError("Invalid file type. Please upload a JPG, JPEG, PNG, or WEBP image.") from exc
        return image


class ReviewAttendanceForm(forms.Form):
    name = forms.CharField(max_length=255)
    date = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}))
    time_in_1 = forms.TimeField(required=False, widget=forms.TimeInput(attrs={"type": "time"}))
    time_out_1 = forms.TimeField(required=False, widget=forms.TimeInput(attrs={"type": "time"}))
    time_in_2 = forms.TimeField(required=False, widget=forms.TimeInput(attrs={"type": "time"}))
    time_out_2 = forms.TimeField(required=False, widget=forms.TimeInput(attrs={"type": "time"}))

    def clean(self):
        cleaned = super().clean()
        pairs = [
            (cleaned.get("time_in_1"), cleaned.get("time_out_1"), "first"),
            (cleaned.get("time_in_2"), cleaned.get("time_out_2"), "second"),
        ]
        has_complete_pair = False
        for time_in, time_out, label in pairs:
            if bool(time_in) ^ bool(time_out):
                raise forms.ValidationError(f"The {label} work period needs both Time In and Time Out.")
            if time_in and time_out:
                has_complete_pair = True
        if not has_complete_pair:
            raise forms.ValidationError("Enter at least one complete Time In and Time Out pair.")
        return cleaned
