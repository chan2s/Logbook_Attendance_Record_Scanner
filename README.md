# Django OCR Logbook Time Scanner

A no-login Django app for uploading logbook page images, OCR-reading the selected person's row, reviewing the extracted attendance, batching multiple confirmed scans in a session, and saving totals only when the user clicks Total Time.

## Run

```powershell
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/`.

## OCR Requirements

The upload workflow and validation work with Django and Pillow. Real OCR also needs:

```powershell
pip install pytesseract
```

Install the Tesseract OCR engine separately and make sure `tesseract` is available on `PATH`.

## Privacy Behavior

There are no Django user accounts, registrations, passwords, or authentication screens. The browser stores only `localStorage.person_id`; attendance data lives in the Django database.
