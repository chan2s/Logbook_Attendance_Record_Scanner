from django.urls import path

from . import views

app_name = "scanner"

urlpatterns = [
    path("", views.identify, name="identify"),
    path("person/<str:student_id>/", views.person_lookup, name="person_lookup"),
    path("dashboard/<str:student_id>/", views.dashboard, name="dashboard"),
    path("scan/<str:student_id>/", views.scan, name="scan"),
    path("api/scanner/upload/", views.upload_scan_api, name="upload_scan_api"),
    path("api/scanner/select-row/", views.select_scan_row_api, name="select_scan_row_api"),
    path("review/<str:student_id>/", views.review, name="review"),
    path("session/<str:student_id>/", views.scan_session, name="scan_session"),
    path("summary/<str:student_id>/", views.summary, name="summary"),
    path("switch/", views.switch_person, name="switch_person"),
]
