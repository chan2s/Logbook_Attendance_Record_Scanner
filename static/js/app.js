function restoreRememberedPerson() {
    const personId = localStorage.getItem("person_id");
    if (!personId) return;

    fetch(`/person/${encodeURIComponent(personId)}/`)
        .then((response) => {
            if (!response.ok) throw new Error("Person not found");
            return response.json();
        })
        .then(() => {
            window.location.href = `/dashboard/${encodeURIComponent(personId)}/`;
        })
        .catch(() => {
            localStorage.removeItem("person_id");
        });
}

function setupUploadScanner() {
    const root = document.querySelector(".upload-scanner");
    if (!root) return;
    // Guard against double initialization (e.g. a script block running twice).
    if (root.dataset.scannerInitialized === "true") return;
    root.dataset.scannerInitialized = "true";

    const uploadForm = document.getElementById("uploadForm");
    const imageInput = document.getElementById("imageInput");
    const dropZone = document.getElementById("dropZone");
    const chooseButton = document.getElementById("chooseImageButton");
    const changeButton = document.getElementById("changeImageButton");
    const previewPanel = document.getElementById("previewPanel");
    const imagePreview = document.getElementById("imagePreview");
    const fileName = document.getElementById("fileName");
    const scanState = document.getElementById("scanState");
    const scanStateText = document.getElementById("scanStateText");
    const scanError = document.getElementById("scanError");
    const scanErrorTitle = document.getElementById("scanErrorTitle");
    const scanErrorDetail = document.getElementById("scanErrorDetail");
    const ocrDebug = document.getElementById("ocrDebug");
    const possibleMatches = document.getElementById("possibleMatches");
    const possibleMatchesList = document.getElementById("possibleMatchesList");
    const tryAnotherButton = document.getElementById("tryAnotherButton");
    const resultForm = document.getElementById("resultForm");
    const scanImageButton = document.getElementById("scanImageButton");
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    const validExtensions = [".jpg", ".jpeg", ".png", ".webp"];
    const maxSize = 10 * 1024 * 1024;
    const states = [
        "Uploading image...",
        "Preparing image...",
        "Detecting table...",
        "Detecting rows...",
        "Finding your name...",
        "Reading attendance times...",
        "Preparing result..."
    ];
    let selectedFile = null;
    let stateTimer = null;
    let activeRequestId = 0;
    let scanInProgress = false;
    let lastScanData = null;

    const csrfToken = uploadForm.querySelector("[name=csrfmiddlewaretoken]").value;

    function show(element) {
        element.classList.remove("hidden");
    }

    function hide(element) {
        element.classList.add("hidden");
    }

    function resetState() {
        clearInterval(stateTimer);
        hide(scanState);
        hide(scanError);
        hide(resultForm);
        hide(possibleMatches);
        possibleMatchesList.innerHTML = "";
        ocrDebug.textContent = "";
        hide(ocrDebug);
    }

    function setScanning(scanning) {
        scanInProgress = scanning;
        scanImageButton.disabled = scanning;
        changeButton.disabled = scanning;
        tryAnotherButton.disabled = scanning;
    }

    function validateFile(file) {
        if (!file) return "Please choose an image.";
        const lowerName = file.name.toLowerCase();
        const hasAllowedExtension = validExtensions.some((extension) => lowerName.endsWith(extension));
        if (!validTypes.includes(file.type) || !hasAllowedExtension) {
            return "Invalid file type. Please upload a JPG, JPEG, PNG, or WEBP image.";
        }
        if (file.size > maxSize) {
            return "Image is too large. Please upload an image up to 10 MB.";
        }
        return "";
    }

    function displayFile(file) {
        const error = validateFile(file);
        if (error) {
            selectedFile = null;
            showError(error, [], { stage: "image_validation" }, 400);
            return;
        }

        resetState();
        activeRequestId += 1;
        setScanning(false);
        selectedFile = file;
        imagePreview.src = URL.createObjectURL(file);
        fileName.textContent = file.name;
        hide(dropZone);
        show(previewPanel);
    }

    function showError(message, rows = [], data = {}, status = 0) {
        resetState();
        scanErrorTitle.textContent = titleForError(data.stage, status);
        scanErrorDetail.textContent = detailForError(message, data, status);
        if (data.ocr_debug) {
            ocrDebug.textContent = JSON.stringify(data.ocr_debug, null, 2);
            show(ocrDebug);
        }
        if (rows.length) {
            rows.forEach((row) => {
                const button = document.createElement("button");
                button.className = "possible-match";
                button.type = "button";
                button.innerHTML = `
                    <strong>${escapeHtml(row.name || "Unread name")} - ${row.similarity || 0}%</strong>
                    <span>${escapeHtml(row.date_text || row.date || "No date")} - ${escapeHtml(row.am_in_text || "")} to ${escapeHtml(row.am_out_text || "")} - ${escapeHtml(row.pm_in_text || "")} to ${escapeHtml(row.pm_out_text || "")}</span>
                `;
                button.addEventListener("click", () => selectRow(row.index));
                possibleMatchesList.appendChild(button);
            });
            show(possibleMatches);
        }
        show(scanError);
    }

    function titleForError(stage, status) {
        if (status >= 500) return "Server processing error";
        if (status === 404) return "Person not found";
        if (status === 400) return "Invalid upload";
        if (stage === "name_matching") return "We could not confidently identify your row";
        if (stage === "time_extraction") return "Attendance times need review";
        if (stage === "table_detection") return "Attendance table not detected";
        if (stage === "ocr_processing") return "OCR processing failed";
        return "Scan needs review";
    }

    function detailForError(message, data, status) {
        if (status >= 500) return message || "A server error occurred while processing the image.";
        if (data.detail) return `${message} ${data.detail}`;
        return message || "Unable to read the logbook.";
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function startScanState() {
        let index = 0;
        scanStateText.textContent = states[index];
        show(scanState);
        stateTimer = setInterval(() => {
            index = Math.min(index + 1, states.length - 1);
            scanStateText.textContent = states[index];
        }, 850);
    }

    function formatDate(value) {
        if (!value) return "Needs review";
        const date = new Date(`${value}T00:00:00`);
        if (Number.isNaN(date.getTime())) return "Needs review";
        return date.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
    }

    function formatTime(value) {
        if (!value) return "";
        const match = String(value).trim().match(/^(\d{1,2}):(\d{2})$/);
        if (!match) return String(value);
        const hours = parseInt(match[1], 10);
        const minutes = match[2];
        if (hours > 23) return String(value);
        const period = hours >= 12 ? "PM" : "AM";
        let displayHours = hours % 12;
        if (displayHours === 0) displayHours = 12;
        return `${String(displayHours).padStart(2, "0")}:${minutes} ${period}`;
    }

    function arrowRange(start, end) {
        if (!start || !end) return "Not detected";
        return `${formatTime(start)} - ${formatTime(end)}`;
    }

    const FIELD_LABELS = {
        name: "Name",
        date: "Date",
        am_in: "AM-in",
        am_out: "AM-out",
        pm_in: "PM-in",
        pm_out: "PM-out"
    };

    function populateConfidence(data) {
        const grid = document.getElementById("confidenceGrid");
        if (!grid) return;
        grid.innerHTML = "";
        const confidences = data.field_confidences || {};
        const needsReview = data.needs_review || [];
        Object.keys(FIELD_LABELS).forEach((field) => {
            const cell = document.createElement("span");
            cell.textContent = FIELD_LABELS[field];
            const value = document.createElement("strong");
            if (needsReview.includes(field)) {
                value.className = "needs-review";
                value.textContent = "Needs review";
            } else {
                const percent = Math.round((confidences[field] || 0) * 100);
                value.textContent = `${percent}%`;
            }
            grid.appendChild(cell);
            grid.appendChild(value);
        });
    }

    function populateReviewBadges(data) {
        const needsReview = data.needs_review || [];
        document.querySelectorAll("[data-review]").forEach((badge) => {
            badge.classList.toggle("hidden", !needsReview.includes(badge.dataset.review));
        });
    }

    function setText(id, value) {
        const element = document.getElementById(id);
        if (!element) {
            console.error(`[SCAN] Missing result element: ${id}`);
            return;
        }
        element.textContent = value;
    }

    function setValue(id, value) {
        const element = document.getElementById(id);
        if (!element) {
            console.error(`[SCAN] Missing form field: ${id}`);
            return;
        }
        element.value = value;
    }

    function populateResult(data) {
        lastScanData = data;

        setText("resultPerson", data.person || "");
        setText("resultDate", formatDate(data.date));
        setText("resultMorning", arrowRange(data.time_in_1, data.time_out_1));
        setText("resultAfternoon", arrowRange(data.time_in_2, data.time_out_2));
        setText("resultTotal", data.total_display || "0h 00m");
        setText("resultWarnings", data.warnings && data.warnings.length ? "OCR is uncertain. Please verify before confirming." : "");
        setText("resultConfidence", `${Math.round((data.confidence || 0) * 100)}%`);

        setValue("fieldName", data.person || "");
        setValue("fieldDate", data.date || "");
        setValue("fieldTimeIn1", data.time_in_1 || "");
        setValue("fieldTimeOut1", data.time_out_1 || "");
        setValue("fieldTimeIn2", data.time_in_2 || "");
        setValue("fieldTimeOut2", data.time_out_2 || "");

        if (resultForm) {
            resultForm.dataset.requestId = data.request_id || "";
        }

        populateConfidence(data);
        populateReviewBadges(data);

        hide(scanState);
        hide(scanError);
        show(resultForm);
    }

    function postForm(url, body) {
        return fetch(url, {
            method: "POST",
            headers: { "X-CSRFToken": csrfToken },
            body
        }).then((response) => {
            return response.json()
                .catch(() => ({
                    success: false,
                    stage: "server_error",
                    error: "The server returned an unreadable response."
                }))
                .then((data) => ({
                    ok: response.ok,
                    status: response.status,
                    data
                }));
        });
    }

    function selectRow(rowIndex) {
        if (scanInProgress) return;
        const body = new FormData();
        body.append("person_id", root.dataset.personId);
        body.append("row_index", rowIndex);
        body.append("request_id", crypto.randomUUID());
        hide(resultForm);
        hide(scanError);
        startScanState();
        setScanning(true);
        const scanId = activeRequestId + 1;
        activeRequestId = scanId;
        postForm(root.dataset.selectRowUrl, body)
            .then(({ ok, status, data }) => {
                if (scanId !== activeRequestId) return;
                console.log("[SCAN] HTTP status:", status);
                console.log("[SCAN] HTTP OK:", ok);
                console.log("[SCAN] Response data:", data);
                console.log("[SCAN] Response success:", data?.success);
                clearInterval(stateTimer);
                setScanning(false);
                if (!ok || !data.success) {
                    console.error("[SCAN] Backend reported failure:", data);
                    if (data.pending) {
                        populatePending(data);
                        return;
                    }
                    showError(data.error || "Unable to read the selected row.", data.possible_rows || [], data, status);
                    return;
                }
                console.log("[SCAN] Backend successful. Calling populateResult()");
                try {
                    populateResult(data);
                    console.log("[SCAN] populateResult() completed successfully");
                } catch (error) {
                    console.error("[SCAN] populateResult() threw:", error);
                    console.error("[SCAN] Error stack:", error?.stack);
                    showError("The scan was processed, but the result could not be displayed.", [], { stage: "result_display", error: error?.message || String(error) }, 500);
                }
            })
            .catch((error) => {
                console.error("[SCAN] FRONTEND ERROR:", error);
                console.error("[SCAN] Error stack:", error?.stack);
                if (scanId !== activeRequestId) return;
                clearInterval(stateTimer);
                setScanning(false);
                showError("The scan was processed, but the result could not be displayed.", [], { stage: "result_display", error: error?.message || String(error) }, 500);
            });
    }

    function populatePending(data) {
        const pending = data.pending || {};
        populateResult({
            success: true,
            person: pending.name || "",
            date: pending.date || "",
            time_in_1: pending.time_in_1 || "",
            time_out_1: pending.time_out_1 || "",
            time_in_2: pending.time_in_2 || "",
            time_out_2: pending.time_out_2 || "",
            total_display: pending.total_display || "0h 00m",
            confidence: data.confidence || pending.confidence || 0,
            field_confidences: pending.field_confidences || {},
            needs_review: pending.needs_review || [],
            warnings: data.warnings || pending.warnings || ["OCR needs manual correction."]
        });
    }

    chooseButton.addEventListener("click", () => imageInput.click());
    changeButton.addEventListener("click", () => imageInput.click());
    tryAnotherButton.addEventListener("click", () => {
        activeRequestId += 1;
        setScanning(false);
        hide(scanError);
        show(dropZone);
        hide(previewPanel);
        imageInput.value = "";
        selectedFile = null;
    });
    dropZone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            imageInput.click();
        }
    });
    imageInput.addEventListener("change", () => displayFile(imageInput.files[0]));

    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (event) => {
        displayFile(event.dataTransfer.files[0]);
    });

    uploadForm.addEventListener("submit", (event) => {
        event.preventDefault();
        if (scanInProgress) return;
        const error = validateFile(selectedFile);
        if (error) {
            showError(error, [], { stage: "image_validation" }, 400);
            return;
        }

        hide(resultForm);
        hide(scanError);
        startScanState();
        setScanning(true);
        const scanId = activeRequestId + 1;
        activeRequestId = scanId;

        const body = new FormData();
        body.append("image", selectedFile);
        body.append("person_id", root.dataset.personId);
        body.append("request_id", crypto.randomUUID());

        postForm(root.dataset.uploadUrl, body)
            .then(({ ok, status, data }) => {
                if (scanId !== activeRequestId) return;
                console.log("[SCAN] HTTP status:", status);
                console.log("[SCAN] HTTP OK:", ok);
                console.log("[SCAN] Response data:", data);
                console.log("[SCAN] Response success:", data?.success);
                clearInterval(stateTimer);
                setScanning(false);
                if (!ok || !data.success) {
                    console.error("[SCAN] Backend reported failure:", data);
                    if (data.pending) {
                        populatePending(data);
                        return;
                    }
                    showError(data.error || "Unable to read the logbook.", data.possible_rows || data.candidates || [], data, status);
                    return;
                }
                console.log("[SCAN] Backend successful. Calling populateResult()");
                try {
                    populateResult(data);
                    console.log("[SCAN] populateResult() completed successfully");
                } catch (error) {
                    console.error("[SCAN] populateResult() threw:", error);
                    console.error("[SCAN] Error stack:", error?.stack);
                    showError("The scan was processed, but the result could not be displayed.", [], { stage: "result_display", error: error?.message || String(error) }, 500);
                }
            })
            .catch((error) => {
                console.error("[SCAN] FRONTEND ERROR:", error);
                console.error("[SCAN] Error stack:", error?.stack);
                if (scanId !== activeRequestId) return;
                clearInterval(stateTimer);
                setScanning(false);
                showError("The scan was processed, but the result could not be displayed.", [], { stage: "result_display", error: error?.message || String(error) }, 500);
            });
    });
}
