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
        return date.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
    }

    function formatTime(value) {
        if (!value) return "";
        const [hours, minutes] = value.split(":").map(Number);
        const date = new Date();
        date.setHours(hours, minutes, 0, 0);
        return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    }

    function arrowRange(start, end) {
        if (!start || !end) return "Not detected";
        return `${formatTime(start)} -> ${formatTime(end)}`;
    }

    function populateResult(data) {
        document.getElementById("resultPerson").textContent = data.person;
        document.getElementById("resultDate").textContent = formatDate(data.date);
        document.getElementById("resultMorning").textContent = arrowRange(data.time_in_1, data.time_out_1);
        document.getElementById("resultAfternoon").textContent = arrowRange(data.time_in_2, data.time_out_2);
        document.getElementById("resultTotal").textContent = data.total_display || "0h 00m";
        document.getElementById("resultConfidence").textContent = `${Math.round((data.confidence || 0) * 100)}%`;
        document.getElementById("resultWarnings").textContent = data.warnings && data.warnings.length
            ? "OCR is uncertain. Please verify before confirming."
            : "";

        document.getElementById("fieldName").value = data.person || "";
        document.getElementById("fieldDate").value = data.date || "";
        document.getElementById("fieldTimeIn1").value = data.time_in_1 || "";
        document.getElementById("fieldTimeOut1").value = data.time_out_1 || "";
        document.getElementById("fieldTimeIn2").value = data.time_in_2 || "";
        document.getElementById("fieldTimeOut2").value = data.time_out_2 || "";

        hide(scanState);
        hide(scanError);
        show(resultForm);
    }

    function postForm(url, body) {
        return fetch(url, {
            method: "POST",
            headers: { "X-CSRFToken": csrfToken },
            body
        }).then((response) => response.json().then((data) => ({
            ok: response.ok,
            status: response.status,
            data
        })));
    }

    function selectRow(rowIndex) {
        const body = new FormData();
        body.append("person_id", root.dataset.personId);
        body.append("row_index", rowIndex);
        hide(resultForm);
        hide(scanError);
        startScanState();
        postForm(root.dataset.selectRowUrl, body)
            .then(({ ok, status, data }) => {
                clearInterval(stateTimer);
                if (!ok || !data.success) {
                    if (data.pending) {
                        populatePending(data);
                        return;
                    }
                    showError(data.error || "Unable to read the selected row.", data.possible_rows || [], data, status);
                    return;
                }
                populateResult(data);
            })
            .catch(() => {
                clearInterval(stateTimer);
                showError("Unable to process the selected row. Please try again.", [], { stage: "server_error" }, 500);
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
            warnings: data.warnings || pending.warnings || ["OCR needs manual correction."]
        });
    }

    chooseButton.addEventListener("click", () => imageInput.click());
    changeButton.addEventListener("click", () => imageInput.click());
    tryAnotherButton.addEventListener("click", () => {
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
        const error = validateFile(selectedFile);
        if (error) {
            showError(error, [], { stage: "image_validation" }, 400);
            return;
        }

        hide(resultForm);
        hide(scanError);
        startScanState();

        const body = new FormData();
        body.append("image", selectedFile);
        body.append("person_id", root.dataset.personId);

        postForm(root.dataset.uploadUrl, body)
            .then(({ ok, status, data }) => {
                clearInterval(stateTimer);
                if (!ok || !data.success) {
                    if (data.pending) {
                        populatePending(data);
                        return;
                    }
                    showError(data.error || "Unable to read the logbook.", data.possible_rows || data.candidates || [], data, status);
                    return;
                }
                populateResult(data);
            })
            .catch(() => {
                clearInterval(stateTimer);
                showError("Unable to upload or process this image. Please try again.", [], { stage: "server_error" }, 500);
            });
    });
}
