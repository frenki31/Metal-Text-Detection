// script.js
const imageUploadArea = document.getElementById('imageUploadArea');
const actualImageInput = document.getElementById('actualImageInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const predictButton = document.getElementById('predictButton');
const previewImageElem = document.getElementById('previewImage');
const resultsHeader = document.getElementById('resultsHeader');
const loader = document.getElementById('loader');
const errorMessageDiv = document.getElementById('errorMessage');

var API_ENDPOINT = "http://localhost:8000/predict";

if (window.location.hostname.includes("azurewebsites.net")) {
    API_ENDPOINT = window.location.origin + "/predict";
} else if (window.location.protocol === "file:") {
    console.log("Frontend running locally as a file. API endpoint: " + API_ENDPOINT);
}

let currentFile = null;
let isImagePreviewing = false; // State to track if an image is selected and previewed

// --- UI Update Functions ---
function showPreviewState(file) {
    currentFile = file;
    isImagePreviewing = true;
    const reader = new FileReader();
    reader.onload = (e) => {
        if (previewImageElem) {
            previewImageElem.src = e.target.result;
            previewImageElem.style.display = 'block';
        }
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        if (imageUploadArea) {
            imageUploadArea.style.borderStyle = 'solid';
            imageUploadArea.style.borderColor = '#dadde1';
            imageUploadArea.style.backgroundColor = '#ffffff';
            imageUploadArea.style.minHeight = 'auto';
            imageUploadArea.style.cursor = 'default'; // Make it not look clickable
        }
    }
    reader.readAsDataURL(file);

    if (predictButton) {
        predictButton.textContent = 'Detect Objects';
        predictButton.disabled = false;
    }
    hideError();
}

function showResultsState(data) {
    isImagePreviewing = true; // Image is still technically previewing (the annotated one)
    if (previewImageElem && data.annotated_image_base64) {
        previewImageElem.src = `data:image/png;base64,${data.annotated_image_base64}`;
        previewImageElem.style.display = 'block';
    }
    if (uploadPrompt) uploadPrompt.style.display = 'none';
    if (imageUploadArea) {
        imageUploadArea.style.borderStyle = 'solid';
        imageUploadArea.style.borderColor = '#dadde1';
        imageUploadArea.style.backgroundColor = '#ffffff';
        imageUploadArea.style.cursor = 'default'; // Still not clickable
    }
    
    if (resultsHeader) resultsHeader.style.display = 'block';

    if (predictButton) {
        predictButton.textContent = 'Reset'; // Change button to Reset
        predictButton.disabled = false;
    }
    hideError();
}

function resetToInitialState() {
    currentFile = null;
    isImagePreviewing = false;
    if (actualImageInput) actualImageInput.value = ''; // Clear the file input selection
    if (previewImageElem) {
        previewImageElem.style.display = 'none';
        previewImageElem.src = '#';
    }
    if (uploadPrompt) uploadPrompt.style.display = 'block';
    if (imageUploadArea) {
        imageUploadArea.style.borderStyle = 'dashed';
        imageUploadArea.style.borderColor = '#007bff';
        imageUploadArea.style.backgroundColor = '#f8f9fa';
        imageUploadArea.style.minHeight = '200px';
        imageUploadArea.style.cursor = 'pointer'; // Make it clickable again
    }
    if (predictButton) {
        predictButton.textContent = 'Detect Objects';
        predictButton.disabled = false;
    }
    hideError();
}

function showError(message) {
    if (errorMessageDiv) {
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block';
    }
}

function hideError() {
    if (errorMessageDiv) errorMessageDiv.style.display = 'none';
}

// --- Event Listeners ---
if (imageUploadArea) {
    imageUploadArea.addEventListener('click', () => {
        if (!isImagePreviewing && actualImageInput) { // Only allow click if no image is previewing
            actualImageInput.click();
        }
    });

    imageUploadArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        if (!isImagePreviewing) { // Only show hover effect if no image is previewing
            imageUploadArea.style.borderColor = '#0056b3';
        }
    });
    imageUploadArea.addEventListener('dragleave', () => {
        if (!isImagePreviewing) {
            imageUploadArea.style.borderColor = '#007bff';
        }
    });
    imageUploadArea.addEventListener('drop', (event) => {
        event.preventDefault();
        if (!isImagePreviewing) { // Only process drop if no image is previewing
            imageUploadArea.style.borderColor = '#007bff';
            if (event.dataTransfer.files && event.dataTransfer.files[0]) {
                const file = event.dataTransfer.files[0];
                if (file.type.startsWith("image/")) {
                    if (actualImageInput) actualImageInput.files = event.dataTransfer.files;
                    showPreviewState(file);
                } else {
                    showError("Please drop an image file (JPEG, PNG, GIF).");
                }
            }
        }
    });
}

if (actualImageInput) {
    actualImageInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            showPreviewState(file);
        }
    });
}

if (predictButton) {
    predictButton.addEventListener('click', async () => {
        if (predictButton.textContent === 'Reset') {
            resetToInitialState();
            return;
        }

        if (!currentFile) {
            showError("Please select an image file first.");
            return;
        }

        hideError();

        if (loader) loader.style.display = 'block';
        predictButton.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(API_ENDPOINT, { method: 'POST', body: formData });
            
            if (!response.ok) {
                let errorDetail = `HTTP error ${response.status}.`;
                try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (e) { /* Ignore */ }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            showResultsState(data); // Update UI to show results and "Reset" button

        } catch (error) {
            console.error("Error during prediction:", error);
            showError(`Error: ${error.message}. Check API server.`);
            // If error, button should probably revert to "Detect Objects" if no results are shown
            // or stay as "Reset" if a previous result was already there.
            // For simplicity, we re-enable it. The user can then reset or try again.
            predictButton.textContent = 'Detect Objects'; // Or keep as "Reset" and let user manually reset
        } finally {
            if (loader) loader.style.display = 'none';
            predictButton.disabled = false; // Re-enable button in both success/error cases
        }
    });
}

// Initialize button text correctly on page load
document.addEventListener('DOMContentLoaded', () => {
    if(predictButton) predictButton.textContent = 'Detect Objects';
});