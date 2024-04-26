document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onerror = () => {
        console.error("Error reading file");
        alert("Failed to read file.");
    };

    // Check if the file is a TIFF image
    if (file.type === "image/tiff" || file.name.endsWith('.tif') || file.name.endsWith('.tiff')) {
        reader.onload = function(e) {
            try {
                const buffer = e.target.result;
                const tiff = new Tiff({buffer: buffer});
                const canvas = tiff.toCanvas();
                const previewContainer = document.getElementById('imagePreview');
                previewContainer.innerHTML = ''; // Clear existing content
                previewContainer.appendChild(canvas); // Add the TIFF image as a canvas
            } catch (error) {
                console.error("Error processing TIFF image", error);
                alert("Failed to process TIFF image.");
            }
        };
        reader.readAsArrayBuffer(file); // Read the file as an ArrayBuffer
    } else {
        reader.onload = function(e) {
            const previewImage = document.getElementById('previewImage');
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            document.querySelector("#imagePreview p").style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('cancelProcessingBtn').addEventListener('click', function() {
    fetch('/cancel_process', { method: 'POST' })
        .then(response => window.location.reload())
        .catch(error => window.location.reload()); // Reload even if there's an error sending the cancel request
});


function showLoadingDialog() {
    const loadingDialog = document.getElementById('loadingDialog');
    loadingDialog.style.display = 'block'; // Show the loading dialog
}

function hideLoadingDialog() {
    const loadingDialog = document.getElementById('loadingDialog');
    loadingDialog.style.display = 'none'; // Hide the loading dialog
}

// Function to send image to server or process it further
// Example JavaScript to send the file to the Flask endpoint
function sendToServer(file) {
    showLoadingDialog(); // Show loading dialog when processing starts
    const outputImage = document.getElementById('outputImage');
    const formData = new FormData();
    formData.append('file', file);

    // Clear previous images and indicate loading
    outputImage.src = "";
    outputImage.hidden = true;
    outputImage.alt = "Loading...";  // Indicate that loading is happening

    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        hideLoadingDialog(); // Hide loading dialog when processing ends
        if (data.heatmap) {
            outputImage.src = data.heatmap;
            outputImage.hidden = false;
            outputImage.alt = "Processed output image";  // Reset the alt text
            data.heatmap='';
        }
    })
    .catch(error => {
        console.error('Error uploading the image:', error);
        alert('Error processing the image. Please try again.');
        outputImage.alt = "Failed to load image";  // Provide feedback on error
    });
}

// Event listener for the process button
document.getElementById('processBtn').addEventListener('click', function() {
    const fileInput = document.getElementById('imageUpload');
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        sendToServer(file); // Function call to handle the file
    } else {
        alert('Please select an image file first.');
    }
});
