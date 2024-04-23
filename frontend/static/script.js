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
        reader.readAsArrayBuffer(file);
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

// Function to send image to server or process it further
// Example JavaScript to send the file to the Flask endpoint
function sendToServer(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.blob())
    .then(imageBlob => {
        // Assuming you want to display the image response
        const imageObjectURL = URL.createObjectURL(imageBlob);
        document.getElementById('outputImage').src = imageObjectURL;
        document.getElementById('outputImage').hidden = false;
    })
    .catch(error => console.error('Error uploading the image:', error));
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
