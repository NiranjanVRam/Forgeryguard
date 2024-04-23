document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onerror = () => {
        console.error("Error reading file");
        alert("Failed to read file.");
    };

    if (file.type === "image/tiff" || file.name.endsWith('.tif') || file.name.endsWith('.tiff')) {
        reader.onload = function(e) {
            try {
                const buffer = e.target.result;
                const tiff = new Tiff({buffer: buffer});
                const canvas = tiff.toCanvas();
                const previewContainer = document.getElementById('imagePreview');
                previewContainer.innerHTML = ''; // Clear existing content
                previewContainer.appendChild(canvas); // Add the canvas
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
