from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            filename = 'normal-01'  # Set the standard filename
            extension = os.path.splitext(file.filename)[1]  # Get the file extension from the uploaded file
            filename += extension

            # Define the path to the testimages directory
            save_path = os.path.join(app.root_path, '..', 'testimages', filename)

            # Check if a file with the same name already exists and delete it
            if os.path.exists(save_path):
                os.remove(save_path)

            file.save(save_path)  # Save the file to the desired directory
            # Open the image directly without using matplotlib
            img = Image.open(file.stream)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return send_file(buffer, mimetype='image/png')
        except IOError:
            return jsonify({'error': 'Invalid image file'})


if __name__ == '__main__':
    app.run(debug=True)
