from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import io

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
