from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from utils import predict_mask
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run prediction
    mask = predict_mask(filepath)  # Should return a 2D numpy array (H, W)

    # Save result
    mask_path = os.path.join(UPLOAD_FOLDER, 'mask.png')
    cv2.imwrite(mask_path, mask)

    return send_file(mask_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)