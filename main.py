from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
import tensorflow as tf
import librosa
from torchvision import models

# Initialize Flask App
app = Flask(__name__)
app.config['deepfake_detection'] = ''
os.makedirs(app.config['deepfake_detection'], exist_ok=True)

# Load Pre-trained Models
image_model = tf.keras.models.load_model('models/xception_deepfake.h5')  # CNN Model for images
video_model = torch.load('models/mesonet.pth', map_location=torch.device('cpu'))  # PyTorch Model for videos
audio_model = torch.load('models/wavenet_audio.pth', map_location=torch.device('cpu'))  # PyTorch Model for audio


# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))  # Resize to match model input size
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Function to preprocess a video (extract frames)
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames = torch.Tensor(frames) / 255.0  # Normalize
    return frames


# Function to preprocess an audio file
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = torch.Tensor(mfccs)
    return mfccs


# Deepfake detection for images
def detect_image(image_path):
    img = preprocess_image(image_path)
    prediction = image_model.predict(img)
    confidence = float(prediction[0][0])
    return {'result': 'Fake' if confidence < 0.5 else 'Real', 'confidence': confidence}


# Deepfake detection for videos
def detect_video(video_path):
    frames = preprocess_video(video_path)
    prediction = video_model(frames.unsqueeze(0))
    confidence = prediction.item()
    return {'result': 'Fake' if confidence < 0.5 else 'Real', 'confidence': confidence}


# Deepfake detection for audio
def detect_audio(audio_path):
    features = preprocess_audio(audio_path)
    prediction = audio_model(features.unsqueeze(0))
    confidence = prediction.item()
    return {'result': 'Fake' if confidence < 0.5 else 'Real', 'confidence': confidence}


# Flask Routes
@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepfake Detection</title>
        <style>
            body { text-align: center; font-family: Arial, sans-serif; }
            .container { margin-top: 50px; }
            .upload-section { margin: 20px; }
        </style>
    </head>
    <body>
        <h1>Deepfake Detection System</h1>
        <div class="container">
            <label>Select Media Type:</label>
            <select id="mediaType">
                <option value="image">Image</option>
                <option value="video">Video</option>
                <option value="audio">Audio</option>
            </select>
            <br><br>
            <input type="file" id="fileInput">
            <button onclick="uploadFile()">Upload & Detect</button>
            <p id="result"></p>
        </div>
        <script>
            function uploadFile() {
                let mediaType = document.getElementById("mediaType").value;
                let fileInput = document.getElementById("fileInput").files[0];
                if (!fileInput) {
                    alert("Please select a file");
                    return;
                }
                let formData = new FormData();
                formData.append("file", fileInput);
                fetch(`/detect/${mediaType}`, {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById("result").innerText = `Result: ${data.result} (Confidence: ${data.confidence.toFixed(2)})`;
                })
                .catch(error => console.error("Error:", error));
            }
        </script>
    </body>
    </html>
    ''')


@app.route('/detect/<media_type>', methods=['POST'])
def detect(media_type):
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if media_type == 'image':
        result = detect_image(file_path)
    elif media_type == 'video':
        result = detect_video(file_path)
    elif media_type == 'audio':
        result = detect_audio(file_path)
    else:
        return jsonify({'error': 'Invalid media type'}), 400

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
