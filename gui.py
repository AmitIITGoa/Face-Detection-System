from flask import Flask, render_template_string, request, jsonify
import pickle
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# HTML template with embedded JavaScript
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Age and Gender Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 0;
        }
        h1 {
            margin-top: 20px;
        }
        #video, #canvas {
            border: 1px solid black;
        }
        #result {
            margin-top: 20px;
            color :  green ;
            boarder : 1px solid black;
        }
    </style>
</head>
<body>
    <h1>Age and Gender Prediction</h1>
    <div>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
    </div>
    <div id="result" >
        <p>Gender: <span id="gender"></span></p>
        <p>Age: <span id="age"></span></p>
    </div>
    <script>
        (function() {
            var video = document.getElementById('video');
            var canvas = document.getElementById('canvas');
            var context = canvas.getContext('2d');
            var genderElement = document.getElementById('gender');
            var ageElement = document.getElementById('age');

            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
                    video.srcObject = stream;
                    video.play();
                });
            }

            function captureAndPredict() {
                context.drawImage(video, 0, 0, 640, 480);
                var dataURL = canvas.toDataURL('image/png');
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ image: dataURL })
                })
                .then(response => response.json())
                .then(data => {
                    genderElement.textContent = data.gender;
                    ageElement.textContent = data.age;
                })
                .catch(error => console.error('Error:', error));
            }

            setInterval(captureAndPredict, 1500); // Capture and predict every 2 seconds
        })();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']
    img_data = img_data.split(',')[1]
    img_data = base64.b64decode(img_data)
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize the image
    img = img.reshape(1, 128, 128, 1)
    
    prediction = model.predict(img)
    gender = 'Male' if prediction[0][0] <= 0.5 else 'Female'
    age = int(prediction[1][0])
    
    return jsonify({'gender': gender, 'age': age})

if __name__ == '__main__':
    app.run(debug=True)
