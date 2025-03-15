from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import legacy as optimizers_legacy
from sklearn.preprocessing import MinMaxScaler
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

IMG_SIZE = (32, 32)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

churn_model = joblib.load('churn_model.pkl')
movie_review_model = joblib.load('movie_review_model.joblib')
iris_model = joblib.load('iris_model.pkl')
iris_label_encoder = joblib.load('label_encoder.pkl')
kmeans = joblib.load('kmeans_model.pkl')
mall_scaler = joblib.load('scaler_model.pkl')
image_model = load_model('fake_vs_real_images.h5')

try:
    air_passenger_model = load_model(
        'air_passenger_lstm.h5',
        custom_objects={
            'mse': MeanSquaredError(),
            'MeanSquaredError': MeanSquaredError(),
            'Adam': optimizers_legacy.Adam
        }
    )
    air_passenger_scaler = joblib.load('air_passenger_scaler.pkl')
except FileNotFoundError:
    class FallbackScaler:
        def __init__(self, data_min=104.0, data_max=622.0):
            self.data_min_ = np.array([data_min])
            self.data_max_ = np.array([data_max])
            self.scale_ = np.array([1/(data_max - data_min)])
            self.min_ = 0.0
            
        def transform(self, X):
            return (X - self.data_min_) * self.scale_
            
        def inverse_transform(self, X):
            return X/self.scale_ + self.data_min_

    air_passenger_scaler = FallbackScaler()
    air_passenger_model = None
    print("Warning: Using fallback scaler - predictions may be inaccurate")


@app.route('/mall_customer_model', methods=['POST'])
def predict_mall():
    try:
        data = request.get_json()
        annual_income = data['annual_income']
        spending_score = data['spending_score']
        
        new_data = np.array([[annual_income, spending_score]])
        new_data_scaled = mall_scaler.transform(new_data)
        predicted_cluster = kmeans.predict(new_data_scaled)
        
        return jsonify({'predicted_cluster': int(predicted_cluster[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/iris_model', methods=['POST'])
def predict_iris():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = iris_model.predict(features)
    predicted_species = iris_label_encoder.inverse_transform(prediction)
    return jsonify({'species': predicted_species[0]})

@app.route('/churn_model', methods=['POST'])
def predict_churn():
    data = request.get_json()
    input_data = pd.DataFrame(data, index=[0])
    prediction = churn_model.predict(input_data)
    prediction_proba = churn_model.predict_proba(input_data)[:, 1]
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0])
    })

@app.route('/movie_review_model', methods=['POST'])
def predict_movie():
    review_text = request.json['review']
    prediction = movie_review_model.predict([review_text])
    return jsonify({'prediction': 'Positive' if prediction[0] == 1 else 'Negative'})

@app.route('/air_passenger/forecast', methods=['POST'])
def predict_air_passengers():
    try:
        if not air_passenger_model:
            return jsonify({'error': 'Air passenger model not loaded'}), 500
            
        data = request.get_json()
        history = data.get('history', [])
        steps = data.get('steps', 12)
        
        if len(history) != 18:
            return jsonify({
                'error': 'Need exactly 18 historical values',
                'received': len(history)
            }), 400
        # all values in the history are within a valid range
        if any(x < air_passenger_scaler.data_min_[0] or 
               x > air_passenger_scaler.data_max_[0] for x in history):
            return jsonify({
                'error': f'Values must be between {air_passenger_scaler.data_min_[0]} and {air_passenger_scaler.data_max_[0]}'
            }), 400
        raw_seq = np.array(history).reshape(-1, 1).astype(float)
        scaled_seq = air_passenger_scaler.transform(raw_seq).flatten()

        predictions = []
        current_seq = scaled_seq.copy()
        
        for _ in range(steps):
            # keep input shape same as training
            X = current_seq.reshape(1, 18, 1)
            pred = air_passenger_model.predict(X, verbose=0)[0][0]
            # ensure valid scaled values
            pred = np.clip(pred, 0, 1)  
            predictions.append(pred)
            # shift sequence
            current_seq = np.append(current_seq[1:], pred)

        predictions = air_passenger_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1))
            
        return jsonify({
            'forecast': predictions.flatten().round(1).tolist(),
            'input_range': {
                'min': float(air_passenger_scaler.data_min_[0]),
                'max': float(air_passenger_scaler.data_max_[0])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return tf.expand_dims(image / 255.0, axis=0)


@app.route('/image_classification', methods=['POST'])
def image_classification():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Empty file submission'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = preprocess_image(filepath)
        prediction = image_model.predict(image, verbose=0)[0][0]
        
        return jsonify({
            'prediction': 'fake' if prediction > 0.5 else 'real',
            'confidence': round(float(prediction if prediction > 0.5 else 1 - prediction), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)