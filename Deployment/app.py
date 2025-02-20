from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
app = Flask(__name__)

churn_model = joblib.load('churn_model.pkl')
movie_review_model =  joblib.load('movie_review_model.joblib')
iris_model = joblib.load('iris_model.pkl')
iris_label_encoder = joblib.load('label_encoder.pkl')  

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler_model.pkl')

@app.route('/mall_customer_model', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        annual_income = data['annual_income']
        spending_score = data['spending_score']
        
        new_data = np.array([[annual_income, spending_score]])
        
        new_data_scaled = scaler.transform(new_data)
        
        predicted_cluster = kmeans.predict(new_data_scaled)
        
        return jsonify({'predicted_cluster': int(predicted_cluster[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/iris_model', methods=['POST'])
def predict_iris_model():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)  
    prediction = iris_model.predict(features)
    predicted_species = iris_label_encoder.inverse_transform(prediction)

    return jsonify({'species': predicted_species[0]})

@app.route('/churn_model', methods=['POST'])
def predict_churn_model():
    data = request.get_json()
    
    input_data = pd.DataFrame(data, index=[0])

    prediction = churn_model.predict(input_data)
    prediction_proba = churn_model.predict_proba(input_data)[:, 1]

    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0])
    })

@app.route('/movie_review_model', methods=['POST'])
def predict_movie_review():
    review_text = request.json['review']
    
    prediction = movie_review_model.predict([review_text])
    
    return jsonify({'prediction': 'Positive' if prediction[0] == 1 else 'Negative'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
