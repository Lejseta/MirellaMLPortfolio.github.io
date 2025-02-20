# [POST] Movie Review Prediction
Endpoint: localhost:5000/movie_review_model
Sample Payload:
{
  "review": "This movie was fantastic and I loved every minute of it!"
}

# [POST] Churn Prediction
Endpoint: http://127.0.0.1:5000/churn_model
Sample Payload:
{
  "SeniorCitizen": 1,
  "tenure": 12,
  "MonthlyCharges": 50.5,
  "TotalCharges": 600,
  "gender": "Female",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}

# [POST] Iris Flower Prediction
Endpoint: http://localhost:5000/iris_model
Sample payload: 
{
  "features": [5.1, 3.5, 1.4, 0.2,3]
}

# [POST] Mall Customer Segmentation Prediction
Endpoint:localhost:5000/mall_customer_model
Sample payload:
{
  "annual_income": 60,
  "spending_score": 45
}
