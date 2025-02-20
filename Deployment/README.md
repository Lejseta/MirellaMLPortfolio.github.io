# Docker Usage
Open terminal inside the project folder and paste the command:
```
docker build -t flaskapp .
```
After the container has been builded, run the container with this command: 
```
docker run -p 5000:5000 flaskapp
```
----
# Endpoints
## [POST] Movie Review Prediction
 Endpoint:` localhost:5000/movie_review_model `<br>
Sample Payload:
```json
{
  "review": "This movie was fantastic and I loved every minute of it!"
}
```

## [POST] Churn Prediction
Endpoint: `http://127.0.0.1:5000/churn_model` <br>
Sample Payload:
```json
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
```
## [POST] Iris Flower Prediction
Endpoint:` http://localhost:5000/iris_model`<br>
Sample payload: 
```json
{
  "features": [5.1, 3.5, 1.4, 0.2,3]
}
```

## [POST] Mall Customer Segmentation Prediction
Endpoint: `localhost:5000/mall_customer_model` <br>
Sample payload:
```json
{
  "annual_income": 60,
  "spending_score": 45
}
```
