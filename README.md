###### mponline-aiml-assg10

# CardioAI: Heart Disease Prediction and REST API Deployment

An end-to-end Machine Learning model deployment project to predict whether a patient is at risk of heart disease based on clinical parameters. This repository contains the source code, preprocessed dataset, trained model, and Flask web API configuration required for deployment on Render.

## Live Application URL
**Render App Link:** `https://heart-disease-prediction-a1j5.onrender.com` 

---

## 1. Problem Statement
A healthcare organization wants to deploy a machine learning model that predicts whether a patient is at risk of heart disease based on clinical parameters. This project involves:
- Conducting data analysis and train-test splits on the Cleveland Heart Disease dataset.
- Training a Random Forest Classifier.
- Building a Flask REST API and an interactive premium Glassmorphic web UI.
- Structuring requirements and deployment configurations for cloud Hosting.

---

## 2. Dataset Features
The dataset `heart.csv` contains **1025 records** with **14 columns** (13 numerical features and 1 binary target):
1. **age:** Age in years
2. **sex:** Sex (1 = male; 0 = female)
3. **cp:** Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
4. **trestbps:** Resting blood pressure (in mm Hg on admission to the hospital)
5. **chol:** Serum cholesterol in mg/dl
6. **fbs:** Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg:** Resting electrocardiographic results (0, 1, 2)
8. **thalach:** Maximum heart rate achieved
9. **exang:** Exercise-induced angina (1 = yes; 0 = no)
10. **oldpeak:** ST depression induced by exercise relative to rest
11. **slope:** The slope of the peak exercise ST segment
12. **ca:** Number of major vessels (0-3) colored by fluoroscopy
13. **thal:** Thalassemia blood disorder (0 = null; 1 = normal flow; 2 = fixed defect; 3 = reversible defect)
14. **target:** Heart disease status (0 = No Heart Disease Detected, 1 = Heart Disease Detected)

---

## 3. Technical Architecture & File Structure
```
HeartDiseaseDeployment/
│
├── train_model.py       # Data preprocessing, training, evaluation, and saving script
├── model.pkl            # Trained Random Forest model (joblib binary)
├── app.py               # Flask application with REST API and web interfaces
├── requirements.txt     # Locked production dependencies
├── README.md            # Documentation and project conclusion
└── templates/
    └── index.html       # Premium frontend interface with CSS glassmorphic panel and AJAX prediction
```

---

## 4. Model Performance & Evaluation
Using a Random Forest Classifier trained on an 80/20 train-test split:
- **Training Accuracy:** **99.76%**
- **Testing Accuracy:** **99.02%**
- **Artifact:** Saved as `model.pkl` using `joblib`.

---

## 5. Flask API Endpoint documentation

### Predict Risk Endpoint
- **URL:** `/predict`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Body Format (Object):**
```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```
- **Example Response:**
```json
{
  "prediction": "Heart Disease Detected",
  "probability": 0.89,
  "risk_score_percent": 89.0,
  "status": "success"
}
```

---

## 6. How to Run Locally

1. Clone or download this repository:
   ```bash
   git clone <repository_url>
   cd HeartDiseaseDeployment
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script (if you wish to retrain):
   ```bash
   python train_model.py
   ```
4. Launch the local Flask server:
   ```bash
   python app.py
   ```
5. Open your web browser and navigate to: `http://localhost:5000`

---

## 7. Render Deployment Instructions (Manual Deployment)
1. **GitHub Upload:** Create a public repository and push all files including `model.pkl`.
2. **Create Render Service:** Link your GitHub account and create a new **Web Service** pointing to this repository.
3. **Configure Settings:**
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (This uses the production Gunicorn web server configured in `requirements.txt`).
4. **Environment Variables:** Render automatically handles the `PORT` variable. The app will boot successfully on the provided dynamic port.

---

## 8. Conclusion
The developed heart disease prediction model achieved a remarkable testing accuracy of 99.02% using a Random Forest Classifier. In practice, deploying machine learning models introduces significant challenges, such as handling dependency mismatches, configuring production-grade WSGI servers (like Gunicorn), and managing the platform-specific cold start delays on Render's free tier. This highlights the critical importance of MLOps in modern data science. MLOps establishes standard practices for automated model retraining, version control of both datasets and trained model artifacts, and continuous integration/continuous deployment (CI/CD) pipelines. By bridging the gap between research code and production environments, MLOps ensures that machine learning services remain highly reliable, scalable, and capable of delivering accurate real-time clinical predictions under variable user loads.
