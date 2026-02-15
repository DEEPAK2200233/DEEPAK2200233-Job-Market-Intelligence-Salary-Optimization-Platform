from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# =========================
# LOAD MODEL FILES
# =========================
model = joblib.load("salary_model.pkl")
role_encoder = joblib.load("encoder_role.pkl")
location_encoder = joblib.load("encoder_location.pkl")
work_mode_encoder = joblib.load("encoder_work_mode.pkl")
employment_encoder = joblib.load("encoder_employment_type.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# =========================
# HOME ROUTE
# =========================
@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}


# =========================
# PREDICT ROUTE
# =========================
@app.post("/predict")
def predict_salary(
    role: str,
    location: str,
    work_mode: str,
    employment_type: str,
    experience_years: int,
    skills: str
):
    # Prepare input
    df = pd.DataFrame([{
        "role": role,
        "location": location,
        "work_mode": work_mode,
        "employment_type": employment_type,
        "experience_years": experience_years,
        "skills": skills.lower()
    }])

    # Create empty feature row
    X = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Encode categorical
    def normalize(text):
        return " ".join(word.capitalize() for word in text.strip().lower().split())


    role = normalize(role)
    location = normalize(location)
    work_mode = normalize(work_mode)
    employment_type = normalize(employment_type)

    X["role"] = role_encoder.transform([role])[0]
    X["location"] = location_encoder.transform([location])[0]
    X["work_mode"] = work_mode_encoder.transform([work_mode])[0]
    X["employment_type"] = employment_encoder.transform([employment_type])[0]

    # Numeric
    X["experience_years"] = df["experience_years"][0]

    # Skills
    for col in feature_columns:
        if col.startswith("has_"):
            skill_name = col.replace("has_", "").replace("_", " ")
            if skill_name in df["skills"][0]:
                X[col] = 1

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    return {
        "predicted_salary_lpa": round(float(prediction), 2)
    }
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
