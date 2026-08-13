# 🏥 Insurance Premium Category Predictor

An end-to-end machine learning application that predicts a user's **insurance premium category** from basic personal and lifestyle details. The project wraps a trained scikit-learn model behind a **FastAPI** REST API with strict **Pydantic** input validation, and ships a **Streamlit** frontend for interactive use.

---

## 📖 Overview

Given raw inputs like age, weight, height, income, smoking status, city, and occupation, the app derives risk-relevant features on the fly (BMI, age group, lifestyle risk, city tier) and feeds them to a pre-trained classification model to predict the applicant's insurance premium category.

**🔄 Live flow:**

```
Streamlit UI  →  POST /predict  →  Pydantic validation + feature derivation  →  scikit-learn model  →  JSON response
```

---

## ✨ Features

- 🚀 **REST API** built with FastAPI, including `/`, `/health`, and `/predict` endpoints
- 🧮 **Automatic feature engineering** via Pydantic `computed_field`s — BMI, age group, lifestyle risk, and city tier are derived from raw inputs instead of being supplied by the client
- ✅ **Strict input validation** using `Annotated` types, `Field` constraints, and a custom `field_validator` to normalize city names
- 🤖 **Pre-trained scikit-learn model** loaded once at startup and served via a lightweight prediction wrapper
- 🖥️ **Streamlit frontend** for a no-code way to test predictions in the browser
- 🏷️ **Model versioning** exposed through the `/health` endpoint

---

## 🛠️ Tech Stack

| Layer               | Technology                     |
|---------------------|---------------------------------|
| ⚡ API framework        | FastAPI, Uvicorn                |
| 🧾 Data validation       | Pydantic v2                     |
| 🧠 ML model              | scikit-learn (pickled model)    |
| 🎨 Frontend               | Streamlit                       |
| 📊 Data handling          | pandas, NumPy                   |

---

## 📂 Project Structure

```
fast_api_serving_ml_model/
├── app.py                  # FastAPI application and route definitions
├── front_end.py             # Streamlit frontend that consumes the API
├── requirement.txt          # Python dependencies
├── config/
│   └── city_tier.py         # Tier-1 / Tier-2 city lists used for city_tier feature
├── model/
│   ├── model.pkl             # Pre-trained scikit-learn classifier
│   └── predict.py            # Model loading and prediction logic
└── schema/
    └── user_input.py         # Pydantic schema, validators, and computed features
```

---

## ⚙️ How It Works

1. 📥 The client (Streamlit UI or any HTTP client) sends raw applicant details to `POST /predict`.
2. 🧮 `schema/user_input.py` validates the payload and automatically computes:
   - **`bmi`** — `weight / height²`
   - **`age_group`** — `young` / `adult` / `middle_aged` / `senior`
   - **`lifestyle_risk`** — `low` / `medium` / `high`, based on smoking status and BMI
   - **`city_tier`** — `1`, `2`, or `3`, based on `config/city_tier.py` lookup lists
3. 🔗 `app.py` assembles the derived features into a dictionary and passes them to `model/predict.py`.
4. 🤖 `predict.py` loads `model/model.pkl` once at import time and returns the predicted category for the request.
5. 📤 The API responds with the predicted category as JSON.

---

## 📡 API Reference

### `GET /`
Health/welcome message.

```json
{ "message": "Insurance Premium Prediction API" }
```

### `GET /health` 💓
Reports API status, loaded model version, and whether the model loaded successfully.

```json
{
  "status": "OK",
  "version": "1.1.0",
  "model_loaded": true
}
```

### `POST /predict` 🎯
Predicts the insurance premium category for a given applicant.

**Request body**

| Field         | Type    | Constraints                                                                                     |
|---------------|---------|---------------------------------------------------------------------------------------------------|
| `age`          | int     | `0 < age < 120`                                                                                    |
| `weight`        | float   | `> 0` (kg)                                                                                          |
| `height`         | float   | `0 < height < 2.5` (meters)                                                                        |
| `income_lpa`      | float   | `> 0` (annual income, in Lakhs Per Annum)                                                          |
| `smoker`           | bool    | —                                                                                                    |
| `city`               | str     | Normalized to title case                                                                            |
| `occupation`          | str     | One of: `retired`, `freelancer`, `student`, `goverment_job`, `business_owner`, `unemployed`, `private_job` |

**Example request**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "age": 35,
        "weight": 72,
        "height": 1.75,
        "income_lpa": 12,
        "smoker": false,
        "city": "Pune",
        "occupation": "private_job"
      }'
```

**Example response**

```json
{
  "predicted_category": "Medium"
}
```

> ⚠️ Note: `occupation` currently accepts `goverment_job` (as spelled in the source schema) rather than `government_job`.

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.10+
- pip

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Abhijeet11187/fast_api_serving_ml_model.git
cd fast_api_serving_ml_model
```

### 2️⃣ Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirement.txt
```

### 3️⃣ Run the FastAPI server
```bash
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`, with interactive Swagger docs at `http://127.0.0.1:8000/docs`. 📘

### 4️⃣ Run the Streamlit frontend (in a separate terminal)
```bash
streamlit run front_end.py
```
This opens a browser UI where you can fill in applicant details and get predictions without calling the API directly. 🖱️

---


## 📄 License

This project is for educational and demonstration purposes. Feel free to fork and extend.
This project is for educational purposes. Feel free to use and adapt it for your own learning.



## ⭐ If you found this helpful

Give this repository a star ⭐ 
