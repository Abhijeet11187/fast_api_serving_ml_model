from fastapi import FastAPI
from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Literal,Annotated
import pickle
import pandas as pd
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
# Import the ML model

with open('model/model.pkl','rb') as f:
    model=pickle.load(f)
    
app=FastAPI()

MODEL_VERSION='1.1.0'


# Pydantic model to validate incoming data



@app.get("/")
def home():
    return {'message':"Insurance Premium Prediction API"}

@app.get("/health")
def health_check():
    return {
        "status":"OK",
        "version":MODEL_VERSION,
        "model_loaded":model is not None
    } 

@app.post("/predict")
def predict_premium(data:UserInput):
    
    input_df=pd.DataFrame([{
        "bmi":data.bmi,
        "age_group":data.age_group,
        "lifestyle_risk":data.lifestyle_risk,
        "city_tier":data.city_tier,
        "income_lpa":data.income_lpa,
        "occupation":data.occupation,
    }])
    
    prediction=model.predict(input_df)[0]
    
    return JSONResponse(
        status_code=200,
        content={'predicted_category':prediction}
    )