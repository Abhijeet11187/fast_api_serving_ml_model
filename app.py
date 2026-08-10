from fastapi import FastAPI
from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Literal,Annotated
import pickle
import pandas as pd
from fastapi.responses import JSONResponse

# Import the ML model

with open('model/model.pkl','rb') as f:
    model=pickle.load(f)
    
app=FastAPI()

MODEL_VERSION='1.1.0'

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

# Pydantic model to validate incoming data

class UserInput(BaseModel):
    age:Annotated[int,Field(...,gt=0,lt=120,description="Age of the user")]
    weight:Annotated[float,Field(...,gt=0,description="Wight of the user")]
    height:Annotated[float,Field(...,gt=0,lt=2.5,description="Height of the user")]
    income_lpa:Annotated[float,Field(...,gt=0,description="Annual Salary of the user in LPA")]
    smoker:Annotated[bool,Field(...,description="Is user a smoker or not")]
    city:Annotated[str,Field(...,description="City of the user")]
    occupation:Annotated[Literal['retired','freelancer','student','goverment_job','business_owner','unemployed','private_job'],Field(...,description="Users Occupation")]
    
    @field_validator('city')
    @classmethod
    def normalize_city(cls,v:str)->str:
        v=v.strip().title()
        return v
    
    @computed_field
    @property
    def bmi(self)->float:
        return self.weight/(self.height**2)
    
    @computed_field
    @property
    def lifestyle_risk(self)->str:
        if self.smoker and self.bmi>30:
            return "high"
        elif self.smoker and self.bmi>27:
            return "medium"
        else:
            return "low"
        
    @computed_field
    @property
    def age_group(self)->str:
        if self.age<25:
            return  "young"
        elif self.age<45:
            return "adult"
        elif self.age<60:
            return "middle_aged"
        else:
            return "senior"
        
    @computed_field
    @property
    def city_tier(self)->int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
       

@app.get("/")
def home():
    return {'message':"Insurance Premium Prediction API"}

@app.get("/health")
def health_check():
    return {
        "status":"OK",
        "version":MODEL_VERSION
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