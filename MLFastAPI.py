#pip install fastapi uvicorn joblib pydantic
#python -m uvicorn MLFastAPI:app --reload --host 0.0.0.0 --port 8000
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from model2.medication_model import MedicationRecommender
from model2.diet_model import DietRecommendationModel
app=FastAPI()

scaling=joblib.load('./model1/scaling.pkl')
serverity=joblib.load('./model1/serverity_pred.pkl')
deficiency=joblib.load('./model1/deficiency_pred.pkl')

class nutrition_data(BaseModel):
    fatigue:int
    hair_loss:int
    acidity:int
    dizziness:int
    muscle_pain:int
    numbness:int
    vegetarian:int
    iron_food_freq:int
    dairy_freq:int
    sunlight_min:int
    junk_food_freq:int
    smoking:int
    alcohol:int
    hemoglobin:float
    ferritin:float
    vitamin_b12:float
    vitamin_d:float
    calcium:float

@app.post("/predict")
def predict(data:nutrition_data):
    input_data=dict(data)
    df=pd.DataFrame([data.dict()])
    data_scaled=scaling.transform(df)
    prediction=deficiency.predict(data_scaled)
    pred_serveri=serverity.predict(data_scaled)
    diction={}
    i=0
    colY=['iron_def','b12_def', 'vitd_def', 'calcium_def', 'magnesium_def',
        'potassium_def', 'protein_def', 'zinc_def', 'folate_def', 'omega3_def',
        'electrolyte_imbalance', 'general_malnutrition', 'vitamin_b6_def',
        'copper_def', 'selenium_def', 'iodine_def',
        'choline_def', 'gut_malabsorption', 'chronic_inflammation',
       'chronic_dehydration', 'protein_quality_def','severity']
    for ele in colY:
        diction[ele]=int(prediction[0][i])
        i=i+1
    diction['severity']=int(pred_serveri)

    Medirecommender = MedicationRecommender(
        csv_path="./model2/deficiency_with_medications.csv"
    )
    MediRecomm=Medirecommender.predict(diction)

    DietRecommender=DietRecommendationModel(
        model_path="model2/diet_knn_model.pkl",
        scaler_path="model2/scaler.pkl",
        outputs_path="model2/outputs.pkl"
    )
    total_data=input_data|diction
    total_data["vitamin_a_def"]=0
    DietRecomm=DietRecommender.predict(total_data)

    return {
        "prediction1":diction,
        "prediction2":MediRecomm,
        "prediction3":DietRecomm
    }
 