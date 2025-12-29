import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
df=pd.read_csv('./updated_with_deficiency_columns.csv')

col_X=['fatigue', 'hair_loss', 'acidity', 'dizziness', 'muscle_pain',
       'numbness', 'vegetarian', 'iron_food_freq', 'dairy_freq',
       'sunlight_min', 'junk_food_freq', 'smoking', 'alcohol', 'hemoglobin',
       'ferritin', 'vitamin_b12', 'vitamin_d', 'calcium']
col_Y=['iron_def','b12_def', 'vitd_def', 'calcium_def', 'magnesium_def',
        'potassium_def', 'protein_def', 'zinc_def', 'folate_def', 'omega3_def',
        'electrolyte_imbalance', 'general_malnutrition', 'vitamin_b6_def',
        'copper_def', 'selenium_def', 'iodine_def',
        'choline_def', 'gut_malabsorption', 'chronic_inflammation',
       'chronic_dehydration', 'protein_quality_def','severity']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[col_X],df[col_Y],test_size=0.2,random_state=4)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
scaler.fit(X_test)
X_test_scaled=scaler.transform(X_test)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
dt=MultiOutputClassifier(DecisionTreeClassifier())
dt.fit(X_train_scaled,y_train)
y_pred=dt.predict(X_test_scaled)

#print(y_pred)
accuracy=[]
for i in range(len(col_Y)):
  accuracy.append(accuracy_score(y_test[col_Y[i]],y_pred[:,i]))

#print(accuracy)
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train_scaled,y_train['severity'])
y_pred=rf.predict(X_test_scaled)
#print(r2_score(y_test['severity'],y_pred))

joblib.dump(dt,"deficiency_pred.pkl")
joblib.dump(scaler,"scaling.pkl")
joblib.dump(rf,"serverity_pred.pkl")


dataf={
    "fatigue":2,
    "hair_loss":1,
    "acidity":1,
    "dizziness":2,
    "muscle_pain":1,
    "numbness":2,
    "vegetarian":0,
    "iron_food_freq":3,
    "dairy_freq":1,
    "sunlight_min":12,
    "junk_food_freq":1,
    "smoking":1,
    "alcohol":1,
    "hemoglobin":15.8,
    "ferritin":21.8,
    "vitamin_b12":169.8,
    "vitamin_d":4,
    "calcium":12.5
}

TestDf=pd.DataFrame(dataf,index=[0])
TestDf_Scaled=scaler.transform(TestDf)
diction={}
i=0
y_pred=dt.predict(TestDf_Scaled)
y_pred_serverity=rf.predict(TestDf_Scaled)
print(y_pred)
colY=['iron_def','b12_def', 'vitd_def', 'calcium_def', 'magnesium_def',
        'potassium_def', 'protein_def', 'zinc_def', 'folate_def', 'omega3_def',
        'electrolyte_imbalance', 'general_malnutrition', 'vitamin_b6_def',
        'copper_def', 'selenium_def', 'iodine_def',
        'choline_def', 'gut_malabsorption', 'chronic_inflammation',
       'chronic_dehydration', 'protein_quality_def','severity']
for ele in colY:
  diction[ele]=y_pred[0][i]
  i=i+1
diction['severity']=y_pred_serverity[0]
print(diction)