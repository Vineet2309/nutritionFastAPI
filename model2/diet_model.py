import pandas as pd
import joblib

# ===============================
# CONSTANTS
# ===============================
INPUT_COLS = [
    "fatigue","hair_loss","acidity","dizziness","muscle_pain","numbness",
    "vegetarian","iron_food_freq","dairy_freq","sunlight_min","junk_food_freq",
    "smoking","alcohol","hemoglobin","ferritin","vitamin_b12","vitamin_d",
    "calcium","iron_def","b12_def","vitd_def","calcium_def","severity",
    "magnesium_def","potassium_def","protein_def","zinc_def","folate_def",
    "omega3_def","electrolyte_imbalance","general_malnutrition",
    "vitamin_b6_def","copper_def","selenium_def","iodine_def","vitamin_a_def",
    "choline_def","gut_malabsorption","chronic_inflammation",
    "chronic_dehydration","protein_quality_def"
]

OUTPUT_COLS = [
    "Diet_Additions",
    "Nutrient_Requirements",
    "Vegetarian_Food_Mapping",
    "Mandatory_Diet_Changes"
]

# ===============================
# MODEL CLASS
# ===============================
class DietRecommendationModel:

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        outputs_path: str
    ):
        self.knn = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.outputs = joblib.load(outputs_path)

    # ---------------------------
    # MAIN PREDICTION METHOD
    # ---------------------------
    def predict(self, input_data: dict) -> dict:
        """
        input_data: merged Input A + Input B (dict)
        """

        # Ensure correct column order
        input_row = [input_data[col] for col in INPUT_COLS]

        input_df = pd.DataFrame([input_row], columns=INPUT_COLS)
        input_scaled = self.scaler.transform(input_df)

        distances, indices = self.knn.kneighbors(input_scaled)
        idx = indices[0][0]

        return {
            "Diet_Additions": self.outputs.iloc[idx]["Diet_Additions"],
            "Nutrient_Requirements": self.outputs.iloc[idx]["Nutrient_Requirements"],
            "Vegetarian_Food_Mapping": self.outputs.iloc[idx]["Vegetarian_Food_Mapping"],
            "Mandatory_Diet_Changes": self.outputs.iloc[idx]["Mandatory_Diet_Changes"]
        }
