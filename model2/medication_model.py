import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ===============================
# CONSTANTS
# ===============================
FEATURE_COLUMNS = [
    "iron_def","b12_def","vitd_def","calcium_def","severity",
    "magnesium_def","potassium_def","protein_def","zinc_def",
    "folate_def","omega3_def","electrolyte_imbalance",
    "general_malnutrition","vitamin_b6_def","copper_def",
    "selenium_def","iodine_def","choline_def",
    "gut_malabsorption","chronic_inflammation",
    "chronic_dehydration"
]

TARGET_COLUMNS = [
    "Medication_Brand_Names",
    "Medication_Text"
]

# ===============================
# MODEL CLASS
# ===============================
class MedicationRecommender:

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.X = self.df[FEATURE_COLUMNS]
        self.y = self.df[TARGET_COLUMNS]

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        self.knn = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean"
        )
        self.knn.fit(self.X_scaled)

    # ---------------------------
    # MAIN PREDICTION METHOD
    # ---------------------------
    def predict(self, deficiency_dict: dict) -> dict:
        """
        deficiency_dict: JSON from nutrition model
        """

        # Order input according to FEATURE_COLUMNS
        input_row = [
            deficiency_dict[col] for col in FEATURE_COLUMNS
        ]

        input_df = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)
        input_scaled = self.scaler.transform(input_df)

        distances, indices = self.knn.kneighbors(input_scaled)

        idx = indices[0][0]
        distance = float(distances[0][0])

        return {
            "Medication_Brand_Names": self.y.iloc[idx]["Medication_Brand_Names"],
            "Medication_Text": self.y.iloc[idx]["Medication_Text"]
        }
