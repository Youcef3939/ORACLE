import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from features import FEATURES_OUTPUT_PATH
from regime_detector import REGIME_OUTPUT_PATH

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
PREDICTIONS_PATH = os.path.join(PROCESSED_DIR, "predicted_regimes.parquet")

class PredictiveModel:
    def __init__(self):
        self.features = pd.read_parquet(FEATURES_OUTPUT_PATH)
        self.regimes = pd.read_parquet(REGIME_OUTPUT_PATH)
        self.regimes = self.regimes.reindex(self.features.index) 
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)

    def train_test_split(self, test_size=0.2):
        X = self.features
        y = self.regimes['regime']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained.")

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, preds))
        print("\nClassification Report:\n", classification_report(self.y_test, preds))
        return preds

    def save_predictions(self, preds):
        pred_df = pd.DataFrame({"predicted_regime": preds}, index=self.X_test.index)
        pred_df.to_parquet(PREDICTIONS_PATH)
        print(f"Predicted regimes saved to {PREDICTIONS_PATH}")

if __name__ == "__main__":
    pm = PredictiveModel()
    pm.train_test_split(test_size=0.2)
    pm.train()
    preds = pm.evaluate()
    pm.save_predictions(preds)
