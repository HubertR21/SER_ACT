import argparse
import pickle
import warnings

import numpy as np
import opensmile


class EmotionPredictor:
    def __init__(self, *, model_path: str, scaler_path: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with open(model_path or r'..\models\xgb_model.pkl', "rb") as input_file:
                    self.model = pickle.load(input_file)
            except Exception:
                raise Exception("Provided xgboost model is either invalid or missing")
        
        try:
            with open(scaler_path or r'..\models\xgb_scaler.pkl', "rb") as input_file:
                scaler = pickle.load(input_file)
                self.labels = scaler[0].classes_
                self.scaler = scaler[1]
        except Exception:
            raise Exception("Provided scaler is either invalid or missing")
        
    def predict_verbally(self, filename: str, *, top_picks: int = 3):
        proba = self.predict_single(filename)
        order = np.argsort(proba)

        for i in range(0, top_picks):
            i = order[-i-1]
            print(f"{self.labels[i]}: {proba[i]*100:.3g}%")

    def predict(self, filenames: str | list[str]):
        if not isinstance(filenames, list):
            return self.predict_single(filenames)
        return self.predict_multiple(filenames)
    
    def predict_single(self, filename: str):
        return self.predict_multiple([filename])[0]
    
    def predict_multiple(self, filenames: list[str]):
        features = self._preprocess_files(filenames)
        return self.model.predict_proba(features)
    
    def _preprocess_files(self, filenames: list[str]):
        functional_smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        functional_features = functional_smile.process_files(filenames)
        functional_features.reset_index(inplace=True)
        functional_features.drop(columns=['start', 'end', 'file'], inplace=True)
        functional_features = self.scaler.transform(functional_features)

        return functional_features


def main():
    parser = argparse.ArgumentParser(description="Detect emotion from an audio file.")
    parser.add_argument("file", type=str, help="Path to the audio file")
    parser.add_argument("--xgboost-model-path", type=str, help="Path to the trained xgboost model in pickle format")
    parser.add_argument("--scaler-path", type=str, help="Path to the trained scaler in pickle format")
    args = parser.parse_args()

    predictor = EmotionPredictor(
        model_path=args.xgboost_model_path,
        scaler_path=args.scaler_path,
    )
    predictor.predict_verbally(args.file)

if __name__ == "__main__":
    main()