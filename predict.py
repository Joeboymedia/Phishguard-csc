"""
predict.py — Phishing Website Detector
========================================
Use the trained best model to classify new website samples.

Usage:
    python predict.py --input new_urls.csv
    python predict.py --demo              (runs 5 sample predictions)
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

BASE      = os.path.dirname(os.path.abspath(__file__))
MDL_DIR   = os.path.join(BASE, 'models')

def load_model():
    model    = joblib.load(os.path.join(MDL_DIR, 'best_model.pkl'))
    features = joblib.load(os.path.join(MDL_DIR, 'feature_names.pkl'))
    return model, features

def predict(X: pd.DataFrame, model, features) -> pd.DataFrame:
    """Returns dataframe with Prediction and Confidence columns."""
    X = X[features]
    preds  = model.predict(X)
    probas = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else [None]*len(X)
    labels = ['Legitimate ✅' if p == 1 else 'PHISHING ⚠️' for p in preds]
    conf   = [f"{p*100:.1f}%" if p is not None else 'N/A' for p in probas]
    return pd.DataFrame({'Prediction': labels, 'Confidence': conf})

def demo():
    """Run 5 hand-crafted demo samples."""
    model, features = load_model()

    # Sample 1: highly suspicious (phishing)
    phish = {f: -1 for f in features}
    phish.update({'SSLfinal_State': -1, 'Domain_registeration_length': -1,
                  'having_IP_Address': -1, 'URL_Length': -1})

    # Sample 2: legitimate site
    legit = {f: 1 for f in features}
    legit.update({'SSLfinal_State': 1, 'Domain_registeration_length': 1,
                  'Google_Index': 1, 'web_traffic': 1})

    demos = pd.DataFrame([phish, legit,
                          {f: np.random.choice([-1,1]) for f in features},
                          {f: np.random.choice([-1,1]) for f in features},
                          {f: np.random.choice([-1,1]) for f in features}])

    results = predict(demos, model, features)
    demos['Prediction']  = results['Prediction'].values
    demos['Confidence']  = results['Confidence'].values

    print("\n=== DEMO PREDICTIONS ===")
    print(demos[['SSLfinal_State', 'having_IP_Address',
                 'URL_Length', 'Prediction', 'Confidence']].to_string(index=False))

def from_file(csv_path: str):
    model, features = load_model()
    df = pd.read_csv(csv_path)
    results = predict(df, model, features)
    out = pd.concat([df, results], axis=1)
    out_path = csv_path.replace('.csv', '_predictions.csv')
    out.to_csv(out_path, index=False)
    print(f"Predictions saved → {out_path}")
    print(f"\nSummary:\n{results['Prediction'].value_counts()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phishing Website Detector')
    parser.add_argument('--input', type=str, help='CSV file with feature columns')
    parser.add_argument('--demo',  action='store_true', help='Run demo predictions')
    args = parser.parse_args()

    if args.demo or not args.input:
        demo()
    else:
        from_file(args.input)
