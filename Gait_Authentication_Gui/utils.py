# utils.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def load_dataset(csv_path='features.csv'):
    """Load and preprocess the gait features dataset"""
    try:
        df = pd.read_csv(csv_path, header=None)
        if len(df) < 2:
            raise ValueError("Dataset needs at least 2 samples")
            
        labels = df.iloc[:, 0]
        features = df.iloc[:, 1:]
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Configure outlier detection
        n_samples = len(scaled_features)
        n_neighbors = min(5, n_samples - 1)  # Adjust for small datasets
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
        lof.fit(scaled_features)
        
        return labels, scaled_features, scaler, lof
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {str(e)}")
        return None, None, None, None