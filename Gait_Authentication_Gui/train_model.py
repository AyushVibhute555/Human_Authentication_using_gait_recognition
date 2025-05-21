import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Load the data
try:
    df = pd.read_csv('features.csv', header=None)
    if len(df) < 5:  # Minimum 5 samples required
        raise ValueError("Insufficient data - need at least 5 samples")
        
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    
    # Check feature consistency
    if features.shape[1] == 0:
        raise ValueError("No features extracted - check extract_features.py")

    # Split data (only if sufficient samples)
    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels)
    else:
        print("[WARNING] Using all data for training (small dataset)")
        X_train, y_train = features, labels
        X_test, y_test = features, labels  # Not ideal but works for small datasets

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) if len(df) >= 10 else X_train

    # Train models
    print(f"\nTraining with {len(X_train)} samples...")
    
    svm = SVC(probability=True, kernel='rbf', class_weight='balanced')
    svm.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Evaluate
    if len(df) >= 10:
        print("\nSVM Test Results:")
        print(classification_report(y_test, svm.predict(X_test)))
        
        print("\nRandom Forest Test Results:")
        print(classification_report(y_test, rf.predict(X_test)))

    # Save models
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(svm, 'svm_model.pkl')
    joblib.dump(rf, 'rf_model.pkl')
    print("\n[SUCCESS] Models saved to disk")

except Exception as e:
    print(f"\n[ERROR] Training failed: {str(e)}")
    print("\nTroubleshooting:")
    print("- Run build_dataset.py first")
    print("- Ensure you have multiple videos per person")
    print("- Check features.csv contains valid data")