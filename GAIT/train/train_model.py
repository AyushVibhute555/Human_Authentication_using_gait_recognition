import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from model import GaitRecognitionModel

# Load dataset
df = pd.read_csv('features_classifier.csv', header=None)
labels = df.iloc[:, 0].values
features = df.iloc[:, 1:].values

# Preprocessing
scaler = StandardScaler()
features = scaler.fit_transform(features)

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Save scaler and encoder
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')

# Prepare dataset
features = torch.tensor(features, dtype=torch.float32)
labels_encoded = torch.tensor(labels_encoded, dtype=torch.long)
dataset = TensorDataset(features, labels_encoded)

# Split into train and validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Build model
input_dim = features.shape[1]
num_classes = len(set(labels_encoded.numpy()))
model = GaitRecognitionModel(input_dim=input_dim, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epochs = 100
best_val_acc = 0.0
patience = 10
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_output = model(X_val)
            val_preds = val_output.argmax(dim=1)
            val_correct += (val_preds == y_val).sum().item()
            val_total += y_val.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/gait_model.pth')
        patience_counter = 0
        print("[INFO] Best model updated and saved.")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience:
        print("[EARLY STOPPING] No improvement, stopping training.")
        break

print("[INFO] Training complete.")
