import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# List of telemetry files to combine
CSV_FILES = [
    'telemetry_log_unknown.csv',
    'telemetry_etrack-corolla.csv',
    'telemetry_dirt-corolla.csv',
    'telemetry_dirt-p406.csv',
    'telemetry_dirt-evo.csv',
    'telemetry_oval.csv',
    'telemetry_oval-p406.csv',
    'telemetry_oval-pw-evoviwrc.csv'
]

MODEL_OUT = 'torcs_model.pt'
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1) Load and concatenate all data
df_list = [pd.read_csv(f) for f in CSV_FILES]
df = pd.concat(df_list, ignore_index=True)

# 2) Feature & label matrices
feature_cols = [
    'angle','curLapTime','damage','distFromStart','distRaced',
    'fuel','racePos','rpm','speedX','speedY','speedZ','trackPos','z'
] + [f'track_{i}' for i in range(19)] \
  + [f'opponent_{i}' for i in range(36)] \
  + [f'focus_{i}' for i in range(5)]

label_cols = ['accel','brake','steer','gear_cmd']

X = df[feature_cols].fillna(0).values.astype(np.float32)
y = df[label_cols].fillna(0).values.astype(np.float32)

# 3) Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

# 4) Standardize features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)

# Save scaler for inference
import joblib
joblib.dump(scaler, 'scaler.save')

# 5) PyTorch Dataset
class TelemetryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TelemetryDataset(X_train, y_train)
val_ds   = TelemetryDataset(X_val,   y_val)

# Load your data in mini-batches and Shuffle it
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# 6) Model definition
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = Net(len(feature_cols), len(label_cols)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 7) Training loop
best_val_loss = float('inf')
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad() #clear previous gradeients
        preds = model(xb)
        loss  = criterion(preds, yb) # calculate MSE
        loss.backward()
        optimizer.step() # applies gradiesnt descesnt
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)  #avg training loss

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            val_loss += criterion(preds, yb).item() * xb.size(0)
    val_loss /= len(val_ds)

    print(f"Epoch {epoch:2d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    # save best
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), MODEL_OUT)
        best_val_loss = val_loss

print("Training complete. Best val loss:", best_val_loss)
