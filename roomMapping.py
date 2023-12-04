import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Step 2: Create a PyTorch Dataset
class SensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Step 3: Build the LSTM Model
class SensorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SensorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

# Step 4: Define the Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs = data[:, :-1, :]  # Input features
            labels = data[:, -1, :]    # Labels (sensor to be used next)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Step 5: Evaluate the Model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[:, :-1, :]
            labels = data[:, -1, :]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Example usage:
# Assuming your data is a numpy array, and you have split it into training and testing sets
train_data = ...  # Your training data
test_data = ...   # Your testing data

train_dataset = SensorDataset(train_data)
test_dataset = SensorDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_size = train_data.shape[2] - 1  # Number of features excluding the label
hidden_size = 64
num_classes = 1  # Binary classification (sensor used or not)

model = SensorLSTM(input_size, hidden_size, num_classes)
criterion = nn.BCEWithLogitsLoss()  # Binary CrossEntropyLoss with logits for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Step 6: Make Predictions
# You can use the trained model to make predictions on new data
