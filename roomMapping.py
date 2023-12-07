import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from statistics import mode
df = pd.read_pickle('kitchenData.pkl')
daily_window = []
dfSize  = len(df)
windowLens =[]
#print(dfSize)
i=0
while(i<dfSize):
    j=i
    while(True):
        if(i==dfSize-1):
            window = df.iloc[j:i]
            daily_window.append(window)
            i+=1
            windowLens.append(len(window))
            break
        elif(df.at[i,'hour']>df.at[i+1,'hour']):
            window = df.iloc[j:i+1]
            daily_window.append(window)
            i+=1
            windowLens.append(len(window))
            break
        i+=1
lenMode = mode(windowLens)
daily_window[:] = [window for window in daily_window if len(window) == lenMode]

#print(daily_window)


"""
# Assume you have a DataFrame result_df with 'timestamp' and 'cooking_label' columns
result_df['hour_of_day'] = result_df['timestamp'].dt.hour

# Create daily windows
daily_windows = []
for day in pd.date_range(result_df['timestamp'].min(), result_df['timestamp'].max(), freq='D'):
    daily_window = result_df[(result_df['timestamp'] >= day) & (result_df['timestamp'] < day + pd.Timedelta(days=1))]
    daily_windows.append(daily_window)
"""
# Initialize the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only consider the output of the last time step
        out = self.sigmoid(out)
        return out

# Convert data to PyTorch tensors
X_tensors = []
y_tensors = []
for window in daily_window:
    X = window['hour'].values.reshape(-1, 1)
    y = window['cooking_label'].values
    X_tensors.append(torch.tensor(X, dtype=torch.float32))
    y_tensors.append(torch.tensor(y, dtype=torch.float32))

# Train-Test Split
X_train_tensors, X_test_tensors, y_train_tensors, y_test_tensors = train_test_split(
    X_tensors, y_tensors, test_size=0.2, shuffle=False
)

# Normalize the data
scaler = StandardScaler()
X_train_tensors = [torch.tensor(scaler.fit_transform(x), dtype=torch.float32) for x in X_train_tensors]
X_test_tensors = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_test_tensors]
y_train_tensors = [y.unsqueeze(1) for y in y_train_tensors]
y_test_tensors = [y.unsqueeze(1) for y in y_test_tensors]
#for tensor in X_train_tensors:
#    print(tensor.shape)
#print("X")
#for tensor in y_train_tensors:
#    print(tensor.shape)
#print("Y")
# Create DataLoader for PyTorch
train_dataset = TensorDataset(*X_train_tensors, *y_train_tensors)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Initialize and train the model
input_size = 1
hidden_size = 50
output_size = 1
num_epochs = 10

model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in train_loader:
        X_batch = batch[:-1]
        y_batch = batch[-1]
        X_batch = torch.stack(X_batch, dim=1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        outputs = torch.sigmoid(outputs)
        # Print statements for debugging
        print("Outputs min:", outputs.min().item())
        print("Outputs max:", outputs.max().item())
        print("y_batch min:", y_batch.min().item())
        print("y_batch max:", y_batch.max().item())
        loss = criterion(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()

# Evaluate the model
X_test_tensors = torch.cat(X_test_tensors, dim=0)
y_test_tensors = torch.cat(y_test_tensors, dim=0)

X_test_tensors = torch.tensor(scaler.transform(X_test_tensors), dtype=torch.float32)

with torch.no_grad():
    model.eval()
    y_pred = model(X_test_tensors).squeeze().numpy()

# Convert probabilities to binary predictions
y_pred_binary = (y_pred >= 0.5).astype(float)

# Evaluate the model
accuracy = np.mean(y_pred_binary == y_test_tensors.numpy())
print(f'Test Accuracy: {accuracy}')

