import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load the dataset
file_path = 'combined/sensor_combined.pkl'
df = pd.read_pickle(file_path)

# Convert the DataFrame to a PyTorch tensor
data = torch.tensor(df.values, dtype=torch.float32)


# Define a function to create input-output sequences
def create_inout_sequences(input_data, sequence_length):
    inout_seq = []
    L = len(input_data)
    for i in range(L - sequence_length):
        train_seq = input_data[i:i + sequence_length]
        train_label = input_data[i + sequence_length:i + sequence_length + 1]
        inout_seq.append((train_seq, train_label.squeeze()))  # Squeeze the label to remove unnecessary dimensions
    return inout_seq

# Define the sequence length
sequence_length = 1

train_inout_seq = create_inout_sequences(data, sequence_length)

train_loader = DataLoader(train_inout_seq, batch_size=1, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, output_size=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_layer_size).to(device),
                torch.zeros(1, batch_size, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        self.hidden_cell = self.init_hidden(input_seq.size(0))
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_size=4).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 10

for epoch in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        seq, labels = seq.to(device), labels.to(device)
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if epoch % 25 == 1:
        print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {epoch:3} loss: {single_loss.item():10.10f}')
