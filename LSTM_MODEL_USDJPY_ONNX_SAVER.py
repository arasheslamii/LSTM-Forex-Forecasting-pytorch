# Libraries and Framework importation
!pip install onnx
!pip insall yfinance
import torch
from torch import nn
from torch.utils.data import DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime, timedelta
import sklearn
from sklearn.preprocessing import MinMaxScaler
import torch.onnx as onnx
from torch.utils.data import Dataset, DataLoader

end  = pd.to_datetime( datetime.now())
start = end  - timedelta(days= 59)
data = yf.download('USDJPY=X', start = start , end = end, interval= '5m' )
df = torch.tensor(data.drop(columns=['Adj Close', 'Volume']).reset_index().drop(columns = ['Datetime']).to_numpy())

X,y =  [], []
for i in range(0,len(df)-144-60):
  X.append(df[i:i+144])
  y.append(df[i+144:(i+144+60)])

#creating X_train, X_test, y_train, y_test

train_size = 0.8
X_train  = np.array(X[:int(train_size*len(X))])
X_test  = np.array(X[int(train_size*len(X)):])
y_train  = np.array(y[:int(train_size*len(y))])
y_test  = np.array(y[int(train_size*len(y)):])

#reshape to get ready for minmax scaling, because minmax scale got only 2D array

X_train_reshape  = X_train.reshape(-1, X_train.shape[-1])
X_test_reshape  = X_test.reshape(-1, X_test.shape[-1])
y_train_reshape  = y_train.reshape(-1, y_train.shape[-1])
y_test_reshape  = y_test.reshape(-1, y_train.shape[-1])

#Scaling the data from #_###_reshape to
scaler = MinMaxScaler(feature_range=(0,1))
X_train_sc = scaler.fit_transform(X_train_reshape)
y_train_sc = scaler.fit_transform(y_train_reshape)
X_test_sc = scaler.transform(X_test_reshape)
y_test_sc = scaler.transform(y_test_reshape )

# Reshape the scaled data to getting back to their original shape
X_train_sc = torch.from_numpy( X_train_sc.reshape(X_train.shape)).float()
y_train_sc = torch.from_numpy(y_train_sc .reshape(y_train.shape)).float()
X_test_sc = torch.from_numpy(X_test_sc .reshape(X_test.shape)).float()
y_test_sc = torch.from_numpy(y_test_sc .reshape(y_test.shape)).float()

print(f'shape of the X_train: {X_train_sc.shape}, shpe of th X_test: {X_test.shape}, shpe of th y_train: {y_train.shape}, shpe of th y_test: {y_test.shape}')

class custom_dataset(Dataset):
  def __init__(self, X,y):
    self.X = X
    self.y = y
  def  __len__(self):
    return len(self.X)
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

train_dataloader = DataLoader(custom_dataset(
    X_train_sc, y_train_sc), batch_size = 32)

test_dataloader = DataLoader(custom_dataset(
    X_test_sc, y_test_sc), batch_size=32)

"""Lets Creeat a custome Dataset to make a dataloader

"""

class LSTM(nn.Module):
  def __init__(self, input_size , hidden_size , num_layers, output_size , dropout_prob):
    super(LSTM, self).__init__()
    self.lstm  = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                         num_layers=num_layers, batch_first=True,
                         dropout=dropout_prob)
    self.fc1 = nn.Linear(hidden_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc_out = nn.Linear(64, output_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, X):  # This should be outside __init__
    lstm_out, _ = self.lstm(X)
    lstm_out = lstm_out[:, -1, :]  # Only keep the output from the last time step
    out = self.relu(self.fc1(lstm_out))
    out = self.dropout(out)
    out = self.relu(self.fc2(out))
    out = self.fc_out(out)
    return out.view(out.size(0), 60, 4)

input_size = 4  # Number of input features (Open, High, Low, Close)
hidden_size = 128  # Number of LSTM neurons in each layer
num_layers = 2  # Number of LSTM layers
output_size = 240  # Predicting 60 future time steps, 4 features for each (60 * 4 = 240)
dropout_prob = 0.3  # 30% dropout rate

model = LSTM(input_size, hidden_size, num_layers, output_size, dropout_prob)

# Initialize the loss function adn Optimizer
loss_fn = nn.MSELoss()
#Lets define our crucial things
learning_rate = 1e-5
batch_size = 32
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):

  size = len(dataloader.dataset)
  model.train()
  for batch, (X,y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss = loss.item()
      current = batch * batch_size +len(X)
      print(f"loss \U0001F432: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Testing/Validation loop
def test_loop(dataloader, model, loss_fn):
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():  # No need to compute gradients during testing
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # Accumulate the loss

    test_loss /= num_batches  # Average loss over all batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Here we Wrap everything up and start the for loop to iterate through our two funcitions to do the training things, and custrunction is done here 👷🏼‍♀️ 🦸🏼‍♂️
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------\U0001F680")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")




# Dummy input for the model, which needs to match the shape of your actual input
dummy_input = torch.randn(1, 144, 4).to(device)  # batch size = 1, sequence length = 144, input features = 4

# Export the model
onnx.export(model,                         # model to be exported
            dummy_input,                   # an example input for tracing the model
            "model.onnx",                  # where to save the ONNX file
            export_params=True,            # store the trained weights inside the model
            opset_version=12,              # ONNX version, 12 should be fine for most tasks
            input_names=['input'],         # name of the input tensor
            output_names=['output'],       # name of the output tensor
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # dynamic batch size
)

print("Model exported as ONNX")

torch.save(model.state_dict(), 'model_weights.pth')

torch.save(model, 'model.pth')