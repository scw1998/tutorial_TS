import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import AutoTokenizer, AutoformerForPrediction, AutoformerConfig,AutoformerPreTrainedModel
import torch
from pytorch_forecasting import TimeSeriesDataSet
lags_sequence =[1,2,3,4,5,6,7]
config = AutoformerConfig(
    num_time_features=5,
    context_length=96,
    prediction_length=96,
    input_size=326,
    lags_sequence=lags_sequence,
    encoder_layers=2,
    decoder_layers= 1,
    autocorrelation_factor=3
)

model = AutoformerForPrediction(config)


import pandas as pd
elec = pd.read_csv("electricity/electricity.csv")

def create_time_features(df):
    df.date = pd.to_datetime(df.date)
    df['month'] = df.date.apply(lambda row: row.month, 1)
    df['day'] = df.date.apply(lambda row: row.day, 1)
    df['weekday'] = df.date.apply(lambda row: row.weekday(), 1)
    df['hour'] = df.date.apply(lambda row: row.hour, 1)
    df["year"] = df.date.apply(lambda row: row.year, 1)
    return df[["hour",  "weekday", "day", "month", "year"]].values


import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoformerConfig, AutoformerForPrediction
from sklearn.preprocessing import StandardScaler

# print(elec.values)
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_length, prediction_length,scales=True):
        self.data = data
        self.scales=scales
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.data_stamp = create_time_features(self.data)
        self.data = self.data.drop(columns = ["date"])
        # future_time_features = create_time_features(y)
        if self.scales:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data.values)
            self.data = self.scaler.fit_transform(self.data.values)
        
    
    def __len__(self):
        return len(self.data) - self.input_length - self.prediction_length +1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_length]
        y = self.data[idx + self.input_length:idx + self.input_length + self.prediction_length]
        
        past_time_features =  self.data_stamp[idx:idx + self.input_length]
        future_time_features = self.data_stamp[idx + self.input_length:idx + self.input_length + self.prediction_length]

        # x= x.values
        # y=y.values
        return torch.tensor(x),torch.tensor(past_time_features), torch.tensor(y),torch.tensor(future_time_features)

# Parameters
input_length = 96
prediction_lengths = [96, 192, 336, 720]
train_size = int(len(elec) * 0.8)
train = elec.iloc[:train_size]
test_size = int(len(elec) * 0.2)
val_size = len(elec) - train_size - test_size

train_data = elec.iloc[:train_size]
val_data = elec.iloc[train_size:train_size+val_size]
test_data = elec.iloc[train_size+val_size:]

# Create datasets
train_datasets = {pl: TimeSeriesDataset(train_data,input_length+max(lags_sequence), pl) for pl in prediction_lengths}
val_datasets = {pl: TimeSeriesDataset(val_data, input_length+max(lags_sequence), pl) for pl in prediction_lengths}
# Create DataLoaders
train_dataloaders = {pl: DataLoader(train_datasets[pl], batch_size=32, shuffle=False) for pl in prediction_lengths}
val_dataloaders = {pl: DataLoader(val_datasets[pl], batch_size=32, shuffle=False) for pl in prediction_lengths}


import torch.optim as optim
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, dataloader, num_epochs=10):
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x, past_time_features, y, future_time_features = [
                datum.float().to(device)
                for datum in batch
            ]
            optimizer.zero_grad()
            
            outputs = model(
                past_values=x,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                past_observed_mask=torch.ones_like(x),
                future_values=y
                )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Training with different prediction lengths
for pl, dataloader in train_dataloaders.items():
    print(f"Training for prediction length: {pl}")
    config.prediction_length = pl
    model = AutoformerForPrediction(config)
    train_model(model, dataloader)
    torch.save(model.state_dict(), f"model_{pl}.pth")
    print(f"Model saved for prediction length: {pl}")

# for i in prediction_lengths:
#     model = AutoformerForPrediction(config)
#     model.load_state_dict(torch.load(f"model_{i}.pth"))
#     model.eval()
#     model.to(device)
#     test_dataset = TimeSeriesDataset(test_data, input_length+max(lags_sequence), i, scales=False)
#     test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#     mae = 0
#     for batch in test_dataloader:
#         x, past_time_features, y, future_time_features = [
#             datum.float().to(device)
#             for datum in batch
#         ]
#         outputs = model(
#             past_values=x,
#             past_time_features=past_time_features,
#             future_time_features=future_time_features,
#             past_observed_mask=torch.ones_like(x),
#             future_values=y
#             )
#         mae += outputs.loss.item()
#     mae /= len(test_dataloader)
#     print(f"MAE for prediction length {i}: {mae}")
