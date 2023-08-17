import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from torchsummary import summary

# Wczytanie danych wejściowych i wyjściowych
input_data = np.load("input_data.npy")
output_data = np.load("output_data.npy")
# print("Output data numpy ", output_data)
# print("Output data type ", output_data.dtype)
# print("Input data numpy ", input_data)
# print("Input data type ", input_data.dtype)

batch_size = 1000
epochs = 500

savedModel = "floatsData/savedModels/float32predictionModel6Layers.pth"  # <- path to save model

# output_data = output_data.astype(np.float32)
# print("Output data float32 ", output_data)

# Konwersja danych na tensory PyTorch
input_data = torch.from_numpy(input_data)
print("Input data before ", input_data)
# # print("shape", input_data.shape)
# input_data = input_data.unsqueeze(1)
# print("Input data after unsqueeze", input_data)
# print("shape", input_data.shape)
output_data = torch.from_numpy(output_data)
print("Output data torch ", output_data)
# print("Output data shape", output_data.shape)

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)  # .reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)  # .reshape(-1, 1)

print("X shape", X_train.shape)
print("Input data type ", input_data.dtype)
print("y shape", y_train.shape)
print("Output data type ", input_data.dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# X_train_dataset = IsingDataset(X_train, transformation)
# y_train_dataset = IsingDataset(y_train, transformation)
# X_test_dataset = IsingDataset(X_train, transformation)
# y_test_dataset = IsingDataset(y_test, transformation)
# data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Float32Predictor(nn.Module):
    def __init__(self):
        super(Float32Predictor, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(in_features=1, out_features=256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),

            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),

            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=23),

            # nn.Tanh(),

        )

        # self.output_layer = nn.Linear(in_features=23, out_features=10000)

    def forward(self, input):
        output = self.main(input)
        # output = self.output_layer(output)
        return output


predictor = Float32Predictor()
if torch.cuda.is_available():
    predictor = predictor.cuda()

print("\nPredictor summary:")
summary(predictor, (batch_size, 1))

# Definicja funkcji straty i optymalizatora
criterion = nn.MSELoss()
optimizer = optim.Adam(predictor.parameters(), lr=0.001)

history = []
batch_start = torch.arange(0, len(X_train), batch_size)
best_mse = np.inf  # init to infinity
best_weights = None


# Saving model
def save_model():
    torch.save(predictor.state_dict(), savedModel)


# Training AI network
def trainingLoop(_epochs, _batch_size):
    global best_mse, best_weights
    for epoch in range(_epochs):
        print("Epoch: ", epoch)
        predictor.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + _batch_size]
                y_batch = y_train[start:start + _batch_size]
                # forward pass
                y_pred = predictor(X_batch)
                loss = criterion(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        predictor.eval()
        y_pred = predictor(X_test)
        mse = criterion(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(predictor.state_dict())

    predictor.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()

    save_model()
    print("MODEL SAVED!")


def main():
    print(" ")
    trainingLoop(epochs, batch_size)
    # restoreModel()


if __name__ == "__main__":
    main()
