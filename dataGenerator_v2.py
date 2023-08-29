import numpy as np
import torch
import isingOne_Linear_64  # <- importing the appropriate model file

noise_dim = 32  # <- size of input noise
volume = 40000  # <- expected amount of data
savedDataPath = "outIsingData/s0100/64-s0100[32]/outputData(64-s0100)TestFileLinear[32]Generated.npy"  # <-path to save the data

inputData = torch.randn(volume, 1, noise_dim)
model = isingOne_Linear_64.Generator()  # <- running the appropriate model

model.load_state_dict(torch.load('savedModels/isingOne(64-s0100)Linear[32].pth'))  # <- path of the saved model
model.eval()
output = model(inputData)

_outputData = output.sign().detach().numpy()
np.save(savedDataPath, _outputData)

print("Job Done!")