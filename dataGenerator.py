import numpy as np
import torch
import isingOne_Linear_32Float  # <- importing the appropriate model file

noise_dim = 32  # <- size of input noise
volume = 100000  # <- expected amount of data
savedDataPath = "outIsing/outputData(32-0150)TestFileLinearFloat[32].npy"  # <-path to save the data

inputData = torch.randn(volume, 1, noise_dim)
model = isingOne_Linear_32Float.Generator()  # <- running the appropriate model

model.load_state_dict(torch.load('isingOne(32-0150)LinearFloat[32].pth'))  # <- path of the saved model
model.eval()
output = model(inputData)

_outputData = output.sign().detach().numpy()
np.save(savedDataPath, _outputData)

print("Job Done!")
