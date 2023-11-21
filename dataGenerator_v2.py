import numpy as np
import torch
import isingOne_Linear_64  # <- importing the appropriate model file

noise_dim = isingOne_Linear_64.noise_dim  # <- size of input noise
volume = 40000  # <- expected amount of data
beta = isingOne_Linear_64.beta
savedDataPath = "outIsingData/" + beta + "/64-" + beta + "[" + str(noise_dim) + "]/outputData(64-" + beta + ")TestFileLinear[" + str(noise_dim) + "]Generated.npy"  # <-path to save the data
savedModelPath = isingOne_Linear_64.savedModel

inputData = torch.randn(volume, 1, noise_dim)
model = isingOne_Linear_64.Generator()  # <- running the appropriate model

model.load_state_dict(torch.load(savedModelPath))  # <- path of the saved model
model.eval()
output = model(inputData)

_outputData = output.sign().detach().numpy()
np.save(savedDataPath, _outputData)

print("saved model path: ", savedModelPath)
print("saved data path: ", savedDataPath)
# print("Job Done!")