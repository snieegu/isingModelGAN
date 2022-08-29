import numpy as np
import torch
import isingOne_Linear_64  # <- importing the appropriate model file

noise_dim = 64  # <- size of input noise
volume = 100000  # <- expected amount of data
savedDataPath = "outIsing/outputDataCustomGeneratorLinear64.npy"  # <-path to save the data

inputData = torch.randn(volume, 1, noise_dim)
model = isingOne_Linear_64.Generator()  # <- running the appropriate model

# print("Generator summary:")
# summary(model, (1, noise_dim))

model.load_state_dict(torch.load('isingOneLinear64.pth'))  # <- path of the saved model
model.eval()
output = model(inputData)

# print("Out data: ", output.shape)
# print(output.sign())

_outputData = output.sign().detach().numpy()
np.save(savedDataPath, _outputData)

print("Job Done!")
