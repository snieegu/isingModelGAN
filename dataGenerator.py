import torch
import isingOne_Linear_16
import numpy as np

noise_dim = 16
volume = 50000
savedDataPath = "outIsing/outputDataCustomGenerator.npy"

inputData = torch.randn(volume, 1, noise_dim)

model = isingOne_Linear_16.Generator()
model.load_state_dict(torch.load('isingOneLinear16.pth'))
model.eval()

output = model(inputData)

print("Out data: ", output.shape)
print(output.sign())

_outputData = output.sign().detach().numpy()
np.save(savedDataPath, _outputData)

print("Job Done!")
