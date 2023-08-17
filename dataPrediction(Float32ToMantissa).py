import numpy as np
import torch
import struct
import Float32learningModel  # <- importing the appropriate model file

random_floats = np.array([[0.1855], [0.1870], [0.5277]])  # sample data
input_data = torch.from_numpy(np.array(random_floats, dtype=np.float32))


def threshold(x):
    return (x >= 0.5).to(torch.int)


def binary_to_float(binary_list):
    binary_string = ''.join([str(x) for x in binary_list])
    int_num = int(binary_string, 2)
    float_num = int_num.to_bytes(4, byteorder='big', signed=False)
    return struct.unpack('>f', float_num)[0]


model = Float32learningModel.Float32Predictor()  # <- running the appropriate model
print("Model architecture ", model)
try:
    model.load_state_dict(torch.load('floatsData/savedModels/float32predictionModel5Layers.pth'), strict=False)
except FileNotFoundError:
    print("Model not found, please check the file path.")
    exit(1)
except Exception as e:
    print("Error occurred while loading the model: ", e)
    exit(1)

model.eval()
output = model(input_data)

binary_mantissas = threshold(output)
print(binary_mantissas)

# Convert binary mantissas to float numbers
predicted_floats = []
for i in range(len(binary_mantissas)):
    binary_list = [int(x) for x in binary_mantissas[i].tolist()]
    predicted_float = binary_to_float(binary_list)
    predicted_floats.append(predicted_float)


print("Original floats: ", random_floats.flatten())
print("Predicted floats: ", predicted_floats)

print("Job Done!")


# import numpy as np
# import torch
#
# import Float32learningModel  # <- importing the appropriate model file
#
# random_floats = np.array([[0.1855], [0.1870], [0.5277]])  # sample data
# input_data = torch.from_numpy(np.array(random_floats, dtype=np.float32))
#
#
# def threshold(x):
#     return (x >= 0.5).to(torch.int)
#
#
# # inputData = torch.randn(volume, 1, noise_dim)
# model = Float32learningModel.Float32Predictor()  # <- running the appropriate model
# print("Model architecture ", model)
# model.load_state_dict(torch.load('floatsData/savedModels/float32predictionModel6Layers.pth'),
#                       strict=False)  # <- path of the saved model
# model.eval()
# output = model(input_data)
#
# binary_mantissas = threshold(output)
# print(binary_mantissas)
# # _outputData = output.sign().detach().numpy()
# # np.save(savedDataPath, _outputData)
#
# print("Job Done!")
