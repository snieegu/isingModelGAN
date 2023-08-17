import numpy as np
import torch
import mantissaFloat32learningModel  # <- importing the appropriate model file

savedInputData = '../floatsData/input_data.npy'
savedOutputData = '../floatsData/output_data.npy'

random_floats = np.load(savedInputData)
real_representation = np.load(savedOutputData)
random_floats = random_floats[0:5]
real_representation = real_representation[0:5]
input_data = torch.from_numpy(np.array(random_floats, dtype=np.float32))


def threshold(x):
    return (x >= 0.5).to(torch.int)


model = mantissaFloat32learningModel.Float32Predictor()  # <- running the appropriate model
try:
    model.load_state_dict(torch.load('../floatsData/savedModels/float32predictionModel5Layers.pth'), strict=False)
except FileNotFoundError:
    print("Model not found, please check the file path.")
    exit(1)
except Exception as e:
    print("Error occurred while loading the model: ", e)
    exit(1)

model.eval()
output = model(input_data)

binary_mantissas = threshold(output)
print("Generated mantissas: \n", binary_mantissas.numpy(), '\n')
print("Real mantissas: \n", real_representation)

# Compute the predicted mantissa
predicted_mantissas = []
for i in range(len(binary_mantissas)):
    binary_list = ['1'] + [str(x.item()) for x in binary_mantissas[i]]  # Add '1' to the beginning
    mantissa_str = ''.join(binary_list)
    mantissa_int = int(mantissa_str, 2)
    mantissa_float = mantissa_int / (2 ** 23)
    predicted_mantissas.append(mantissa_float)


# Compute the exponent
exponents = []
for float_value in random_floats.flatten():
    _, exponent = np.frexp(float_value)
    exponents.append(exponent)

# Reconstruct the float values using the predicted mantissas and the original exponents
predicted_floats = []
for i in range(len(predicted_mantissas)):
    predicted_float = float(predicted_mantissas[i]) * (2.0 ** float(exponents[i]))
    predicted_floats.append(predicted_float)

# print("Original floats: ", random_floats.flatten())
# print("Predicted floats: ", predicted_floats)
