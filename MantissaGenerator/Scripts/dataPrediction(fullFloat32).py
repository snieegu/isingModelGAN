import numpy as np
import torch

import fullFloat32learningModel  # <- importing the appropriate model file

savedInputData = '../floatsData/fullFloats32_input_data.npy'
savedOutputData = '../floatsData/fullFloats32_output_data.npy'

random_floats = np.random.uniform(low=0.0, high=1.0, size=10)
random_floats = random_floats.reshape(-1, 1)
# random_floats = np.load(savedInputData)
real_representation = np.load(savedOutputData)
# random_floats = random_floats[1000:1005]
real_representation = real_representation[1000:1005]
input_data = torch.from_numpy(np.array(random_floats, dtype=np.float32))

print("generated random floats: \n", random_floats)


def threshold(x):
    return (x >= 0.5).to(torch.int)


model = fullFloat32learningModel.Float32Predictor()  # <- running the appropriate model
try:
    model.load_state_dict(torch.load('../floatsData/savedModels/fullFloat32predictionModel5Layers.pth'), strict=False)
except FileNotFoundError:
    print("Model not found, please check the file path.")
    exit(1)
except Exception as e:
    print("Error occurred while loading the model: ", e)
    exit(1)

model.eval()
output = model(input_data)


# def float_to_binary(float_value: float) -> str:
#     hex_repr = float.hex(float_value)
#     exponent, mantissa = hex_repr.split('1.')
#     mantissa = mantissa.split('p')[0]
#
#     # Konwersja wykładnika na liczbę dziesiętną i normalizacja
#     exponent = int(exponent[-3:-1], 16) - 127
#     binary_exponent = format(exponent, 'b').zfill(8)
#
#     int_value = int(mantissa, 16)
#     binary_mantissa = format(int_value, 'b').zfill(23)
#
#     return '0' + binary_exponent + binary_mantissa


# Compute the true binary representations
# true_binary_representations = [float_to_binary(float_val) for float_val in random_floats.flatten()]

def numpy_array_to_binary_strings(array):
    return [''.join([str(bit) for bit in row]) for row in array]


binary_representations = threshold(output)
binary_representations_str = numpy_array_to_binary_strings(binary_representations.numpy())
print("Generated binary representations: \n", binary_representations_str, '\n')

# binary_representations = threshold(output)
# print("Generated binary representations: \n", binary_representations.numpy(), '\n')
# print("Real binary representations: \n", binary_representations)

# Reconstruct the float values using the generated binary representations
predicted_floats = []
for i in range(len(binary_representations)):
    binary_list = [str(x.item()) for x in binary_representations[i]]
    binary_str = ''.join(binary_list)
    sign = 1
    exponent = int(binary_str[1:9], 2) - 127
    mantissa = int(binary_str[9:], 2) / (2 ** 23)
    predicted_float = sign * (1 + mantissa) * (2 ** exponent)
    predicted_floats.append(predicted_float)

# print("Original floats: ", random_floats.flatten())
# print("Predicted floats: ", predicted_floats)
