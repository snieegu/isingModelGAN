import struct

import numpy as np
import torch

savedInputData = './floatsData/fullFloats32_input_data.npy'
savedOutputData = './floatsData/fullFloats32_output_data.npy'


def float_to_bin_string(float_num):
    packToBin = struct.pack('!f', float_num)
    binary_string = ''.join(format(binary, '08b') for binary in packToBin)
    return binary_string


def generateRealData(quantity):
    input_data = torch.FloatTensor(quantity, 1).uniform_(0, 1)
    input_numpy_data = input_data.numpy()
    try:
        print("Saved generated floats to file")
        # torch.save(input_data, 'input_data.pt')
        np.save(savedInputData, input_numpy_data)
    except:
        print("Couldn't save generated floats to file")
    binary_mantissa = []

    for i in range(quantity):
        binary = float_to_bin_string(input_numpy_data[i])
        binary_list = [int(x) for x in binary]
        binary_mantissa.append(binary_list)

    try:
        print("Saved generated binary mantissa to file")
        np.save(savedOutputData, binary_mantissa)
    except:
        print("Couldn't save binary mantissa to file")


def printData():
    input_data = np.load(savedInputData)
    print("Floats: \n", input_data)
    output_data = np.load(savedOutputData)
    print("Binary representations: \n", output_data)

    print("lengths ", len(input_data), " ", len(output_data))


def main():
    generateRealData(10000)
    printData()


if __name__ == "__main__":
    main()
