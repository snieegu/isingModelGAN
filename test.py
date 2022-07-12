import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

# energy = []
temperature = []

ising_data = np.load('ising/cfg_x016_b0100.npy')
data = ising_data.squeeze()
data = data[0:400]
# print(data.shape)


def showRealData():
    for i in range(len(data)):
        temperature.append(data[i].sum())

    plt.figure(figsize=(7, 5))
    plt.title("Real data Magnetization")
    plt.plot(temperature, label="Magnetization")
    plt.xlabel("")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.show()

    averageTemp = numpy.asarray(temperature)
    print("Real Magnetization mean: ", np.average(averageTemp))

    # energySum = 0
    # for i in range(len(data)):
    #     for j in range(0, 16):
    #         neighbor = data[i][j - 1] + data[i][j]
    #         if neighbor == 0:
    #             energySum = energySum + 1
    #         else:
    #             energySum = energySum - 1
    #     energy.append(energySum)
    #     energySum = 0
    #
    # averageEnergy = numpy.asarray(energy)
    # print("Average real energy: ", np.average(energy))

    energy = -(data * np.roll(data, 1, 1)).sum(1).astype('int64')
    print("Real Energy mean:", energy.mean())

    plt.figure(figsize=(7, 5))
    plt.title("Real data Energy")
    plt.plot(energy, label="Energy")
    plt.xlabel("")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()


def countDuplicates():
    resultList = data.tolist()
    duplicates = {tuple(_) for _ in resultList if resultList.count(_) > 1}
    duplicatesCount = len(duplicates)
    print("duplicates: ", duplicatesCount, "/", len(data))


def main():
    showRealData()
    countDuplicates()


if __name__ == "__main__":
    main()
