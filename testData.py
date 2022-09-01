import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

ising_data = np.load('outIsing/outputData(128)CustomGeneratorLinear[64].npy')  # <- data for the test from the generated
# batch data
# ising_data = np.load('outIsing/outputDataTestFileLinear4.npy')  # <- test data coming from the generator
ising_realData = np.load('ising/cfg_x128_b0100.npy')
print("clear data shape: ", ising_data.shape)
isingSize = 128  # <- modify configuration length based on input
data = ising_data.squeeze()
dataLength = len(data)
realData = ising_realData.squeeze()
realMagnetization = realData.sum(axis=1)

energy = -(data * np.roll(data, 1, 1)).sum(1).astype('int64')
magnetization = data.sum(axis=1)


def showSyntheticDataChart():
    plt.figure(figsize=(7, 5))
    plt.title("Fake data Magnetization")
    plt.plot(magnetization, label="Magnetization")
    plt.xlabel("")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.show()

    averageMagnetization = np.asarray(magnetization)
    print("Fake Magnetization mean: ", np.average(averageMagnetization))
    print("Fake Energy mean:", energy.mean())

    plt.figure(figsize=(7, 5))
    plt.title("Fake data Energy")
    plt.plot(energy, label="Energy")
    plt.xlabel("")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()


def countDuplicates():
    resultList = data.tolist()
    duplicates = {tuple(x) for x in resultList if resultList.count(x) > 1}
    duplicatesCount = len(duplicates)
    print("duplicates: ", duplicatesCount, "/", dataLength)


def energyHistogram(InData):
    plt.title("Energy Histogram")
    es = np.arange(-isingSize, isingSize + 0.5, 4)
    L = isingSize
    rho = 2 ** (1 - L) * binom(L, (L + es) / 2) * np.exp(-es) / (np.cosh(1) ** L + np.sinh(1) ** L)
    plt.xlabel("Histogram for " + str(dataLength) + " data")
    plt.scatter(es, rho, marker='+', s=500, c='red', label='theory')
    plt.hist(InData, bins=np.arange(-isingSize - 0.5, isingSize + 1, 1), density=True)
    plt.legend()
    plt.show()


def magnetizationHistogram(InData):
    plt.title("Magnetization Histogram")
    ms = np.arange(-isingSize, isingSize + 0.5, 2)
    es = np.arange(-isingSize - 1, isingSize + 1.5, 2)
    hist, bins = np.histogram(realMagnetization, es)
    rho = (hist / 100000)
    plt.xlabel("Histogram for " + str(dataLength) + " data")
    plt.scatter(ms, rho, marker='+', s=500, c='red', label='theory')
    plt.hist(InData, bins=np.arange(-isingSize - 0.5, isingSize + 1, 1), density=True)
    plt.legend()
    plt.show()


def main():
    # showSyntheticDataChart()
    # countDuplicates()
    energyHistogram(energy)
    magnetizationHistogram(magnetization)
    # count = 0
    # for i in range(len(magnetization)):
    #     if magnetization[i] == 15:
    #         count = count + 1
    # print(count)


if __name__ == "__main__":
    main()
