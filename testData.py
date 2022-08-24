import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import binom

magnetization = []
ising_data = np.load('outIsing/outputDataCustomGenerator.npy')
# ising_data = np.load('outIsing/outputDataTestFileLinear16.npy')
print("clear data shape: ", ising_data.shape)
data = ising_data.squeeze()
data = data[0:30000]
# np.set_printoptions(threshold=sys.maxsize)
energy = -(data * np.roll(data, 1, 1)).sum(1).astype('int64')

print("data after squeeze", data.shape)
dataSize = len(data)


def showSyntheticDataChart():
    for i in range(len(data)):
        magnetization.append(data[i].sum())

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
    print("duplicates: ", duplicatesCount, "/", len(data))


def histogram(InData, histTitle):
    plt.title(histTitle)
    plt.xlabel("Histogram for " + str(dataSize) + " data")
    plt.ylabel("")
    es = np.arange(-16, 16.5, 4)
    ms = np.arange(-16, 16.5, 2)
    L = 16
    if histTitle == "Fake Energy Histogram":
        rho = 2 ** (1 - L) * binom(L, (L + es) / 2) * np.exp(-es) / (np.cosh(1) ** L + np.sinh(1) ** L)
        plt.scatter(es, rho, marker='+', s=500, c='red', label='theory')
        plt.legend()
    else:
        rho = [0.13, 0.04, 0.043, 0.05, 0.046, 0.059, 0.06, 0.055, 0.061, 0.051, 0.05, 0.047, 0.042, 0.041, 0.047,
               0.032, 0.121]
        plt.scatter(ms, rho, marker='+', s=500, c='red', label='theory')
        plt.legend()
    plt.hist(InData, bins=np.arange(-16.5, 17, 1), density=True)
    plt.show()


def main():
    showSyntheticDataChart()
    countDuplicates()
    histogram(magnetization, "Fake Magnetization Histogram")
    histogram(energy, "Fake Energy Histogram")
    count = 0
    for i in range(len(magnetization)):
        if magnetization[i] == 15:
            count = count + 1
    print(count)


if __name__ == "__main__":
    main()
