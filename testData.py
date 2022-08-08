import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import binom

magnetization = []

ising_data = np.load('outIsing/outputDataTestFile.npy')
data = ising_data.squeeze()
np.set_printoptions(threshold=sys.maxsize)
energy = -(data * np.roll(data, 1, 1)).sum(1).astype('int64')


def showSyntheticData():
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
    if histTitle == "Energy Histogram":
        L = 16
        es = np.arange(-16, 16.5, 4)
        rho = 2 ** (1 - L) * binom(L, (L + es) / 2) * np.exp(-es) / (np.cosh(1) ** L + np.sinh(1) ** L)
        plt.scatter(es, rho, marker='+', s=500, c='red', label='theory')
        plt.legend()
    plt.hist(InData, bins=np.arange(-16.5, 17, 1), density=True)
    plt.show()


def main():
    showSyntheticData()
    countDuplicates()
    histogram(magnetization, "Magnetization Histogram")
    histogram(energy, "Energy Histogram")
    print(magnetization)
    count = 0
    for i in range(len(magnetization)):
        if magnetization[i] == 15:
            count = count + 1
    print(count)

if __name__ == "__main__":
    main()
