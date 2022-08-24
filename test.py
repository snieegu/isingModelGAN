import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

magnetization = []

ising_data = np.load('ising/cfg_x016_b0100.npy')
data = ising_data.squeeze()
# data = data[0:2000]
energy = -(data * np.roll(data, 1, 1)).sum(1).astype('int64')

print(len(data))

def showSyntheticData():
    for i in range(len(data)):
        magnetization.append(data[i].sum())

    plt.figure(figsize=(7, 5))
    plt.title("Real data Magnetization")
    plt.plot(magnetization, label="Magnetization")
    plt.xlabel("")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.show()

    averageMagnetization = np.asarray(magnetization)
    print("Real Magnetization mean: ", np.average(averageMagnetization))
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
    duplicates = {tuple(x) for x in resultList if resultList.count(x) > 1}
    duplicatesCount = len(duplicates)
    print("duplicates: ", duplicatesCount, "/", len(data))


def histogram(InData, histTitle):
    plt.title(histTitle)
    es = np.arange(-16, 16.5, 4)
    ms = np.arange(-16, 16.5, 2)
    L = 16
    if histTitle == "Energy Histogram":
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
