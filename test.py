import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

ising_realData = np.load('ising/isingData2/s_cfg_x064_b1387.npy')
ising_realData = np.sign(ising_realData)
isingSize = 64
realData = ising_realData.squeeze()
dataLength = len(realData)

energy = -(realData * np.roll(realData, 1, 1)).sum(1).astype('int64')
magnetization = realData.sum(axis=1)


def showSyntheticData():
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


# def countDuplicates():
#     resultList = data.tolist()
#     duplicates = {tuple(x) for x in resultList if resultList.count(x) > 1}
#     duplicatesCount = len(duplicates)
#     print("duplicates: ", duplicatesCount, "/", dataLength)


def energyHistogram(InData):
    plt.title("Energy Histogram")
    ms = np.arange(-isingSize, isingSize + 0.5, 4)
    es = np.arange(-isingSize - 2, isingSize + 2.5, 4)
    hist, bins = np.histogram(energy, es)
    rho = (hist / dataLength)
    plt.xlabel("Histogram for " + str(dataLength) + " data")
    plt.scatter(ms, rho, marker='+', s=500, c='red', label='theory')
    plt.hist(InData, bins=np.arange(-isingSize - 0.5, isingSize + 1, 1), density=True)
    plt.legend()
    plt.show()


def magnetizationHistogram(InData):
    plt.title("Magnetization Histogram")
    ms = np.arange(-isingSize, isingSize + 0.5, 2)
    es = np.arange(-isingSize - 1, isingSize + 1.5, 2)
    hist, bins = np.histogram(magnetization, es)
    rho = (hist / dataLength)
    plt.xlabel("Histogram for " + str(dataLength) + " data")
    plt.scatter(ms, rho, marker='+', s=500, c='red', label='theory')
    plt.hist(InData, bins=np.arange(-isingSize - 0.5, isingSize + 1, 1), density=True)
    plt.legend()
    plt.show()


def main():
    # print("Real Energy ", energy)
    # showSyntheticData()
    # countDuplicates()
    energyHistogram(energy)
    magnetizationHistogram(magnetization)
    # print(magnetization)
    # count = 0
    # for i in range(len(magnetization)):
    #     if magnetization[i] == 15:
    #         count = count + 1
    # print(count)


if __name__ == "__main__":
    main()
