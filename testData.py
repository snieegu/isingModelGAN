import matplotlib.pyplot as plt
import numpy as np
# from isingOne_Linear_64 import noise_dim, beta, latent_dim
from scipy.special import binom

latent_dim = 64  # <- size of ising model configuration
noise_dim = 32  # <- size of input noise
beta = "s0100"

ising_fakeData = np.load(
    "outIsingData/" + beta + "/" + str(latent_dim) + "-" + beta + "[" + str(noise_dim) + "]/outputData(" + str(
        latent_dim) + "-" + beta + ")TestFileLinear[" + str(
        noise_dim) + "]Generated.npy")  # <- data for the test from the generated batch data
# ising_data = np.load('outIsingData/outputDataTestFileLinear4.npy')  # <- test data coming from the generator
ising_realData = np.load("ising/s_cfg_x064_b" + beta[1:] + ".npy")
ising_realData = np.sign(ising_realData)

# print("clear data shape: ", ising_data.shape)
isingSize = 64  # <- modify configuration length based on input

fakeData = ising_fakeData.squeeze()
realData = ising_realData.squeeze()

dataLength = len(fakeData)

realEnergy = -(realData * np.roll(realData, 1, 1)).sum(1).astype('int64')
realMagnetization = realData.sum(axis=1)

fakeEnergy = -(fakeData * np.roll(fakeData, 1, 1)).sum(1).astype('int64')
fakeMagnetization = fakeData.sum(axis=1)


def showSyntheticDataChart():
    plt.figure(figsize=(7, 5))
    plt.title("Fake data Magnetization")
    plt.plot(fakeMagnetization, label="Magnetization")
    plt.xlabel("")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.show()

    averageMagnetization = np.asarray(fakeMagnetization)
    print("Fake Magnetization mean: ", np.average(averageMagnetization))
    print("Fake Energy mean:", fakeEnergy.mean())

    plt.figure(figsize=(7, 5))
    plt.title("Fake data Energy")
    plt.plot(fakeEnergy, label="Energy")
    plt.xlabel("")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()


def countDuplicates():
    resultList = fakeData.tolist()
    duplicates = {tuple(x) for x in resultList if resultList.count(x) > 1}
    duplicatesCount = len(duplicates)
    print("duplicates: ", duplicatesCount, "/", dataLength)


def energyHistogram(InData):
    plt.title("Energy Histogram")
    ms = np.arange(-isingSize, isingSize + 0.5, 4)
    es = np.arange(-isingSize - 2, isingSize + 2.5, 4)
    hist, bins = np.histogram(realEnergy, es)
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
    energyHistogram(fakeEnergy)
    magnetizationHistogram(fakeMagnetization)
    # count = 0
    # for i in range(len(magnetization)):
    #     if magnetization[i] == 15:
    #         count = count + 1
    # print(count)


if __name__ == "__main__":
    main()
