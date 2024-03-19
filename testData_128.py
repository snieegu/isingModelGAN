import matplotlib.pyplot as plt
import numpy as np

isingSize = 64  # <- size of ising model configuration
inputNoise = 8  # <- size of input noise
beta = "s0018"

fakeDataPath = "outIsingData/" + beta + "_128/64-" + beta + "[" + str(inputNoise) + "]_128/outputData(64-" + beta + ")TestFileLinear[" + str(inputNoise) + "]Generated.npy"  # <-path to save the data
ising_fakeData = np.load(fakeDataPath)  # <- data for the test from the generated batch data
# ising_data = np.load('outIsingData/outputDataTestFileLinear4.npy')  # <- test data coming from the generator

ising_realData = np.load("ising/isingData2/s_cfg_x064_b" + beta[1:] + ".npy")
ising_realData = np.sign(ising_realData)

print("\nsaved data path: ", fakeDataPath)

# print("clear data shape: ", ising_data.shape)

fakeData = ising_fakeData.squeeze()
realData = ising_realData.squeeze()

dataLength = len(fakeData)

realEnergy = -(realData * np.roll(realData, 1, 1)).sum(1).astype('int64')
realMagnetization = realData.sum(axis=1)

fakeEnergy = -(fakeData * np.roll(fakeData, 1, 1)).sum(1).astype('int64')
fakeMagnetization = fakeData.sum(axis=1)

energy_histogram_filename = "outIsingData/" + beta + "_128/64-" + beta + "[" + str(inputNoise) + "]_128/EnergyHistogram.png"
magnetization_histogram_filename = "outIsingData/" + beta + "_128/64-" + beta + "[" + str(
    inputNoise) + "]_128/MagnetizationHistogram.png"


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


def energyHistogram(fakeEnergy, realEnergy):
    plt.title("Energy Histogram")
    ms = np.arange(-isingSize, isingSize + 0.5, 4)
    es = np.arange(-isingSize - 2, isingSize + 2.5, 4)
    hist_fake, bins_fake = np.histogram(fakeEnergy, es, density=True)
    hist_real, bins_real = np.histogram(realEnergy, es, density=True)

    plt.xlabel(f"Histogram for {dataLength} data")
    plt.scatter(ms, hist_real, marker='+', s=500, c='red', label='theory')
    plt.hist(fakeEnergy, bins=bins_fake, density=True, alpha=0.9, label='Generated Data')
    plt.legend()
    plt.savefig(energy_histogram_filename)
    plt.show()


def magnetizationHistogram(fakeMagnetization, realMagnetization):
    plt.title("Magnetization Histogram")
    ms = np.arange(-isingSize, isingSize + 0.5, 2)
    es = np.arange(-isingSize - 1, isingSize + 1.5, 2)
    hist_fake, bins_fake = np.histogram(fakeMagnetization, es, density=True)
    hist_real, bins_real = np.histogram(realMagnetization, es, density=True)

    plt.xlabel(f"Histogram for {dataLength} data")
    plt.scatter(ms, hist_real, marker='+', s=500, c='red', label='theory')
    plt.hist(fakeMagnetization, bins=bins_fake, density=True, alpha=0.9, label='Generated Data')
    plt.legend()
    plt.savefig(magnetization_histogram_filename)
    plt.show()


def print_statistics(magnetization, energy):
    print("Magnetization mean: ", np.average(magnetization))
    print("Energy mean:", energy.mean())
    print("Magnetization variance: ", np.var(magnetization))
    print("Energy variance:", np.var(energy))
    print("Magnetization max: ", np.max(magnetization))
    print("Magnetization min: ", np.min(magnetization))
    print("Energy max:", np.max(energy))
    print("Energy min:", np.min(energy))


def main():
    # showSyntheticDataChart()
    # countDuplicates()
    energyHistogram(fakeEnergy, realEnergy)
    magnetizationHistogram(fakeMagnetization, realMagnetization)
    # count = 0
    # for i in range(len(magnetization)):
    #     if magnetization[i] == 15:
    #         count = count + 1
    # print(count)

    print("\n--- Real Data Statistics ---")
    print_statistics(realMagnetization, realEnergy)

    print("\n--- Fake Data Statistics ---")
    print_statistics(fakeMagnetization, fakeEnergy)


if __name__ == "__main__":
    main()
    exit(1)
