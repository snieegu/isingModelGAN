import matplotlib.pyplot as plt
import numpy as np

isingSize = 16  # <- size of ising model configuration
inputNoise = 32  # <- size of input noise
beta = "s0100"

fakeDataPath = "outIsingData/" + beta + "_x016/16-" + beta + "[" + str(
    inputNoise) + "]/outputData(16-" + beta + ")TestFileLinear[" + str(inputNoise) + "].npy"  # <-path to save the data
ising_fakeData = np.load(fakeDataPath)  # <- data for the test from the generated batch data
# ising_data = np.load('outIsingData/outputDataTestFileLinear4.npy')  # <- test data coming from the generator

ising_realData = np.load("ising/s_cfg_x016_b" + beta[1:] + ".npy")
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

energy_histogram_filename = "outIsingData/" + beta + "_x016/16-" + beta + "[" + str(
    inputNoise) + "]/EnergyHistogramWithError.png"
magnetization_histogram_filename = "outIsingData/" + beta + "_x016/16-" + beta + "[" + str(
    inputNoise) + "]/MagnetizationHistogramWithError.png"


def calculate_std_error(hist_counts):
    # Obliczenie błędu standardowego przy użyciu statystyki poissonowskiej
    return np.sqrt(hist_counts)


def histogram_with_errors(data, edges, density, label):
    hist_counts, _ = np.histogram(data, edges, density=density)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    std_errors = calculate_std_error(hist_counts)

    if density:
        # Normalize std error if histogram is density
        area = np.diff(edges).sum()
        std_errors = std_errors / (hist_counts.sum() * area)

    # Użycie funkcji errorbar do narysowania histogramu z niepewnościami
    plt.errorbar(bin_centers, hist_counts, yerr=std_errors, fmt='o', label=label)


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
    plt.figure(figsize=(8, 6))
    plt.title(f"Energy Histogram for {dataLength} data")
    es = np.arange(-isingSize - 2, isingSize + 2.5, 4)
    histogram_with_errors(fakeEnergy, es, density=True, label='Generated Data')
    histogram_with_errors(realEnergy, es, density=True, label='Theory')
    plt.xlabel("Energy")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(energy_histogram_filename)
    plt.show()


def magnetizationHistogram(fakeMagnetization, realMagnetization):
    plt.figure(figsize=(8, 6))
    plt.title(f"Magnetization Histogram for {dataLength} data")
    es = np.arange(-isingSize - 1, isingSize + 1.5, 2)
    histogram_with_errors(fakeMagnetization, es, density=True, label='Generated Data')
    histogram_with_errors(realMagnetization, es, density=True, label='Theory')
    plt.xlabel("Magnetization")
    plt.ylabel("Probability Density")
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
    energyHistogram(fakeEnergy, realEnergy)
    magnetizationHistogram(fakeMagnetization, realMagnetization)

    print("\n--- Real Data Statistics ---")
    print_statistics(realMagnetization, realEnergy)

    print("\n--- Fake Data Statistics ---")
    print_statistics(fakeMagnetization, fakeEnergy)


if __name__ == "__main__":
    main()
    exit(1)
