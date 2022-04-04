import numpy
import numpy as np
import matplotlib.pyplot as plt
import sys

numpy.set_printoptions(threshold=sys.maxsize)

energy = []
temperature = []

ising_data = np.load('outIsing/outputData1epoch.npy')
data = ising_data.squeeze()
print(data)
# print(data.shape)
print(type(data))

for i in range(len(data)):
    temperature.append(data[i].sum())

plt.figure(figsize=(7, 5))
plt.title("Temperature")
plt.plot(temperature, label="Temperature")
plt.xlabel("")
plt.ylabel("Temperature")
plt.legend()
plt.show()

averageTemp = numpy.asarray(temperature)
print("Average temperature: ", np.average(averageTemp))

energySum = 0
duplicates = 0
for i in range(100):
    for j in range(0, 15):
        neighbor = data[i][j - 1] + data[0][j]
        if neighbor == 0:
            energySum = energySum + 1
        else:
            energySum = energySum - 1
    energy.append(energySum)
    energySum = 0

averageEnergy = numpy.asarray(energy)
print("Average energy: ", np.average(energy))

plt.figure(figsize=(7, 5))
plt.title("Energy")
plt.plot(energy, label="Energy")
plt.xlabel("")
plt.ylabel("Energy")
plt.legend()
plt.show()

n = 98
for i in range(len(data) - 1):
    for j in range(n):
        if np.array_equal(data[i], data[j + 1]):
            duplicates = duplicates + 1

    n = n - 1
print("Duplicates", duplicates)
