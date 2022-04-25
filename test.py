import numpy
import numpy as np
import matplotlib.pyplot as plt

energy = []
temperature = []

ising_data = np.load('ising/s_cfg_x016_b0100.npy')
data = ising_data.squeeze()
data = data[0:1000]
print(data.shape)

for i in range(len(data)):
    temperature.append(data[i].sum())

plt.figure(figsize=(7, 5))
plt.title("Real data Magety")
plt.plot(temperature, label="Temperature")
plt.xlabel("")
plt.ylabel("Temperature")
plt.legend()
plt.show()

averageTemp = numpy.asarray(temperature)
print("Average temperature: ", np.average(averageTemp))

energySum = 0
for i in range(len(data)):
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
plt.title("Real data Energy")
plt.plot(energy, label="Energy")
plt.xlabel("")
plt.ylabel("Energy")
plt.legend()
plt.show()

resultList = data.tolist()
duplicates = {tuple(_) for _ in resultList if resultList.count(_) > 1}
duplicatesCount = len(duplicates)
print("duplicates: ", duplicatesCount)
