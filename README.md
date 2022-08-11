<h1 align="center"> Ising Model GAN</h1>

<h3>Generative Adversarial Network for one-dimensional Ising Model</h3><br>

<h4>isingOne_v2.py</h4>
Convolutional networks generating data with a batch of 400, noise of length 1.

<h4>isingOne_Linear.py</h4>
Linear (Fully connected layers) network generating data with a batch of 200, noise of length 1.

<h4>testData.py</h4>
A file that tests the generated data and also produces graphs.

<h4>test.py</h4>
File testing original unprocessed data <br/>

<h3 align="center">Charts of real data</h3>

|               Energy charts               |              Magnetization charts               |
|:-----------------------------------------:|:-----------------------------------------------:|
|       ![](Data/RealDataEnergy.png)        |       ![](Data/RealDataMagnetization.png)       |
|       Energy Histogram for 400 data       |      Magnetization Histogram for 400 data       |
|   ![](Data/RealDataEnergyHistogram.png)   |   ![](Data/RealDataMagnetizationHitogram.png)   |
|      Energy Histogram for 2000 data       |      Magnetization Histogram for 2000 data      |
| ![](Data/RealDataEnergyHistogram2000.png) | ![](Data/RealDataMagnetizationHitogram2000.png) |

<h3 align="center">Charts of Generated data by isingOne_v2.py (Convolutional networks)</h3>

|           Energy charts           |           Magnetization charts           |
|:---------------------------------:|:----------------------------------------:|
|   ![](Data/FakeDataEnergy.png)    |   ![](Data/FakeDataMagnetization.png)    |
|   Energy Histogram for 400 data   |   Magnetization Histogram for 400 data   |
| ![](Data/FakeEnergyHistogram.png) | ![](Data/FakeMagnetizationHistogram.png) |

<h3 align="center">Charts of Generated data by isingOne_Linear.py (Fully connected layers)</h3>

|                   Energy charts                    |                   Magnetization charts                    |
|:--------------------------------------------------:|:---------------------------------------------------------:|
|   ![](outDataLinear-7-200-1-0005/FakeEnergy.png)   |   ![](outDataLinear-7-200-1-0005/FakeMagnetization.png)   |
|           Energy Histogram for 200 data            |           Magnetization Histogram for 200 data            |
| ![](outDataLinear-7-200-1-0005/FakeEnergyHist.png) | ![](outDataLinear-7-200-1-0005/FakeMagnetizationHist.png) |




