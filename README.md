<h1 align="center"> Ising Model GAN</h1>

<h3>Generative Adversarial Network for one-dimensional Ising Model</h3><br>

<h4>isingOne_v2.py</h4>
Convolutional networks generating data with a batch of 400, noise of length 1.

<h4>isingOne_Linear_[x].py</h4>
Linear (Fully connected layers) networks generating data where 'x' is length of the input noise.

<h4>testData.py</h4>
A file that tests the generated data by producing graphs and histograms.

<h4>test.py</h4>
File testing original unprocessed data. <br/>

<h4>dataGenerator.py</h4>
File generating and saving specified amount of data to be processed by testing scripts. <br/>

<h3 align="center">Charts of real data</h3>

|              Energy chart              |             Magnetization chart              |
|:--------------------------------------:|:--------------------------------------------:|
|  ![](Data/RealDataEnergyHistogram.png) |  ![](Data/RealDataMagnetizationHitogram.png) |

[//]: # (<h3 align="center">Charts of Generated data by isingOne_v2.py </h3> <h6 align="center">&#40;Convolutional networks&#41;</h6>)

[//]: # ()
[//]: # (|           Energy charts           |           Magnetization charts           |)

[//]: # (|:---------------------------------:|:----------------------------------------:|)

[//]: # (|   ![]&#40;Data/FakeDataEnergy.png&#41;    |   ![]&#40;Data/FakeDataMagnetization.png&#41;    |)

[//]: # (|         Energy Histogram          |         Magnetization Histogram          |)

[//]: # (| ![]&#40;Data/FakeEnergyHistogram.png&#41; | ![]&#40;Data/FakeMagnetizationHistogram.png&#41; |)

<h3 align="center">Charts of Generated data by Linear Models</h3> <h6 align="center">(Fully connected layers)</h6>

|                            Input noise 2                            |                           Input noise 4                            |                           Input noise 8                            |
|:-------------------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------:|
|           ![](outData[2]Linear/200-12500-2-0002/Loss.png)           |          ![](outData[4]Linear/200-12500-4-0002/Loss.png)           |          ![](outData[8]Linear/400-12500-8-0002/Loss.png)           |                                                                     |         ![](outData[32]Linear/200-12500-32-0002/Loss.png)          |
|      ![](outData[2]Linear/200-12500-2-0002/FakeEnergyHist.png)      |     ![](outData[4]Linear/200-12500-4-0002/FakeEnergyHist.png)      |     ![](outData[8]Linear/400-12500-8-0002/FakeEnergyHist.png)      |    ![](outData[16]Linear-200-12500-16-0002/FakeEnergyHist_2.png)    |    ![](outData[32]Linear/200-12500-32-0002/FakeEnergyHist.png)     |
|  ![](outData[2]Linear/200-12500-2-0002/FakeMagnetizationHist.png)   |  ![](outData[4]Linear/200-12500-4-0002/FakeMagnetizationHist.png)  |  ![](outData[4]Linear/200-12500-4-0002/FakeMagnetizationHist.png)  | ![](outData[16]Linear-200-12500-16-0002/FakeMagnetizationHist2.png) | ![](outData[32]Linear/200-12500-32-0002/FakeMagnetizationHist.png) |
|                           Input noise 16                            |                           Input noise 32                           |                           Input noise 64                           |
|          ![](outData[16]Linear-250-2000-16-0002/Loss.png)           |         ![](outData[32]Linear/200-12500-32-0002/Loss.png)          |         ![](outData[64]Linear/300-12500-64-0001/Loss.png)          |
|    ![](outData[16]Linear-200-12500-16-0002/FakeEnergyHist_2.png)    |    ![](outData[32]Linear/200-12500-32-0002/FakeEnergyHist.png)     |    ![](outData[64]Linear/300-12500-64-0001/FakeEnergyHist.png)     |
| ![](outData[16]Linear-200-12500-16-0002/FakeMagnetizationHist2.png) | ![](outData[32]Linear/200-12500-32-0002/FakeMagnetizationHist.png) | ![](outData[64]Linear/300-12500-64-0001/FakeMagnetizationHist.png) |