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

<h3 align="center">Charts of Generated data (configuration 16) by Linear Models</h3> <h6 align="center">(Fully connected layers)</h6>

|                            Input noise 2                            |                           Input noise 4                            |                           Input noise 8                            |
|:-------------------------------------------------------------------:|:------------------------------------------------------------------:|:------------------------------------------------------------------:|
|           ![](outData[2]Linear/200-12500-2-0002/Loss.png)           |          ![](outData[4]Linear/200-12500-4-0002/Loss.png)           |          ![](outData[8]Linear/400-12500-8-0002/Loss.png)           |
|      ![](outData[2]Linear/200-12500-2-0002/FakeEnergyHist.png)      |     ![](outData[4]Linear/200-12500-4-0002/FakeEnergyHist.png)      |     ![](outData[8]Linear/400-12500-8-0002/FakeEnergyHist.png)      |
|  ![](outData[2]Linear/200-12500-2-0002/FakeMagnetizationHist.png)   |  ![](outData[4]Linear/200-12500-4-0002/FakeMagnetizationHist.png)  |  ![](outData[4]Linear/200-12500-4-0002/FakeMagnetizationHist.png)  |
|                           Input noise 16                            |                           Input noise 32                           |                           Input noise 64                           |
|          ![](outData[16]Linear-250-2000-16-0002/Loss.png)           |         ![](outData[32]Linear/200-12500-32-0002/Loss.png)          |         ![](outData[64]Linear/300-12500-64-0001/Loss.png)          |
|    ![](outData[16]Linear-200-12500-16-0002/FakeEnergyHist_2.png)    |    ![](outData[32]Linear/200-12500-32-0002/FakeEnergyHist.png)     |    ![](outData[64]Linear/300-12500-64-0001/FakeEnergyHist.png)     |
| ![](outData[16]Linear-200-12500-16-0002/FakeMagnetizationHist2.png) | ![](outData[32]Linear/200-12500-32-0002/FakeMagnetizationHist.png) | ![](outData[64]Linear/300-12500-64-0001/FakeMagnetizationHist.png) |


<h3 align="center">Charts of Generated data (configuration 32) by Linear Models</h3> <h6 align="center">(Fully connected layers)</h6>

|                                       Input noise 8                                       |                                       Input noise 16                                        |                                       Input noise 32                                        |                                       Input noise 64                                        |
|:-----------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
|         ![](isingOutData/(32)/outData(32)Linear[8]/300-12500-32-8-0002/Loss.png)          |         ![](isingOutData/(32)/outData(32)Linear[16]/200-12500-32-16-0002/Loss.png)          |         ![](isingOutData/(32)/outData(32)Linear[32]/300-12500-32-32-0001/Loss.png)          |          ![](isingOutData/(32)/outData(32)Linear[64]/300-12500-32-64-0001/Loss.png)         |
|    ![](isingOutData/(32)/outData(32)Linear[8]/300-12500-32-8-0002/FakeEnergyHist.png)     |    ![](isingOutData/(32)/outData(32)Linear[16]/200-12500-32-16-0002/FakeEnergyHist.png)     |    ![](isingOutData/(32)/outData(32)Linear[32]/300-12500-32-32-0001/FakeEnergyHist.png)     |    ![](isingOutData/(32)/outData(32)Linear[64]/300-12500-32-64-0001/FakeEnergyHist.png)     |
| ![](isingOutData/(32)/outData(32)Linear[8]/300-12500-32-8-0002/FakeMagnetizationHist.png) | ![](isingOutData/(32)/outData(32)Linear[16]/200-12500-32-16-0002/FakeMagnetizationHist.png) | ![](isingOutData/(32)/outData(32)Linear[32]/300-12500-32-32-0001/FakeMagnetizationHist.png) | ![](isingOutData/(32)/outData(32)Linear[64]/300-12500-32-64-0001/FakeMagnetizationHist.png) |



<h3 align="center">Charts of Generated data (configuration 64) by Linear Models</h3> <h6 align="center">(Fully connected layers)</h6>

|                                       Input noise 8                                        |                                       Input noise 16                                        |                                       Input noise 32                                        |                                       Input noise 64                                        |
|:------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
|          ![](isingOutData/(64)/outData(64)Linear[8]/300-12500-64-8-0002/Loss.png)          |         ![](isingOutData/(64)/outData(64)Linear[16]/300-12500-64-16-0002/Loss.png)          |         ![](isingOutData/(64)/outData(64)Linear[32]/300-12500-64-32-0002/Loss.png)          |         ![](isingOutData/(64)/outData(64)Linear[64]/200-12500-64-64-0002/Loss.png)          |
|     ![](isingOutData/(64)/outData(64)Linear[8]/300-12500-64-8-0002/FakeEnergyHist.png)     |    ![](isingOutData/(64)/outData(64)Linear[16]/300-12500-64-16-0002/FakeEnergyHist.png)     |    ![](isingOutData/(64)/outData(64)Linear[32]/300-12500-64-32-0002/FakeEnergyHist.png)     |    ![](isingOutData/(64)/outData(64)Linear[64]/200-12500-64-64-0002/FakeEnergyHist.png)     |
|  ![](isingOutData/(64)/outData(64)Linear[8]/300-12500-64-8-0002/FakeMagnetizationHist.png) | ![](isingOutData/(64)/outData(64)Linear[16]/300-12500-64-16-0002/FakeMagnetizationHist.png) | ![](isingOutData/(64)/outData(64)Linear[32]/300-12500-64-32-0002/FakeMagnetizationHist.png) | ![](isingOutData/(64)/outData(64)Linear[64]/200-12500-64-64-0002/FakeMagnetizationHist.png) |



<h3 align="center">Charts of Generated data (configuration 128) by Linear Models</h3> <h6 align="center">(Fully connected layers)</h6>

|                                       Input noise 8                                        |                                        Input noise 16                                         |                                        Input noise 32                                        |                                        Input noise 64                                        |
|:------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
|         ![](isingOutData/128/outData(128)Linear[8]/300-12500-128-8-0002/Loss.png)          |          ![](isingOutData/128/outData(128)Linear[16]/300-12500-128-16-0002/Loss.png)          |         ![](isingOutData/128/outData(128)Linear[32]/200-12500-128-32-0002/Loss.png)          |         ![](isingOutData/128/outData(128)Linear[64]/300-12500-128-64-0002/Loss.png)          |
|    ![](isingOutData/128/outData(128)Linear[8]/300-12500-128-8-0002/FakeEnergyHist.png)     |     ![](isingOutData/128/outData(128)Linear[16]/300-12500-128-16-0002/FakeEnergyHist.png)     |    ![](isingOutData/128/outData(128)Linear[32]/200-12500-128-32-0002/FakeEnergyHist.png)     |    ![](isingOutData/128/outData(128)Linear[64]/300-12500-128-64-0002/FakeEnergyHist.png)     |
| ![](isingOutData/128/outData(128)Linear[8]/300-12500-128-8-0002/FakeMagnetizationHist.png) |  ![](isingOutData/128/outData(128)Linear[16]/300-12500-128-16-0002/FakeMagnetizationHist.png) | ![](isingOutData/128/outData(128)Linear[32]/200-12500-128-32-0002/FakeMagnetizationHist.png) | ![](isingOutData/128/outData(128)Linear[64]/300-12500-128-64-0002/FakeMagnetizationHist.png) |



<h2 align="center">-----------------NEW DATA----------------</h2>

<h2  align="center">Configuration 64</h2>

<h3 align="center">Charts of Generated data by Linear Models</h3> <h6 align="center">(Fully connected layers)</h6>

<h4>Beta: 0018</h4>

|                          Input noise 2                          |                          Input noise 4                          |                           Input noise 8                           |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:-----------------------------------------------------------------:|
|        ![](outIsingData/s0018/64-s0018[2]/Training.png)         |        ![](outIsingData/s0018/64-s0018[4]/Training.png)         |         ![](outIsingData/s0018/64-s0018[8]/Training.png)          |
|     ![](outIsingData/s0018/64-s0018[2]/EnergyHistogram.png)     |     ![](outIsingData/s0018/64-s0018[4]/EnergyHistogram.png)     |      ![](outIsingData/s0018/64-s0018[8]/EnergyHistogram.png)      |
| ![](outIsingData/s0018/64-s0018[2]/MagnetizationHistogram.png)  | ![](outIsingData/s0018/64-s0018[4]/MagnetizationHistogram.png)  |  ![](outIsingData/s0018/64-s0018[8]/MagnetizationHistogram.png)   |
|                         Input noise 16                          |                         Input noise 32                          |                          Input noise 64                           |
|        ![](outIsingData/s0018/64-s0018[16]/Training.png)        |        ![](outIsingData/s0018/64-s0018[32]/Training.png)        |         ![](outIsingData/s0018/64-s0018[64]/Training.png)         |
|    ![](outIsingData/s0018/64-s0018[16]/EnergyHistogram.png)     |    ![](outIsingData/s0018/64-s0018[32]/EnergyHistogram.png)     |     ![](outIsingData/s0018/64-s0018[64]/EnergyHistogram.png)      |
| ![](outIsingData/s0018/64-s0018[16]/MagnetizationHistogram.png) | ![](outIsingData/s0018/64-s0018[32]/MagnetizationHistogram.png) |  ![](outIsingData/s0018/64-s0018[64]/MagnetizationHistogram.png)  |

<br>
<br>
<h4>Beta: 0136</h4>

|                          Input noise 2                          |                           Input noise 4                           |                           Input noise 8                           |
|:---------------------------------------------------------------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------:|
|        ![](outIsingData/s0136/64-s0136[2]/Training.png)         |         ![](outIsingData/s0136/64-s0136[4]/Training.png)          |         ![](outIsingData/s0136/64-s0136[8]/Training.png)          |
|     ![](outIsingData/s0136/64-s0136[2]/EnergyHistogram.png)     |      ![](outIsingData/s0136/64-s0136[4]/EnergyHistogram.png)      |      ![](outIsingData/s0136/64-s0136[8]/EnergyHistogram.png)      |
| ![](outIsingData/s0136/64-s0136[2]/MagnetizationHistogram.png)  |  ![](outIsingData/s0136/64-s0136[4]/MagnetizationHistogram.png)   |  ![](outIsingData/s0136/64-s0136[8]/MagnetizationHistogram.png)   |
|                         Input noise 16                          |                          Input noise 32                           |                          Input noise 64                           |
|        ![](outIsingData/s0136/64-s0136[16]/Training.png)        |         ![](outIsingData/s0136/64-s0136[32]/Training.png)         |         ![](outIsingData/s0136/64-s0136[64]/Training.png)         |
|    ![](outIsingData/s0136/64-s0136[16]/EnergyHistogram.png)     |     ![](outIsingData/s0136/64-s0136[32]/EnergyHistogram.png)      |     ![](outIsingData/s0136/64-s0136[64]/EnergyHistogram.png)      |
| ![](outIsingData/s0136/64-s0136[16]/MagnetizationHistogram.png) |  ![](outIsingData/s0136/64-s0136[32]/MagnetizationHistogram.png)  |  ![](outIsingData/s0136/64-s0136[64]/MagnetizationHistogram.png)  |

<br>
<br>
<h4>Beta: 0386</h4>

|                          Input noise 2                          |                           Input noise 4                           |                           Input noise 8                           |
|:---------------------------------------------------------------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------:|
|        ![](outIsingData/s0386/64-s0386[2]/Training.png)         |         ![](outIsingData/s0386/64-s0386[4]/Training.png)          |          ![](outIsingData/s0386/64-s0386[8]/Training.png)         |
|     ![](outIsingData/s0386/64-s0386[2]/EnergyHistogram.png)     |      ![](outIsingData/s0386/64-s0386[4]/EnergyHistogram.png)      |      ![](outIsingData/s0386/64-s0386[8]/EnergyHistogram.png)      |
| ![](outIsingData/s0386/64-s0386[2]/MagnetizationHistogram.png)  |  ![](outIsingData/s0386/64-s0386[4]/MagnetizationHistogram.png)   |  ![](outIsingData/s0386/64-s0386[8]/MagnetizationHistogram.png)   |
|                         Input noise 16                          |                          Input noise 32                           |                          Input noise 64                           |
|        ![](outIsingData/s0386/64-s0386[16]/Training.png)        |         ![](outIsingData/s0386/64-s0386[32]/Training.png)         |         ![](outIsingData/s0386/64-s0386[64]/Training.png)         |
|    ![](outIsingData/s0386/64-s0386[16]/EnergyHistogram.png)     |     ![](outIsingData/s0386/64-s0386[32]/EnergyHistogram.png)      |     ![](outIsingData/s0386/64-s0386[64]/EnergyHistogram.png)      |
| ![](outIsingData/s0386/64-s0386[16]/MagnetizationHistogram.png) |  ![](outIsingData/s0386/64-s0386[32]/MagnetizationHistogram.png)  |  ![](outIsingData/s0386/64-s0386[64]/MagnetizationHistogram.png)  |

<br>
<br>
<h4>Beta: 0703</h4>

|                          Input noise 2                          |                           Input noise 4                           |                           Input noise 8                           |
|:---------------------------------------------------------------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------:|
|        ![](outIsingData/s0703/64-s0703[2]/Training.png)         |         ![](outIsingData/s0703/64-s0703[4]/Training.png)          |         ![](outIsingData/s0703/64-s0703[8]/Training.png)          |
|     ![](outIsingData/s0703/64-s0703[2]/EnergyHistogram.png)     |      ![](outIsingData/s0703/64-s0703[4]/EnergyHistogram.png)      |      ![](outIsingData/s0703/64-s0703[8]/EnergyHistogram.png)      |
| ![](outIsingData/s0703/64-s0703[2]/MagnetizationHistogram.png)  |  ![](outIsingData/s0703/64-s0703[4]/MagnetizationHistogram.png)   |  ![](outIsingData/s0703/64-s0703[8]/MagnetizationHistogram.png)   |
|                         Input noise 16                          |                          Input noise 32                           |                          Input noise 64                           |
|        ![](outIsingData/s0703/64-s0703[16]/Training.png)        |         ![](outIsingData/s0703/64-s0703[32]/Training.png)         |         ![](outIsingData/s0703/64-s0703[64]/Training.png)         |
|    ![](outIsingData/s0703/64-s0703[16]/EnergyHistogram.png)     |     ![](outIsingData/s0703/64-s0703[32]/EnergyHistogram.png)      |     ![](outIsingData/s0703/64-s0703[64]/EnergyHistogram.png)      |
| ![](outIsingData/s0703/64-s0703[16]/MagnetizationHistogram.png) |  ![](outIsingData/s0703/64-s0703[32]/MagnetizationHistogram.png)  |  ![](outIsingData/s0703/64-s0703[64]/MagnetizationHistogram.png)  |

<br>
<br>
<h4>Beta: 1042</h4>

|                          Input noise 2                          |                           Input noise 4                           |                          Input noise 8                           |
|:---------------------------------------------------------------:|:-----------------------------------------------------------------:|:----------------------------------------------------------------:|
|        ![](outIsingData/s1042/64-s1042[2]/Training.png)         |         ![](outIsingData/s1042/64-s1042[4]/Training.png)          |         ![](outIsingData/s1042/64-s1042[8]/Training.png)         |
|     ![](outIsingData/s1042/64-s1042[2]/EnergyHistogram.png)     |      ![](outIsingData/s1042/64-s1042[4]/EnergyHistogram.png)      |     ![](outIsingData/s1042/64-s1042[8]/EnergyHistogram.png)      |
| ![](outIsingData/s1042/64-s1042[2]/MagnetizationHistogram.png)  |  ![](outIsingData/s1042/64-s1042[4]/MagnetizationHistogram.png)   |  ![](outIsingData/s1042/64-s1042[8]/MagnetizationHistogram.png)  |
|                         Input noise 16                          |                          Input noise 32                           |                          Input noise 64                          |
|        ![](outIsingData/s1042/64-s1042[16]/Training.png)        |         ![](outIsingData/s1042/64-s1042[32]/Training.png)         |        ![](outIsingData/s1042/64-s1042[64]/Training.png)         |
|    ![](outIsingData/s1042/64-s1042[16]/EnergyHistogram.png)     |     ![](outIsingData/s1042/64-s1042[32]/EnergyHistogram.png)      |     ![](outIsingData/s1042/64-s1042[64]/EnergyHistogram.png)     |
| ![](outIsingData/s1042/64-s1042[16]/MagnetizationHistogram.png) |  ![](outIsingData/s1042/64-s1042[32]/MagnetizationHistogram.png)  | ![](outIsingData/s1042/64-s1042[64]/MagnetizationHistogram.png)  |

<br>
<br>
<h4>Beta: 1387</h4>

|                          Input noise 2                          |                          Input noise 4                           |                          Input noise 8                          |
|:---------------------------------------------------------------:|:----------------------------------------------------------------:|:---------------------------------------------------------------:|
|        ![](outIsingData/s1387/64-s1387[2]/Training.png)         |         ![](outIsingData/s1387/64-s1387[4]/Training.png)         |        ![](outIsingData/s1387/64-s1387[8]/Training.png)         |
|     ![](outIsingData/s1387/64-s1387[2]/EnergyHistogram.png)     |     ![](outIsingData/s1387/64-s1387[4]/EnergyHistogram.png)      |     ![](outIsingData/s1387/64-s1387[8]/EnergyHistogram.png)     |
| ![](outIsingData/s1387/64-s1387[2]/MagnetizationHistogram.png)  |  ![](outIsingData/s1387/64-s1387[4]/MagnetizationHistogram.png)  | ![](outIsingData/s1387/64-s1387[8]/MagnetizationHistogram.png)  |
|                         Input noise 16                          |                          Input noise 32                          |                         Input noise 64                          |
|        ![](outIsingData/s1387/64-s1387[16]/Training.png)        |        ![](outIsingData/s1387/64-s1387[32]/Training.png)         |        ![](outIsingData/s1387/64-s1387[64]/Training.png)        |
|    ![](outIsingData/s1387/64-s1387[16]/EnergyHistogram.png)     |     ![](outIsingData/s1387/64-s1387[32]/EnergyHistogram.png)     |    ![](outIsingData/s1387/64-s1387[64]/EnergyHistogram.png)     |
| ![](outIsingData/s1387/64-s1387[16]/MagnetizationHistogram.png) | ![](outIsingData/s1387/64-s1387[32]/MagnetizationHistogram.png)  | ![](outIsingData/s1387/64-s1387[64]/MagnetizationHistogram.png) |

<br>
<br>
<h4>Beta: 1733</h4>

|                          Input noise 2                          |                          Input noise 4                          |                          Input noise 8                           |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:----------------------------------------------------------------:|
|        ![](outIsingData/s1733/64-s1733[2]/Training.png)         |        ![](outIsingData/s1733/64-s1733[4]/Training.png)         |         ![](outIsingData/s1733/64-s1733[8]/Training.png)         |
|     ![](outIsingData/s1733/64-s1733[2]/EnergyHistogram.png)     |     ![](outIsingData/s1733/64-s1733[4]/EnergyHistogram.png)     |     ![](outIsingData/s1733/64-s1733[8]/EnergyHistogram.png)      |
| ![](outIsingData/s1733/64-s1733[2]/MagnetizationHistogram.png)  | ![](outIsingData/s1733/64-s1733[4]/MagnetizationHistogram.png)  |  ![](outIsingData/s1733/64-s1733[8]/MagnetizationHistogram.png)  |
|                         Input noise 16                          |                         Input noise 32                          |                          Input noise 64                          |
|        ![](outIsingData/s1733/64-s1733[16]/Training.png)        |        ![](outIsingData/s1733/64-s1733[32]/Training.png)        |        ![](outIsingData/s1733/64-s1733[64]/Training.png)         |
|    ![](outIsingData/s1733/64-s1733[16]/EnergyHistogram.png)     |    ![](outIsingData/s1733/64-s1733[32]/EnergyHistogram.png)     |     ![](outIsingData/s1733/64-s1733[64]/EnergyHistogram.png)     |
| ![](outIsingData/s1733/64-s1733[16]/MagnetizationHistogram.png) | ![](outIsingData/s1733/64-s1733[32]/MagnetizationHistogram.png) | ![](outIsingData/s1733/64-s1733[64]/MagnetizationHistogram.png)  |

<br>
<br>
<h4>Beta: 2079</h4>

|                          Input noise 2                          |                          Input noise 4                          |                          Input noise 8                           |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:----------------------------------------------------------------:|
|        ![](outIsingData/s2079/64-s2079[2]/Training.png)         |        ![](outIsingData/s2079/64-s2079[4]/Training.png)         |         ![](outIsingData/s2079/64-s2079[8]/Training.png)         |
|     ![](outIsingData/s2079/64-s2079[2]/EnergyHistogram.png)     |     ![](outIsingData/s2079/64-s2079[4]/EnergyHistogram.png)     |     ![](outIsingData/s2079/64-s2079[8]/EnergyHistogram.png)      |
| ![](outIsingData/s2079/64-s2079[2]/MagnetizationHistogram.png)  | ![](outIsingData/s2079/64-s2079[4]/MagnetizationHistogram.png)  |  ![](outIsingData/s2079/64-s2079[8]/MagnetizationHistogram.png)  |
|                         Input noise 16                          |                         Input noise 32                          |                          Input noise 64                          |
|        ![](outIsingData/s2079/64-s2079[16]/Training.png)        |        ![](outIsingData/s2079/64-s2079[32]/Training.png)        |        ![](outIsingData/s2079/64-s2079[64]/Training.png)         |
|    ![](outIsingData/s2079/64-s2079[16]/EnergyHistogram.png)     |    ![](outIsingData/s2079/64-s2079[32]/EnergyHistogram.png)     |     ![](outIsingData/s2079/64-s2079[64]/EnergyHistogram.png)     |
| ![](outIsingData/s2079/64-s2079[16]/MagnetizationHistogram.png) | ![](outIsingData/s2079/64-s2079[32]/MagnetizationHistogram.png) | ![](outIsingData/s2079/64-s2079[64]/MagnetizationHistogram.png)  |
