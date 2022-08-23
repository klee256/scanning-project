# Scanning Project

This is the repo for the scanning project. The goal of this project is to be able to predict the materials parameters of PbS colloidal quantum dot thin film solar cells from JV curves.

Folder Directory:
* data contains measured parameters
* machineLearning contains HPC code and ML related programs
* requiredFunctions is used for data calculations
* tools contains programs used to perform ancillary tasks
* visualizations contains programs to visualize learned features, etc

Requires the following MATLAB toolboxes:
* Simulink                                              Version 10.5        (R2022a)
* Communications Toolbox                                Version 7.7         (R2022a)
* Curve Fitting Toolbox                                 Version 3.7         (R2022a)
* DSP System Toolbox                                    Version 9.14        (R2022a)
* Deep Learning HDL Toolbox                             Version 1.3         (R2022a)
* Deep Learning Toolbox                                 Version 14.4        (R2022a)
* Image Processing Toolbox                              Version 11.5        (R2022a)
* Parallel Computing Toolbox                            Version 7.6         (R2022a)
* Reinforcement Learning Toolbox                        Version 2.2         (R2022a)
* Signal Processing Toolbox                             Version 9.0         (R2022a)
* Statistics and Machine Learning Toolbox               Version 12.3        (R2022a)

Notes:
* High minibatch values won't work on the training dataset, best to update with one sample at a time 
