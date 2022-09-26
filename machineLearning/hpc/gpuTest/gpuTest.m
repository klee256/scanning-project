clc
clear all
close all

availableGPUs = gpuDeviceCount("available");
disp(availableGPUs);

parpool('local',availableGPUs);

[availableGPUs,gpuIndx] = gpuDeviceCount("available")

gpu = gpuDevice();
fprintf('Using a %s GPU.\n', gpu.Name);
disp(gpuDevice);

delete(gcp('nocreate'));
