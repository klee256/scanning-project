
clc
clear all
close all

% filePathName=matlab.desktop.editor.getActiveFilename;
% [folderPath,~,~] = fileparts(filePathName);
% cd(folderPath)
% addpath(genpath(fileparts(pwd)));

hpc="1";
t_usage=0.5; % training usage
v_usage=0.25; % validation usage
neighborhoodSize=2;
padding='n';
lr=1.0e-4;
epochs=100;

jv1=importdata('N57_7cleanJVzoomed.mat');
mat1=importdata('N57_7deltaVzoomed.mat');
jv2=importdata('N17_6cleanJV.mat');
mat2=importdata('N17_6deltaV.mat');

[tj,tm,vj,vm] = dataset_splitter(t_usage,v_usage,neighborhoodSize,padding,jv1,mat1,jv2,mat2);

[trainingLoss,validationLoss,encoderNets] = smallNetFunct(lr,epochs,tj,tm,vj,vm);

%% HPC Config
if hpc == "1"
    poolobj = gcp('nocreate');
    delete(poolobj);
end


