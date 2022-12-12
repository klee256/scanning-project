% 20221126 

clc
clear all
close all

% filePathName=matlab.desktop.editor.getActiveFilename;
% [folderPath,~,~] = fileparts(filePathName);
% cd(folderPath)
% 
% addpath(genpath(fileparts(pwd)));


hpc="1";
t_usage=0.6;
neighborhoodSize=2;
padding='n';

lr=1.0e-4;
epochs=175;


% if hpc == "1"
%     pc=parcluster('local');
%     parpool(pc,4)
%     %parpool(pc,48) % max is 48 on the cluster, verified by admin
% end

jv1=importdata('N57_7cleanJVzoomed.mat');
mat1=importdata('N57_7deltaVzoomed.mat');
jv2=importdata('N17_6cleanJV.mat'); 
mat2=importdata('N17_6deltaV.mat');

[tj,tm,vj,vm]=dataset_splitter(t_usage,neighborhoodSize,padding,jv1,mat1,jv2,mat2);

[trainingLoss,validationLoss,encoderNets] = smallNetFunct(lr,epochs,tj,tm,vj,vm);

% %paramss=

%% HPC Config
if hpc == "1"
    poolobj = gcp('nocreate');
    delete(poolobj);
end



