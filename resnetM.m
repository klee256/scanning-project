clc
clear all
close all

tic

hpc="0";

%% HPC Config
if hpc == "1"
    pc=parcluster('local');
    parpool(pc,48) % max is 48 on the cluster, verified by admin
end

%% Data Selector (change value of "selector" to desired material paramter)
% 1 - deltaV
% 2 - deltaQ
% 3 - dos (n)
% 4 - pl
% 5 - mu

selector=1;
padding='n'; % must be either 'y' or 'n' NOT "Y" or "y"
trainName="N57_7";
validName="N17_6";

% Device Names:
% "N57_7"
% "N17_6"
% "P58_6"
% "P61_10"
% "P61_11"

switch selector
    case 1
        matstr="*deltaV*";
    case 2
        matstr="*deltaQ*";
    case 3
        matstr="*dos*";
    case 4
        matstr="*plA*";
    case 5
        matstr="*mu*";
    otherwise
        disp('Error, invalid selection\n')
end

%% Local repository
if hpc == "0"
    addpath(genpath(fileparts(pwd)));
    oldFolder1=pwd;
    cd("processedData")
    oldFolder2=pwd;
    cd(dir(trainName+"*").name)
    tJV=importdata(dir("*JV*").name);
    tMat=importdata(dir(matstr).name);
    cd(oldFolder2)
    cd(dir(validName+"*").name)
    vJV=importdata(dir("*JV*").name);
    vMat=importdata(dir(matstr).name);
    cd(oldFolder1)
end

%% HPC repository
if hpc == "1"
    tJV=importdata(dir(trainName+"*JV*").name);
    tMat=importdata(dir(trainName+matstr).name);
    vJV=importdata(dir(validName+"*JV*").name);
    vMat=importdata(dir(validName+matstr).name);
end
%
[trainingJV,trainingMat]=neighborhood(tJV,tMat,2,padding);
[validationJV,validationMat]=neighborhood(vJV,vMat,2,padding,tJV);

% Unit Testing
smallUnit=1:2;
trainingJV=trainingJV(:,smallUnit); trainingMat=trainingMat(smallUnit);
validationJV=validationJV(:,smallUnit); validationMat=validationMat(smallUnit);

%% Training 

% Sweeps
% lr=linspace(1.0e-6,1.0e-3,48);
% tLoss=cell(1,length(lr)); vLoss=cell(1,length(lr)); encs=cell(1,length(lr));
% parfor xy=1:length(lr)
%     [trainingLoss,validationLoss,encoderNets] = resnetFunct(lr(xy),200,trainingJV,trainingMat,validationJV,validationMat);
%     tLoss{xy}=trainingLoss; vLoss{xy}=validationLoss; encs{xy}=encoderNets;
% end

% Single
lr=1.0e-4;
[trainingLoss,validationLoss,encoderNets] = resnetFunct(lr,2,trainingJV,trainingMat,validationJV,validationMat);

toc

% figure(); plot(trainingLoss); hold on; plot(validationLoss); legend('train','validation')

%% HPC Config
if hpc == "1"
    poolobj = gcp('nocreate');
    delete(poolobj);
end


