
% main.m 
% This program is for 5 parameter predictions at a single time

clear all
close all

%% user settings 

% profile on % checks performance and memory usage
hpc='y';
padding='n';
testData_split=0.1;
neighborhoodSize=2;
numFolds=5; % k-fold cross validation
nEpochs = 100;

smallDataSet='n';

%%

[folderPath,~,~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(folderPath)
%addpath(genpath(fileparts(pwd)));
addpath('data')
addpath('functions')



%{
    All data must be stored in a .mat file with the name convention: XYY_Yz.mat
    where X is a capital letter, YY_Y are numbers (first two YY is the device #
    the second Y is the pixel #), and z is the material parameter. 

    Allowed string values for z:
    plA - photoluminescence
    deltaV - transient photovoltage
    deltaQ - transient photocurrent
    dos - density of states (electronic midgap trap state density)
    jv - illuminated current voltage
%}
 
file_list = dir(fullfile('data','**/*.mat')); 

% Loading data from all the.mat files 
data = cellfun(@(a) load(a), strcat({file_list.folder},'/',{file_list.name}), 'UniformOutput', false).';
data = cellfun(@(b) struct2cell(b), data, 'UniformOutput', false);
data = cellfun(@(c) c{1}, data, 'UniformOutput', false);

% Separate the material parameters and jv 
deviceNames=unique(cellfun(@(x) extractBefore(x,6),{file_list.name}, 'UniformOutput', false));
deviceCount=numel(deviceNames);

mat_ind = cellfun(@(d) size(d,1)==1, data);
jv_array=data(~mat_ind); mat_array=data(mat_ind); % jv array is nx1 cell, where n is # of devices,
mat_array=reshape(mat_array,length(mat_array)/deviceCount,deviceCount).'; % mat array is n x m, where n (row) are the individual devices and m are the different parameters
num_matParam=size(mat_array,2);

% Check if the 'saved_parameters' folder exists
if ~exist('saved_parameters', 'dir')
    mkdir('saved_parameters');
end

% Neighborhood allocations
if padding == 'y'

    % yes padding

else
    procJVarray = cellfun(@(x) procJV(x,neighborhoodSize,padding), jv_array, 'UniformOutput', false);

    procMatarray = cell(deviceCount,num_matParam);
    splitMaterials = cell(deviceCount,1);

    for i = 1:deviceCount
        splitMaterials{i} = mat_array(i,:);
    end

    for i = 1:deviceCount
        splitMaterials{i} = cellfun(@(x) procMat(x,neighborhoodSize), splitMaterials{i}, 'UniformOutput', false);
    end

    for i = 1:deviceCount
        for j = 1:num_matParam
            procMatarray{i,j} = splitMaterials{i}{j};
        end
    end
end

% Combining all data together

all_jv = cat(2,procJVarray{:}); 
all_mat = cell(1,num_matParam);
for i = 1:num_matParam
    all_mat{i} = cat(2,procMatarray{:,i}); 
end
all_mat = vertcat(all_mat{:});


if smallDataSet == 'y'
    % jv_array=cellfun(@(n) n(:,1:10),jv_array,'UniformOutput',false);
    % mat_array=cellfun(@(n) n(:,1:10),mat_array,'UniformOutput',false);
    all_jv=all_jv(1:10);
    all_mat=all_mat(:,1:10);
end

% Training/validation/testing split
%data_points = sum(cellfun(@(d) size(d,2), procJVarray));
data_points = numel(all_jv);

split_indx = ind_split(data_points,testData_split,1-testData_split);
test_i = split_indx{1}; trainVal_i = split_indx{2};

test_JVset = all_jv(test_i); test_MATset = all_mat(:,test_i);
test.JV=test_JVset; test.MAT=test_MATset;
cd saved_parameters\
save('testSet.m','test')
cd(folderPath)

trainvJVset = all_jv(trainVal_i); trainvMATset = all_mat(:,trainVal_i);

% Set up the training/validation folds

cvIndices = crossvalind('Kfold',length(trainVal_i),numFolds);
trainvJVfolds = cell(numFolds,1); trainvMATfolds = cell(numFolds,1);
setList = unique(cvIndices);

for i = 1:numFolds
    trainvJVfolds{i} = trainvJVset(cvIndices==setList(i));
    trainvMATfolds{i} = trainvMATset(:,cvIndices==setList(i));
end

% Train the network

lr = linspace(2e-3,2e-2,5).';
filtSize = [2,4,8,16].';
hy_permutations = table2array(combinations(lr,filtSize,setList));

kfold_T = zeros(nEpochs,length(hy_permutations));
kfold_V = zeros(nEpochs,length(hy_permutations));

for i = 1:length(hy_permutations)
    [kfold_T(:,i),kfold_V(:,i),~] = smallResNet(hy_permutations(i,:), nEpochs, trainvJVfolds, trainvMATfolds, 'n');
end

[~,best_hy_idx] = min(min(kfold_V,[],2));

trainingPlot = zeros(1,length(hy_permutations));
validationPlot = zeros(1,length(hy_permutations));

[trainingPlot,validationPlot,trainedNets]=smallResNet(hy_permutations(best_hy_idx,:), nEpochs, trainvJVset, trainvMatset, 'y');
[~,best_net_idx] = min(validationPlot);
bestNetwork = trainedNets(best_net_idx);

cd saved_parameters\
save('trainingPlot.m','trainingPlot')
save('validationPlot.m','validationPlot')
save('bestNetwork.m',"bestNetwork")
cd(folderPath)

% Turn off performance measurement tool
if strcmp(profile('status').ProfilerStatus, 'on')
    % checking if user requested profiling >>
    profile off
    p=profile('info');
    save saved_parameters\profileData p
    clear p
end



