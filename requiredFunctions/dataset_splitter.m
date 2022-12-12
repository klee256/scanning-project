
% dataset_splitter.m
% This program splits the data into a training and validation set

function [trainJV,trainMat,validJV,validMat] = dataset_splitter(usage,neighborSize,padding,varargin)

if mod(nargin-3,2) ~= 0
    disp('Error: JV and material set mismatch')
end

allJV=[];
allMat=[];

for i=1:2:(nargin-3)
    [jv,mat]=ns_neighborhood(varargin{i},varargin{i+1},neighborSize,padding);
    allJV=cat(2,allJV,jv);
    allMat=cat(2,allMat,mat);
end

% if nargin ~= 5
% Splitting dataset into training and validation based on usage percentage
ind=1:length(allMat); 
temp_randi=shuffle(ind); temp_randi=shuffle(temp_randi,1000);

train_ind=temp_randi(1:floor(usage*length(allMat)));
valid_ind=setdiff(ind,train_ind); % setdiff(A,B) order matters, finds A not in B
valid_ind=shuffle(valid_ind,1000);

trainJV=allJV(train_ind); trainMat=allMat(train_ind);

[trainJV,trainMat]=standardize(trainJV,trainMat);

if ~isempty(valid_ind)
    validJV=allJV(valid_ind); validMat=allMat(valid_ind);
    [validJV,validMat]=standardize(validJV,validMat);
end




