
% dataset_splitter.m
% This program splits the data into a training and validation set

function [trainJV,trainMat,validJV,validMat] = dataset_splitter(t_usage,v_usage,neighborSize,padding,varargin)

% if mod(nargin-3,2) ~= 0
%     disp('Error: JV and material set mismatch')
% end

allJV=[];
allMat=[];

for i=1:2:(nargin-4)
    [jv,mat]=ns_neighborhood(varargin{i},varargin{i+1},neighborSize,padding);
    allJV=cat(2,allJV,jv);
    allMat=cat(2,allMat,mat);
end

tst_usage=1-t_usage-v_usage;

if (tst_usage+t_usage+v_usage) ~= 1
    disp('Error: invalid dataset split ')
end

% Splitting dataset into training and validation based on usage percentage
tot_points=length(allMat); ind=1:tot_points; 
temp_randi=shuffle(ind); temp_randi=shuffle(temp_randi,1000);

train_count=floor(t_usage*tot_points); valid_count=floor(v_usage*tot_points);

train_ind=temp_randi(1:train_count);
valid_ind=temp_randi(train_count+1:train_count+1+valid_count);
test_ind=setdiff(ind,cat(2,train_ind,valid_ind));

train_ind=shuffle(train_ind);
valid_ind=shuffle(valid_ind);
test_ind=shuffle(test_ind);

%valid_ind=setdiff(ind,train_ind); % setdiff(A,B) order matters, finds A not in B
%valid_ind=shuffle(valid_ind,1000);

trainJV=allJV(train_ind); trainMat=allMat(train_ind);

[trainJV,trainMat]=standardize(trainJV,trainMat,'y');

if ~isempty(valid_ind)
    validJV=allJV(valid_ind); validMat=allMat(valid_ind);
    [validJV,validMat]=standardize(validJV,validMat,'n');

    testJV=allJV(test_ind); testMat=allMat(test_ind);
    [testJV,testMat]=standardize(testJV,testMat,'n');
    testSet.testJV=testJV;
    testSet.testMat=testMat;

    save("testSet.mat","testSet")
end




