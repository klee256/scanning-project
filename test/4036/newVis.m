clc
clear all
close all

% Manually Change
jv1=importdata('N57_7cleanJVzoomed.mat');
mat1=importdata('N57_7deltaVzoomed.mat');

tempEncoder=load('encoderNet924.mat');
encoderNet=tempEncoder.encoderNet;

neighborSize=2; padding='n';

[jv,mat]=ns_neighborhood(jv1,mat1,neighborSize,padding);

[procJV,procMat]=standardize(jv,mat,'n');

a=zeros(28,9,2,length(procJV));
ok=validationJV{1};
for k=1:length(validationJV{1})
    a(:,:,:,k)=ok{k};
end
b=dlarray(a,'SSCB');
c=extractdata(forward(encoderNet,b));

tempPred=extractdata(predict(encoderNet,b));

% Mean Absolute Percentage Error
% MAPE = (1/n) \sum{abs((actual-predicted)/actual)}
tempLoss=abs((mat1-tempPred)./mat1);
sumLoss=sum(tempLoss(isfinite(tempLoss)));
currentError=sumLoss/sum(isfinite(tempLoss));


