% validation.m

%{
Once the network is trained, this program is used to visualize and calculate the results.
%}

clc
clear all
close all

filePathName=matlab.desktop.editor.getActiveFilename;
[folderPath,~,~] = fileparts(filePathName);
cd(folderPath)

addpath(genpath(fileparts(pwd)));

%% Data Selector (change value of "selector" to desired material paramter)
% 1 - deltaV
% 2 - deltaQ
% 3 - dos (n)
% 4 - pl
% 5 - mu

selector=3;
neighborhoodSize=2;
padding='n'; % must be either 'y' or 'n' NOT "Y" or "y"
dataSelect=["N57_7","P61_11"]; % [x, y1, y2, y3, ...] where x is the training, and y is validaiton, need at least one y

% allEncoders=importdata('allEncoders801.mat');
% encoderNet=allEncoders{200};
tempEncoder=load('encoderNet924.mat');
encoderNet=tempEncoder.encoderNet;

% Available device names:
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
% Loading all relevant data into workspace

oldFolder1=pwd;
cd("processedData")
oldFolder2=pwd;
validationJV=cell(1,length(dataSelect)-1);
validationMat=cell(1,length(dataSelect)-1);
for i=1:length(dataSelect)
    if i==1
        cd(dir(dataSelect(i)+"*").name)
        tJV=importdata(dir("*JV*").name);
        tMat=importdata(dir(matstr).name);
        cd(oldFolder2)
        [trainingJV,trainingMat]=neighborhood(tJV,tMat,neighborhoodSize,padding);
    else
        cd(dir(dataSelect(i)+"*").name)
        temp_vJV=importdata(dir("*JV*").name);
        temp_vMat=importdata(dir(matstr).name);
        [validationJV{i-1},validationMat{i-1}]=neighborhood(temp_vJV,temp_vMat,neighborhoodSize,padding,tJV,tMat);
        %[validationJV{i-1},validationMat{i-1}]=neighborhood(temp_vJV,temp_vMat,neighborhoodSize,padding);
        cd(oldFolder2)
    end
end
cd(oldFolder1)

a=zeros(28,9,2,length(validationJV{1}));
ok=validationJV{1};
for k=1:length(validationJV{1})
    a(:,:,:,k)=ok{k};
end
b=dlarray(a,'SSCB');
c=extractdata(forward(encoderNet,b));
d=extractdata(predict(encoderNet,b));

a2=zeros(28,9,2,length(trainingJV));
for k=1:length(trainingJV)
    a2(:,:,:,k)=trainingJV{k};
end
b2=dlarray(a2,'SSCB');
c2=extractdata(forward(encoderNet,b2));
d2=extractdata(predict(encoderNet,b2));

%% MSE error
predictedMat=cell(1,length(dataSelect)-1);
mseValidation=zeros(1,length(dataSelect)-1);

for j=1:length(validationMat)
    currentValidationJV=validationJV{1,j};
    tempMat=zeros(1,length(validationMat{1,j}));
    for i=1:length(tempMat)
        XBatchP=dlarray(double(currentValidationJV{i}),'SSC');
        % XBatchP=b(:,:,:,i);
        % zP=sigmoid(forward(encoderNet,XBatchP));
        zP=forward(encoderNet,XBatchP);
        tempMat(i)=extractdata(zP);
    end
    predictedMat{j}=tempMat;
    mseValidation(j)=(1/length(tempMat)).*sum((validationMat{1,j}-tempMat).^2);
end

% MSE: 1/n summation (y - y_hat)^2
% rmsdValidation=sqrt(mseValidation);

% MAPE
% Mean Absolute Percentage Error
% MAPE = (1/n) \sum{abs((actual-predicted)/actual)}
tempLoss=abs((validationMat{1}-d)./validationMat{1});
sumLoss=sum(tempLoss(isfinite(tempLoss)));
mapeError=sumLoss/sum(isfinite(tempLoss));

% % allMats=cell(1);
% % allMats{1}=trainingMat;
% % allMats=cat(2,allMats,predictedMat);
% % colorsubplot(allMats)

%% Just two plots (for validation)
figure();
subplot(1,2,1)
newColorMap(rescale(predictedMat{1},0.56,0.78))
title('Predicted')

subplot(1,2,2)
newColorMap(validationMat{1})
title('Measured')


% Manual colorplot sharing
combo=cat(2,predictedMat{1},validationMat{1});
minColorLimit=min(combo);
maxColorLimit=max(combo);

step=0.025;
val=validationMat{1};
org=predictedMat{1};

tickMe=0.1:0.2:1.1;

a11=figure();
subplot(1,2,1)
dim=sqrt(length(org));
[XX,YY]=meshgrid(linspace(0,dim*step,dim));
dataMat2=reshape(org,dim,dim);
colormap(turbo)
pcolor(XX,YY,dataMat2);
shading interp
xlim([0 dim*step]); ylim([0 dim*step]);
xticks(tickMe) 
yticks(tickMe)
axis square
c1=colorbar;
caxis([minColorLimit maxColorLimit])
xlabel('Position (mm)'); ylabel('Position (mm)');
set(gca,'FontSize',18)
c1.Visible='off';

subplot(1,2,2)
dim=sqrt(length(val));
dataMat=reshape(val,dim,dim);
colormap(turbo)
pcolor(XX,YY,dataMat);
shading interp
xlim([0 dim*step]); ylim([0 dim*step]);
xticks(tickMe) 
yticks(tickMe)
axis square
c2=colorbar;
caxis([minColorLimit maxColorLimit])
c2.Visible='off';
xlabel('Position (mm)'); ylabel('Position (mm)');
set(gca,'FontSize',18)

% exportgraphics(a11,'pl.png')

function colorsubplot(all_mats)
    numSet=length(all_mats);

    figure();
    subplot(1,numSet,1)
    newColorMap(all_mats{1})

    for k=2:numSet
        subplot(1,numSet,k);
        currentSet=all_mats{k};
        newColorMap(currentSet)
    end
end
 
function newColorMap(dataArray)
    step=0.025;
    dim=sqrt(length(dataArray));
    dataMat=reshape(dataArray,dim,dim);
    [XX,YY]=meshgrid(linspace(0,dim*step,dim));

    colormap(turbo)
    pcolor(XX,YY,dataMat);
    shading interp
    xlim([0 dim*step]); ylim([0 dim*step]);
    axis square
    %tickMark=0:0.2:1.2;
    %xticks(tickMark);
    %yticks(tickMark);
    colorbar
    %h=colorbar;
    %h.Visible='off';
    %h.Label.String=matstr;
    xlabel('Position (mm)'); ylabel('Position (mm)');
    set(gca,'FontSize',18)
end




