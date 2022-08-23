clc
clear all
close all

% a=resnet50('Weights','none'); deepNetworkDesigner(a)
% 
% deepNetworkDesigner(encoderLG)

tic
addpath(genpath(fileparts(pwd)));

load 'N57_7cleanJVzoomed'
load 'N57_7deltaVzoomed'
[trainingPack1,matPack1]=neighborhood(N57_7cleanJVzoomed,N57_7deltaVzoomed,2,'y');

load P58_6cleanJVzoomed.mat
load P58_6deltaVzoomed.mat
[trainingPack2,matPack2]=neighborhood(P58_6cleanJVzoomed,P58_6deltaVzoomed,2,'y');

trainingPack=cat(2,trainingPack1,trainingPack2(1:1250));
matPack=cat(2,matPack1,matPack2(1:1250));

XTrain=zeros(28,9,2,length(trainingPack));
for xx=1:length(trainingPack)
    XTrain(:,:,:,xx)=trainingPack{xx};
end
YTrain=matPack.';
dsX = arrayDatastore(XTrain,IterationDimension=4);
dsY = arrayDatastore(YTrain,IterationDimension=1);
dsTrain = combine(dsX,dsY);

miniBatchSize = 1; 
mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    PartialMiniBatch="discard",...
    MiniBatchFcn=@preprocessData,...
    MiniBatchFormat=["SSCB",""]);

% plots = "training-progress";
% if plots == "training-progress"
%     figure
%     lineLossTrain = animatedline(Color=[0.85 0.325 0.098]);
%     ylim([0 inf])
%     xlabel("Iteration")
%     ylabel("Loss")
%     grid on
% end

%% 28x9x2
encoderLG = layerGraph([
    imageInputLayer([28 9 2],'Name','input1','Normalization','none')
    convolution2dLayer([1 2],64,'Stride',1)
    batchNormalizationLayer
    reluLayer('Name','initial_relu')
    ]);

numberBlocks=[1,1,2]; % not including the initial block w/ off branch conv
startFilts=[64,128,256];
strideL=1;
stageList=1:1:length(startFilts); 
prevMainBranchName='initial_relu';
for xx=1:length(startFilts)
    if xx>1
        strideL=2;
    end
    for yy=1:numberBlocks(xx)+1
        tempLayers=[
        convolution2dLayer(1,startFilts(xx),'Stride',strideL,'Name',"cn"+int2str(xx)+int2str(yy))
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(2,startFilts(xx),'Stride',1,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(1,startFilts(xx)*4,'Stride',1)
        batchNormalizationLayer
        additionLayer(2,'Name',"add"+int2str(xx)+int2str(yy))
        reluLayer('Name',"relu"+int2str(xx)+int2str(yy))
        ];
        encoderLG=addLayers(encoderLG,tempLayers);
        encoderLG=connectLayers(encoderLG,prevMainBranchName,"cn"+int2str(xx)+int2str(yy));
        prevMainBranchName="relu"+int2str(xx)+int2str(yy);
        strideL=1;
    end
end

prevMainBranchName='initial_relu';
nextMainBranchName='add11';
strideL=1;
for xx=1:length(startFilts)
    if xx>1
        strideL=2;
    end
    tempLayers=[
        convolution2dLayer(1,startFilts(xx)*4,'Stride',strideL,'Name',"cn_b"+int2str(xx),'Padding','same')
        batchNormalizationLayer('Name',"bn"+int2str(xx))
        ];
    encoderLG=addLayers(encoderLG,tempLayers);
    encoderLG=connectLayers(encoderLG,prevMainBranchName,"cn_b"+int2str(xx));
    encoderLG=connectLayers(encoderLG,"bn"+int2str(xx),nextMainBranchName+"/in2");
    prevMainBranchName="relu"+int2str(xx)+int2str(numberBlocks(xx)+1);
    nextMainBranchName="add"+int2str(xx+1)+int2str(1);
end

for xx=1:length(startFilts)
    for yy=1:numberBlocks(xx)
        encoderLG=connectLayers(encoderLG,"relu"+int2str(xx)+int2str(yy),"add"+int2str(xx)+int2str(yy+1)+"/in2");
    end
end

encoderLG=addLayers(encoderLG,globalAveragePooling2dLayer('Name','avg_p'));
encoderLG=addLayers(encoderLG,fullyConnectedLayer(1,'Name','fc_1'));
encoderLG=connectLayers(encoderLG,"relu"+int2str(length(numberBlocks))+int2str(numberBlocks(end)+1),'avg_p');
encoderLG=connectLayers(encoderLG,'avg_p','fc_1');
encoderNet = dlnetwork(encoderLG);


%%

numEpochs=100; lr=1.0e-4;
avgGradientsEncoder=[]; avgGradientsSquaredEncoder=[];
iteration=0;
for epoch = 1:numEpochs
    shuffle(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        [X,Y] = next(mbq); 
        [loss,infGrad] = dlfeval(@modelGradients,encoderNet,X,Y);
        [encoderNet.Learnables,avgGradientsEncoder,avgGradientsSquaredEncoder] = adamupdate(encoderNet.Learnables,infGrad,avgGradientsEncoder,avgGradientsSquaredEncoder,iteration,lr);
        %[encoderNet,averageGrad,averageSqGrad] = adamupdate(encoderNet,infGrad,avgGradientsEncoder,avgGradientsSquaredEncoder,iteration);

%         if plots == "training-progress"
%             addpoints(lineLossTrain,iteration,double(loss))
%             title("Epoch: " + epoch)
%             drawnow
%         end
    end

    %disp(extractdata(loss));
    %disp(epoch);

end

predictedMat=[];
for i=1:length(matPack1)
    XBatchP=dlarray(double(trainingPack{i}),'SSC');
    zP=sigmoid(forward(encoderNet,XBatchP));
    predictedMat=cat(2,predictedMat,extractdata(zP));
end

% makeContourPlot(predictedMat,'','pred',0.025,'n')
% makeContourPlot(matPack1,'','real',0.025,'n')

Hold=abs((predictedMat-matPack1)./matPack1);
Hold=Hold(Hold~=Inf);
avgPerError=sum(Hold)./length(Hold);

disp(avgPerError)
save('predictedMat.mat','predictedMat')
save('encoder.mat','encoderNet')

toc

function [X,T1] = preprocessData(dataX,dataY)

% Extract image data from cell and concatenate
X = cat(4,dataX{:});
T1 = cat(2,dataY{:});

end


