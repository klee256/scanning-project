% resnetFunct.m
%{
    This network uses 28x9x2 inputs (28x18x1 performs worse)
    trainingJV and validationJV need to be processed by neighborhood.m 
    Function arguments (6 max): learning rate, total number of epochs, training JV set,
    training material parameter set, training JV validation set (optional),
    training material parameter validation set (optional).
    Typical hyperparameters: lr=1.0e-4, numEpoch=200; 
%}

% For 100 training and 100 validation, 6849s, 20 epochs, lr=1.0e-4

function [trainingLoss,validationLoss,holdEncoders] = smallResNet_noval(hyparm,numEpochs,trainingJV,trainingMat,saveParams)

% plots = "training-progress";
% plots = "off";

lr = hyparm(1);
startFiltSize = hyparm(2);


% trainingJV = 

XTrain=zeros(28,9,2,length(trainingJV));
for xx=1:length(trainingJV)
    XTrain(:,:,:,xx)=trainingJV{xx};
end
YTrain=trainingMat.';




dsX = arrayDatastore(XTrain,IterationDimension=4);
dsY = arrayDatastore(YTrain,IterationDimension=1);
dsTrain = combine(dsX,dsY);

onePush_tJV=zeros(28,9,2,length(trainingJV));
for k=1:length(trainingJV)
    onePush_tJV(:,:,:,k)=trainingJV{k};
end
onePush_tJV=dlarray(onePush_tJV,'SSCB');

onePush_vJV=zeros(28,9,2,length(validationJV));
for k=1:length(validationJV)
    onePush_vJV(:,:,:,k)=validationJV{k};
end
onePush_vJV=dlarray(onePush_vJV,'SSCB');

miniBatchSize = 1;
mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    PartialMiniBatch="discard",...
    MiniBatchFcn=@preprocessData,...
    MiniBatchFormat=["SSCB",""]);

% if plots == "training-progress"
%     figure
%     lineLossTrain = animatedline(Color=[0 0 1]);
%     lineLossValid = animatedline(Color=[1 0 0]);
%     ylim([0 inf])
%     xlabel("Iteration")
%     ylabel("Loss")
%     grid on
% end

%% Building network graph
encoderLG = layerGraph([
    imageInputLayer([28 9 2],'Name','input1','Normalization','none')
    convolution2dLayer([1 2],64,'Stride',1)
    batchNormalizationLayer
    reluLayer('Name','initial_relu')
    ]);
numberBlocks=[1,1]; % not including the initial block w/ off branch conv
startFilts=[8,16]; strideL=1;
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
            reluLayer('Name',"relu"+int2str(xx)+int2str(yy))];
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
encoderLG=addLayers(encoderLG,fullyConnectedLayer(4,'Name','fc_1'));
encoderLG=connectLayers(encoderLG,"relu"+int2str(length(numberBlocks))+int2str(numberBlocks(end)+1),'avg_p');
encoderLG=connectLayers(encoderLG,'avg_p','fc_1');
encoderNet = dlnetwork(encoderLG);


%% Network training
trainingLoss=zeros(1,numEpochs);
validationLoss=zeros(1,numEpochs);
holdEncoders=cell(1,numEpochs);

avgGradientsEncoder=[]; avgGradientsSquaredEncoder=[];
iteration=0;
for epoch = 1:numEpochs
    lr=(0.5)*initial_lr*(1+cos((epoch*pi)/numEpochs));
    shuffle(mbq);
    while hasdata(mbq)
        iteration = iteration + 1;
        [X,Y] = next(mbq);
        [~,state,infGrad] = dlfeval(@modelGradients,encoderNet,X,Y);
        encoderNet.State=state;
        [encoderNet,avgGradientsEncoder,avgGradientsSquaredEncoder] = adamupdate(encoderNet,infGrad,avgGradientsEncoder,avgGradientsSquaredEncoder,iteration,lr);

        %         if plots == "training-progress"
        %             if nargin > 4
        %                 tempValLoss=forwardLoss(encoderNet,validationJV,validationMat);
        %                 addpoints(lineLossValid,iteration,double(tempValLoss))
        %             end
        %             tempTrainLoss=forwardLoss(encoderNet,trainingJV,trainingMat);
        %             addpoints(lineLossTrain,iteration,double(tempTrainLoss))
        %             title("Epoch: " + epoch)
        %             legend('Training','Validation')
        %             drawnow
        %         end

        %         disp("Iteration: "+iteration);
        %         disp(extractdata(loss));
    end

    holdEncoders{epoch}=encoderNet;
    trainingLoss(epoch)=forwardLoss(encoderNet,onePush_tJV,trainingMat);
    if nargin > 4
        validationLoss(epoch)=forwardLoss(encoderNet,onePush_vJV,validationMat);
    end
end

save('trainingLoss.mat','trainingLoss')
save('allEncoders.mat','holdEncoders')
if nargin > 4
    save('validationLoss.mat','validationLoss')
end

end

%% Necessary functions
function [X,T1] = preprocessData(dataX,dataY)
% Extract "image" data from cell and concatenate
X = cat(4,dataX{:});
T1 = cat(2,dataY{:});
end

function [loss,state,infGrad] = modelGradients(encoderNet,x,mat)
[z,state]=forward(encoderNet,x);
%z=sigmoid(z); % Should declare sigmoid explicitly in layerGraph
loss=(mat-z).^2;
infGrad=dlgradient(loss,encoderNet.Learnables);
end

function currentError = forwardLoss(inputNet,jv,material_params)

tempPred=extractdata(predict(inputNet,jv));

% Mean Absolute Percentage Error
% MAPE = (1/n) \sum{abs((actual-predicted)/actual)}
tempLoss=abs((material_params-tempPred)./material_params);
sumLoss=sum(tempLoss(isfinite(tempLoss)));
currentError=sumLoss/sum(isfinite(tempLoss));

%     % Function performs forward pass and calculates MSE for all points in jv
%     tempLoss=0; lossCount=0;
%     for ii=1:length(material_params)
%         XBatch=dlarray(jv{ii},'SSC');
%         zv=forward(inputNet,XBatch,Outputs=["fc_1"]);
%         %zv=sigmoid(zv);
%         %singleLoss=(material_params(ii)-zv).^2; % used to be for MSE
%
%         % Mean Absolute Percentage Error
%         % MAPE = (1/n) \sum{abs((actual-predicted)/actual)}
%         singleLoss=abs((material_params(ii)-zv)./material_params(ii));
%         if isfinite(singleLoss)
%             tempLoss=tempLoss+singleLoss;
% 	else
% 	    lossCount=lossCount+1;
%         end
%     end
%     currentLoss=(1/(length(material_params)-lossCount))*tempLoss; % avg MAPE
%     currentLoss=extractdata(currentLoss);
end


