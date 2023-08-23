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

function [trainingLoss,validationLoss,holdEncoders] = smallResNet(hyparm,numEpochs,jv,mat,saveParam)

% plots = "training-progress";
% plots = "off";
oldFolder = pwd;

initial_lr = hyparm(1);
startFiltSize = hyparm(2);
valFold = hyparm(3);

validationJV = jv{valFold};
validationMat = mat{valFold};

list=1:numel(jv);
trainingJV=jv(list(list~=valFold)); trainingJV=cat(2,trainingJV{:});
trainingMat=mat(list(list~=valFold)); trainingMat=cat(2,trainingMat{:});

save_means = mean(trainingMat,2);
save_stds = std(trainingMat,1,2);
if saveParam == 'y'
    matGauss.mean=save_means;
    matGauss.std=save_stds;
    cd saved_parameters\
    save("matGauss_param.mat","matGauss")
    cd(oldFolder)
end

trainingMat=(trainingMat-save_means)./save_stds;

XTrain=zeros(28,9,2,length(trainingJV));
for xx=1:length(trainingJV)
    XTrain(:,:,:,xx)=trainingJV{xx};
end

YTrain1=trainingMat(1,:).';
YTrain2=trainingMat(2,:).';
YTrain3=trainingMat(3,:).';
YTrain4=trainingMat(4,:).';

dsX = arrayDatastore(XTrain,IterationDimension=4);
dsY1 = arrayDatastore(YTrain1,IterationDimension=1);
dsY2 = arrayDatastore(YTrain2,IterationDimension=1);
dsY3 = arrayDatastore(YTrain3,IterationDimension=1);
dsY4 = arrayDatastore(YTrain4,IterationDimension=1);
dsTrain = combine(dsX,dsY1,dsY2,dsY3,dsY4);

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
    MiniBatchFormat=["SSCB",'','','','']);

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
numberBlocks=0; % not including the initial block w/ off branch conv
startFilts=startFiltSize; strideL=1;
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
        [X,Y1,Y2,Y3,Y4] = next(mbq);
        [~,state,infGrad] = dlfeval(@modelGradients,encoderNet,X,Y1,Y2,Y3,Y4);
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

    if saveParam == 'y'
        holdEncoders{epoch}=encoderNet;
    end
    trainingLoss(epoch)=forwardLoss(encoderNet,onePush_tJV,trainingMat);
    validationLoss(epoch)=forwardLoss(encoderNet,onePush_vJV,validationMat);
 
end

cd saved_parameters\
save('trainingLoss.mat','trainingLoss')
save('validationLoss.mat','validationLoss')
if saveParam == 'y'
    save('allEncoders.mat','holdEncoders')
end
cd(oldFolder)

end

%% Necessary functions

function [X,T1,T2,T3,T4] = preprocessData(dataX,dataY1,dataY2,dataY3,dataY4)
    X = cat(4,dataX{:});
    T1 = cat(2,dataY1{:});
    T2 = cat(2,dataY2{:});
    T3 = cat(2,dataY3{:});
    T4 = cat(2,dataY4{:});
end

function [loss,state,infGrad] = modelGradients(encoderNet,x,mat1,mat2,mat3,mat4)
    [z,state]=forward(encoderNet,x);
    %z=sigmoid(z); % Should declare sigmoid explicitly in layerGraph
    mat=[mat1;mat2;mat3;mat4];
    loss=sum((mat-z).^2);
    infGrad=dlgradient(loss,encoderNet.Learnables);
end

function currentError = forwardLoss(inputNet,jv,material_params)

tempPred=extractdata(forward(inputNet,jv));

% Mean Absolute Percentage Error
% MAPE = (1/n) \sum{abs((actual-predicted)/actual)}

temp1 = abs((material_params-tempPred)./material_params);
currentError = sum(temp1(isfinite(temp1)))./numel(temp1(isfinite(temp1)));

end


