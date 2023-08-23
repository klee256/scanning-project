% smallResNet.m
%{
    This network uses 28x9x2 inputs (28x18x1 performs worse)
    trainingJV and validationJV need to be processed by neighborhood.m 
    Function arguments (6 max): learning rate, total number of epochs, training JV set,
    training material parameter set, training JV validation set (optional),
    training material parameter validation set (optional).
    Typical hyperparameters: lr=1.0e-4, numEpoch=200; 
%}

% For 100 training and 100 validation, 6849s, 20 epochs, lr=1.0e-4

function [trainingLoss,validationLoss,holdEncoders] = smallResNet_nv(hyparm,numEpochs,jv,mat,saveParam)

% plots = "training-progress";
% plots = "off";
oldFolder = pwd;

%initial_lr = hyparm(1);
lr = hyparm(1);
startFiltSize = hyparm(2);

trainingJV=jv; 
trainingMat=mat; 

save_means = mean(trainingMat,2);
save_stds = std(trainingMat,1,2);
if saveParam == 'y'
    matGauss.mean=save_means;
    matGauss.std=save_stds;
    cd saved_parameters
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

% onePush_vJV=zeros(28,9,2,length(validationJV));
% for k=1:length(validationJV)
%     onePush_vJV(:,:,:,k)=validationJV{k};
% end
% onePush_vJV=dlarray(onePush_vJV,'SSCB');

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
    convolution2dLayer([2 3],startFiltSize,'Stride',1,'Name','cn1')
    leakyReluLayer('Name','relu1')
    convolution2dLayer([3 3],startFiltSize*2,'Stride',1,'Name','cn2','Padding','same')
    leakyReluLayer('Name','relu2')
    convolution2dLayer([3 3],startFiltSize*4,'Stride',1,'Name','cn3','Padding','same')
    leakyReluLayer('Name','relu3')
    %dropoutLayer(0.2)
    fullyConnectedLayer(1,'Name','fc1')]);
   
encoderLG = addLayers(encoderLG,fullyConnectedLayer(1,'Name','fc2'));
encoderLG = connectLayers(encoderLG,'relu3','fc2');

encoderLG = addLayers(encoderLG,fullyConnectedLayer(1,'Name','fc3'));
encoderLG = connectLayers(encoderLG,'relu3','fc3');

encoderLG = addLayers(encoderLG,fullyConnectedLayer(1,'Name','fc4'));
encoderLG = connectLayers(encoderLG,'relu3','fc4');

encoderNet = dlnetwork(encoderLG);


%% Network training
trainingLoss=zeros(1,numEpochs);
validationLoss=zeros(1,numEpochs);
holdEncoders=cell(1,numEpochs);

avgGradientsEncoder=[]; avgGradientsSquaredEncoder=[];
iteration=0;
for epoch = 1:numEpochs
    %lr=(0.5)*initial_lr*(1+cos((epoch*pi)/numEpochs));
    shuffle(mbq);

    while hasdata(mbq)

        iteration = iteration + 1;

        [X,Y1,Y2,Y3,Y4] = next(mbq);

        [loss,state,infGrad] = dlfeval(@modelGradients,encoderNet,X,Y1,Y2,Y3,Y4);
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
    % validationLoss(epoch)=forwardLoss(encoderNet,onePush_vJV,validationMat);
 
end

cd saved_parameters
save('trainingLoss.mat','trainingLoss')
% save('validationLoss.mat','validationLoss')
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

function [loss,state,infGrad] = modelGradients(inNet,x,mat1,mat2,mat3,mat4)
    
    [z1,z2,z3,z4,state]=forward(inNet,x,'Outputs',["fc1" "fc2" "fc3" "fc4"]);

    %[z,state]=forward(encoderNet,x);
    %z=sigmoid(z); % Should declare sigmoid explicitly in layerGraph
    %mat=[mat1;mat2;mat3;mat4];
    %loss=sum((z-mat).^2); loss=loss/4;
    loss1 = (z1-mat1).^2;
    loss2 = (z2-mat2).^2;
    loss3 = (z3-mat3).^2;
    loss4 = (z4-mat4).^2;
    loss = (loss1 + loss2 + loss3 + loss4)/4;

    infGrad=dlgradient(loss,inNet.Learnables);
end

function currentError = forwardLoss(inputNet,jv,material_params)

%tempPred=extractdata(forward(inputNet,jv));
[tempPred1,tempPred2,tempPred3,tempPred4,~] = forward(inputNet,jv,'Outputs',["fc1" "fc2" "fc3" "fc4"]);

% Mean Absolute Percentage Error
% MAPE = (1/n) \sum{abs((actual-predicted)/actual)}

temp11 = abs(((material_params(1,:)-tempPred1)./material_params));
temp12 = sum(temp11(isfinite(temp11)))./numel(temp11(isfinite(temp11)));

temp21 = abs(((material_params(2,:)-tempPred2)./material_params));
temp22 = sum(temp21(isfinite(temp21)))./numel(temp21(isfinite(temp21)));

temp31 = abs(((material_params(3,:)-tempPred3)./material_params));
temp32 = sum(temp31(isfinite(temp31)))./numel(temp31(isfinite(temp31)));

temp41 = abs(((material_params(4,:)-tempPred4)./material_params));
temp42 = sum(temp41(isfinite(temp41)))./numel(temp41(isfinite(temp41)));

currentError = (temp12 + temp22 + temp32 + temp42)/4;

%temp1 = abs(((material_params-tempPred)./material_params));
%currentError = sum(temp1(isfinite(temp1)))./numel(temp1(isfinite(temp1)));

end


