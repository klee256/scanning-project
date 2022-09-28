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

function [trainingLoss,validationLoss,holdEncoders] = smallNetFunct(initial_lr,numEpochs,trainingJV,trainingMat,varargin)

% plots = "training-progress";
plots = "off";

if nargin > 4
    validationJV=varargin{1,1};
    validationMat=varargin{1,2};
    %val_length=length(validationMat);
end

XTrain=zeros(28,9,2,length(trainingJV));
for xx=1:length(trainingJV)
    XTrain(:,:,:,xx)=trainingJV{xx};
end
YTrain=trainingMat.';
dsX = arrayDatastore(XTrain,IterationDimension=4);
dsY = arrayDatastore(YTrain,IterationDimension=1);
dsTrain = combine(dsX,dsY);

miniBatchSize = 1;
mbq = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    PartialMiniBatch="discard",...
    MiniBatchFcn=@preprocessData,...
    MiniBatchFormat=["SSCB",""]);

if plots == "training-progress"
    figure
    lineLossTrain = animatedline(Color=[0 0 1]);
    lineLossValid = animatedline(Color=[1 0 0]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

%% Building network graph
encoderLG = layerGraph([
    imageInputLayer([28 9 2],'Name','input1','Normalization','none')
    convolution2dLayer([2 3],64,'Stride',1,'Name','cn1')
    leakyReluLayer
    convolution2dLayer([3 3],128,'Stride',2,'Name','cn2')
    leakyReluLayer
    convolution2dLayer([3 3],256,'Stride',2,'Name','cn3')
    leakyReluLayer
    %convolution2dLayer([6 1],1,'Stride',1,'Name','cn4')
    fullyConnectedLayer(1,'Name','fc_1');
    ]);
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

    tempTrainLoss=forwardLoss(encoderNet,trainingJV,trainingMat);
    trainingLoss(epoch)=tempTrainLoss;

    if nargin > 4
        tempValLoss=forwardLoss(encoderNet,validationJV,validationMat);
        validationLoss(epoch)=tempValLoss;
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
    [z,state]=forward(encoderNet,x,Outputs=["fc_1"]);
    z=sigmoid(z);
    loss=mse(z,mat);
    infGrad=dlgradient(loss,encoderNet.Learnables);
end

function currentLoss = forwardLoss(inputNet,jv,material_params)
    % Function performs forward pass and calculates MSE for all points in jv
    tempLoss=0;
    for ii=1:length(material_params)
        XBatch=dlarray(jv{ii},'SSC');
        zv=forward(inputNet,XBatch,Outputs=["fc_1"]);
        zv=sigmoid(zv);
        singleLoss=mse(zv,material_params(ii));
        tempLoss=tempLoss+singleLoss;
    end
    currentLoss=(1/length(material_params))*tempLoss; % avg mse
    currentLoss=extractdata(currentLoss);
end


