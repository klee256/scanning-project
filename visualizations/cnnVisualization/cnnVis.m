% CNN Layer Visualizations

% If you change the layer(s), you may have to alter the dimensions of some of
% the output images (reshape commands)

clc
clear all
close all

% load 'N57_7cleanJVzoomed'
% load 'N57_7deltaVzoomed'
% [trainingPack,matPack1]=neighborhood(N57_7cleanJVzoomed,N57_7deltaVzoomed,2,'y');
% 
% selectJV=520; 
% singleJVraw=trainingPack{selectJV}; 
% singleJV=cat(3,singleJVraw,zeros(28,9)); singleJV=circshift(singleJV,1,3); 

% t=figure();
% imshow(singleJV,'InitialMagnification',2000)
% exportgraphics(t,'jvImage.eps','BackgroundColor','none')

net=load('encoderResNet9.mat'); 
net=net.encoderNet;

netLayers=net.layerGraph;

appendEndLayers=[
    sigmoidLayer('Name','sig')
    regressionLayer
    ];
netLayers=addLayers(netLayers,appendEndLayers);
netLayers=connectLayers(netLayers,'fc_1','sig');

aNet=assembleNetwork(netLayers);

% Choosing which layer to visualize
layer = [21]; % Use analyzeNetwork to get layer position
% analyzeNetwork(aNet)

% layerDim=net.Layers(layer,1).FilterSize;

% Using deep dream to visualize learned features
cEnd=25; % pick number of filters to visualize w/ perfect square root
channels = 1:cEnd;

for i=1:length(layer)
    
    name = net.Layers(layer(i)).Name;

    % Shallow
    % I = deepDreamImage(aNet,name,channels, ...
    %     'PyramidLevels',1, ...
    %     'Verbose',0);
    % Deep

    I = deepDreamImage(aNet,name,channels, ...
        'PyramidLevels',5, ...
        'NumIterations',30, ...
        'Verbose',1);

    I(:,:,3,:)=zeros(size(I,1),size(I,2),1,length(channels));
    I=circshift(I,1,3);

    h=figure();
    for j = 1:cEnd
        subplot(sqrt(cEnd),sqrt(cEnd),j)
        imshow(I(:,:,:,j))
        title(j,'FontSize',8)
        axis square
    end
    sgtitle("Learned Features, Layer "+layer(i),'FontSize',12)
    fileName1="layer"+int2str(layer(i))+".eps";
    %exportgraphics(h,fileName1,'BackgroundColor','none')
end

% Visualizing activations

% layer = 5; % Use analyzeNetwork to get layer position
% name = net.Layers(layer).Name;
% 
% act1 = activations(aNet,singleJVraw,name);
% sz = size(act1);
% act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
% a = imtile(mat2gray(act1));
% a = reshape(a,[sz(1) sz(2) 1 sz(3)]);
% 
% num=25; % choose number of activations to visualize
% figure();
% for i = 1:num
%     subplot(sqrt(num),sqrt(num),i)
%     imshow(a(:,:,:,i))
%     axis square
% end



