% neighborhood.m
% Grabs 8 points surrounding a point and concats the JV curves (9 points total)
% Distance away from points is n
% Also grabs the related material parameters

function [trainJVArray,trainMatArray] = neighborhood(jvData,materialArray,n,padding)

dim=sqrt(size(jvData,2)/2);

voltage=jvData(:,1:2:end); voltage=rescale(voltage);
current=jvData(:,2:2:end); current=current*(-1.0); current=rescale(current);
 
jvArray=cell(1,size(voltage,2));
for jj=1:length(jvArray)
    jvArray{jj}=cat(2,voltage(:,jj),current(:,jj));
end

if padding == 'n'
    fullJVMatrix=reshape(jvArray,dim,dim);
    startingPos=n+1;
    endingPos=dim-n;
    trainMatArray=windowPlot(materialArray,startingPos,endingPos);
    trainMatArray=rescale(trainMatArray);
end

if padding == 'y'   
    baseArray=cell(1,(dim+n*2)^2);
    baseArray(:,:)={zeros(28,2)};
    baseMatrix=reshape(baseArray,(dim+n*2),(dim+n*2));
    startingPos=n+1;
    baseMatrix(startingPos:dim+n,startingPos:dim+n)=reshape(jvArray,dim,dim);
    fullJVMatrix=baseMatrix;
    endingPos=dim+n;
    trainMatArray=materialArray;
    trainMatArray=rescale(trainMatArray);
end

spacing=[(-1.0)*n,0,n];

%% 28x9x2
trainJVArray=cell(1,(endingPos-startingPos+1)*(endingPos-startingPos+1));
iteration=1;
for v=startingPos:endingPos
    for w=startingPos:endingPos
        tempV=[]; tempC=[];
        for i=1:length(spacing)
            for j=1:length(spacing)
                tempV=cat(2,tempV,fullJVMatrix{w+spacing(i),v+spacing(j)}(:,1));
                tempC=cat(2,tempC,fullJVMatrix{w+spacing(i),v+spacing(j)}(:,2));
            end
        end
        trainJVArray{iteration}=cat(3,tempV,tempC);
        iteration=iteration+1;
    end
end

% jsc=[];
% for i=1:3600
%     jsc=cat(2,jsc,trainJVArray{i}(1,5,2));
% end
% makeContourPlot(jsc,'','',0.025,'y')

%% 28x18x1
% trainJVArray=cell(1,(endingPos-startingPos+1)*(endingPos-startingPos+1));
% iteration=1;
% for v=startingPos:endingPos
%     for w=startingPos:endingPos
%         temp=[];
%         for i=1:length(spacing)
%             for j=1:length(spacing)
%                 temp=cat(2,temp,fullJVMatrix{w+spacing(i),v+spacing(j)});
%             end
%         end
%         trainJVArray{iteration}=temp;
%         iteration=iteration+1;
%     end
% end





