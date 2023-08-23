% procJV.m

%{
    load_param must be either 'y' or 'n', 
    padding must be etiher 'y' or 'n',
    in_jv is assumed to be a [1x1] cell
    n is neighborhoodSize
%}

function out_jv = procJV(in_jv,n,padding)

dim=sqrt(size(in_jv,2)/2);

voltage=in_jv(:,1:2:end); current=in_jv(:,2:2:end); current=current.*10^6;

jvArray=mat2cell([voltage(:),current(:)],28*ones(1,size(voltage,2)), 2).';

if padding == 'n'
    fullJVMatrix=reshape(jvArray,dim,dim);
    startingPos=n+1;
    endingPos=dim-n;
end

if padding == 'y'   
    baseArray=cell(1,(dim+n*2)^2);
    baseArray(:,:)={zeros(28,2)};
    baseMatrix=reshape(baseArray,(dim+n*2),(dim+n*2));
    startingPos=n+1;
    baseMatrix(startingPos:dim+n,startingPos:dim+n)=reshape(jvArray,dim,dim);
    fullJVMatrix=baseMatrix;
    endingPos=dim+n;
end

spacing=[(-1.0)*n,0,n];

% 28x9x2
out_jv=cell(1,(endingPos-startingPos+1)*(endingPos-startingPos+1));
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
        out_jv{iteration}=cat(3,tempV,tempC);
        iteration=iteration+1;
    end
end


