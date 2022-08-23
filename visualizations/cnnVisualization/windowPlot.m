function windowArray = windowPlot(dataArray,startVal,endVal)
    % square only
    % must be size [1 n]
    % first number in XX or YY determines Y

    dim=sqrt(length(dataArray));
    wholeMat=reshape(dataArray,dim,dim);
    windowMat=wholeMat(startVal:endVal,startVal:endVal);
    windowArray=reshape(windowMat,1,[]);
    %makeContourPlot(windowArray,'','windowed',0.025,'n')