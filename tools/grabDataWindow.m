function windowArray = grabDataWindow(dataArray,startVal,endVal,isJV)

    if isJV=='y'
        voltage=dataArray(:,1:2:end);
        current=dataArray(:,2:2:end);
        windowArray1=grabDataWindow(voltage,startVal,endVal,'n');
        windowArray2=grabDataWindow(current,startVal,endVal,'n');

        windowArray=zeros(28,size(windowArray1,2)*2);
        windowArray(:,1:2:end)=windowArray1;
        windowArray(:,2:2:end)=windowArray2;
    else
        dim=sqrt(size(dataArray,2));
        wholeMat=reshape(dataArray,size(dataArray,1),dim,dim);
        windowMat=wholeMat(:,startVal:endVal,startVal:endVal);
        windowArray=reshape(windowMat,size(dataArray,1),[]);
    end


    