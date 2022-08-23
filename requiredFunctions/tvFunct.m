function [deltaV, outRangeI] = tvFunct(tvDataRaw)

    outRangeI=[];
    for i=1:size(tvDataRaw,2)
        temp1=tvDataRaw(:,i);
        countSameMax=temp1(temp1==max(temp1));
        if length(countSameMax)>50
            outRangeI=cat(2,outRangeI,i);
        end
    end
    
    tv=tvDataRaw-mean(tvDataRaw(2000:2500,:));
    filt=ones(1,35)./35;
    deltaV=[]; 
    for i=1:size(tv,2)
        onetv=tv(:,i);
        temp=filter(filt,1,onetv);
        [~,maxI]=max(temp); 
        acceptedRange=temp(maxI:end);
        deltaV=cat(2,deltaV,max(acceptedRange)-min(acceptedRange));
    end

    



