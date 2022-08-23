function deltaQ = tcFunct(tcDataRaw, timDiv)
    
    % timeDiv is an array
    resist=490;

    current=tcDataRaw-mean(tcDataRaw(2001:2500,:));
    current=current./resist;
    tc=medfilt1(current,100); filt=ones(1,100)./100; 
    tc=filter(filt,1,tc);

    intensities=length(timDiv);
    division=size(tcDataRaw,2)/intensities;

    deltaQ=[];
    for s=1:intensities
        time=linspace(0,10*timDiv(s),2500); time=time.';
        for i=1:division
            j=i+division*(s-1);
            [~,maxI]=max(tc(:,j));
            integral=trapz(time(maxI:end),tc(maxI:end,j));
            deltaQ=cat(2,deltaQ,integral);
        end
    end
    



