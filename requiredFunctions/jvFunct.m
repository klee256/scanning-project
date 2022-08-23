function [voc, jsc, cleanJV] = jvFunct(jvDataRaw)
    
    % output jsc has units of mA/cm^2

    voltage=jvDataRaw(:,1:2:end);
    current=jvDataRaw(:,2:2:end);

    % intensities=(size(jvDataRaw,2)/2)/(dim*dim);

    voc=zeros(1,length(voltage)); jsc=zeros(1,length(voltage));
    cleanJV=[];

    x=linspace(0,0.65,1000);
    for i=1:length(voltage)
        % exponential fit funct: a*exp(b*x) + c*exp(d*x)
        % works even when JV is bad (when it looks linear)
        f=fit(voltage(:,i),current(:,i),'exp2');
        y=f.a*exp(f.b.*x) + f.c*exp(f.d.*x);

        % obtaining voc and jsc
        [~,im]=min(abs(y));
        voc(i)=x(im);
        jsc(i)=current(1,i);
        
        cleanV=linspace(0,x(im),28); cleanV=cleanV.';
        cleanJ=f.a*exp(f.b.*cleanV) + f.c*exp(f.d.*cleanV);
        cleanJV=cat(2,cleanJV,cleanV);
        cleanJV=cat(2,cleanJV,cleanJ);
    end

    A=pi*((250)/2)^2;
    jsc=jsc*10^3; jsc=jsc./A; jsc=jsc.*(10000^2);



    