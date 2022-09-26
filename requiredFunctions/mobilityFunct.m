
% tS = time scale on o-scope (tS=[2.5e-6,2.5e-6,1.0e-6,2.5e-6];)
% ab = applied bias (ab=[0,0.1,0.2,0.3];)
% d = film thickness
% ab = applied bias (100 mV, 200mV, etc) in array [0,1,2,3]

function mu = mobilityFunct(tS,ab,varargin)

if length(varargin) ~= length(tS)
    disp('Error: timescale mismatch')
end

d = 400e-9; % film thickness 
d2=(d*100)^2; % square film thickness (cm)

abandon=300; % Need to drop the first 300 values when using the median filter (will decay near edge)
y_axis=zeros(length(varargin),size(varargin{1,1},2));
for jj = 1:length(varargin)
    tcData=varargin{1,jj}; tc=tcData-mean(tcData(2300:2500,:));
    tc=medfilt1(tc,125); 
    time=linspace(0,10*tS(jj),2500); time=time-mean(time); time=time.';
    t=time(abandon:end); y=tc(abandon:end,:);
    tempMax=max(y)*0.9;
    tempMin=max(y)*0.1;
    [~,topPos]=min(abs(y-tempMax));
    [~,botPos]=min(abs(y-tempMin));
    tTop=t(topPos);
    tBot=t(botPos);
    decayTimes=tBot-tTop;
    y_axis(jj,1:size(varargin{1,1},2))=d2./decayTimes;
end

mu=zeros(1,size(varargin{1,1},2));
for i=1:size(varargin{1,1},2)
    temp=y_axis(:,i).';
    slope=polyfit(ab,temp,1);
    mu(i)=slope(1);
end

badPos=find(mu>0);
for kk=1:length(badPos)
    temp=y_axis(:,badPos(kk)).';
    for ll=1:length(varargin)
        slope=polyfit(ab(setdiff(1:end,ll)),temp(setdiff(1:end,ll)),1);
        tempMu=slope(1);
        if tempMu < 0
            mu(badPos(kk))=tempMu;
            break;
        end
    end
end

permSet=1:1:length(varargin);
permutations=nchoosek(permSet,2);

badPos=find(mu>0);
for kk=1:length(badPos)
    temp=y_axis(:,badPos(kk)).';
    for ll=1:length(permutations)
        slope=polyfit(ab(permutations(ll,:)),temp(permutations(ll,:)),1);
        tempMu=slope(1);
        if tempMu < 0
            mu(badPos(kk))=tempMu;
            break;
        end
    end
end

badPos=find(mu>0);
if ~isempty(badPos)
    disp('Error: Unable to calculate all positive mobilities.\n')
    disp("Total issues: "+length(badPos))
end

%% Calculating mobility via sigmoid fit (sigmoid(-x))
% Does not work. Mobility values will average out (across multiple biases for the same point).
% Might be due to the decay not actually being sigmoid. The fit for the
% tail end is poor.

% abandon=300;
% y_axis=zeros(length(varargin),size(varargin{1,1},2));
% for jj = 1:length(varargin)
%     tcData=varargin{1,jj}; tc=tcData-mean(tcData(2300:2500,:));
%     time=linspace(0,10*tS(jj),2500);
%     tcMF=medfilt1(tc,125); 
%     ft = fittype('a/(b+exp(c*x+d))','independent',{'x'},'coefficients',{'a','b','c','d'});
%     decayTimes=zeros(1,size(varargin{1,1},2));
%     for ii=1:size(varargin{1,1},2)
%         [~,meanPos]=min(abs(tcMF(:,ii)-mean(tcMF(:,ii))));
%         tempTime=time-time(meanPos); tempTime=tempTime.';
%         fv=fit(tempTime,tcMF(:,ii),ft,'StartPoint',[0.0113,1.0,2.128e+06,0]);
%         tempTC=fv.a./(fv.b+exp(fv.c*tempTime+fv.d));
%         maxRange=max(tempTC)*0.9;
%         minRange=max(tempTC)*0.1;
%         [~,topRPos]=min(abs(tempTC-maxRange));
%         [~,botRPos]=min(abs(tempTC-minRange));
%         viewingDist=botRPos-topRPos; viewingDist=viewingDist+200;
%         
%         t=tempTime(abandon:end); y=tcMF(abandon:end,ii);
%         tempMin=max(y)*0.1;
%         [~,botPos]=min(abs(y-tempMin));
%         viewingRange=botPos-viewingDist:botPos;
%         y(setdiff(1:end,viewingRange))=NaN;
%         tempMax=max(y)*0.9;
%         [~,topPos]=min(abs(y-tempMax));
%         decayTimes(ii)=t(botPos)-t(topPos);
%         %figure(); plot(t,y); hold on; xline(t(botPos)); hold on; xline(t(topPos)); hold on; plot(tempTime,tcMF(:,ii)); title(decayTimes(ii))
%     end
%     y_axis(jj,1:size(varargin{1,1},2))=d2./decayTimes;
% end
% 
% mu=zeros(1,size(varargin{1,1},2));
% for i=1:size(varargin{1,1},2)
%     temp=y_axis(:,i).';
%     slope=polyfit(ab,temp,1);
%     mu(i)=slope(1);
% end
% 
% badPos=find(mu>0);
% for kk=1:length(badPos)
%     temp=y_axis(:,badPos(kk)).';
%     for ll=1:length(varargin)
%         slope=polyfit(ab(setdiff(1:end,ll)),temp(setdiff(1:end,ll)),1);
%         tempMu=slope(1);
%         if tempMu < 0
%             mu(badPos(kk))=tempMu;
%             break;
%         end
%     end
% end


