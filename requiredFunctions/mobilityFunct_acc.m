
% tS = time scale on o-scope (tS=[2.5e-6,2.5e-6,1.0e-6,2.5e-6];)
% ab = applied bias (ab=[0,0.1,0.2,0.3];)
% d = film thickness
% ab = applied bias (100 mV, 200mV, etc) in array [0,1,2,3]

function mu = mobilityFunct(tS,ab,varargin)

if length(varargin) ~= length(tS)
    disp('Error 1: timescale mismatch')
end

d=400e-9; d2=(d*100)^2; % film thickness and its square (cm^2)
one_e=1/(exp(1).^2);
abandon_filt=300; % need to drop the first 300 values when using the filter (will decay near edge)
%medFiltStr=50; % med filter does not work as well as avg (below)
filt=ones(1,250)./250;

y_axis=zeros(length(varargin),size(varargin{1,1},2));
for ii = 1:length(varargin)
    time=linspace(0,10*tS(ii),2500);
    tcData=varargin{1,ii}; tcData=tcData-mean(tcData(2100:2500,:));

    % tc=medfilt1(tcData,medFiltStr); 
    tc=zeros(size(tcData));
    for ij=1:length(tcData)
        tc(:,ij)=filter(filt,1,tcData(:,ij));
    end

    t=time(abandon_filt:end); 
    y=tc(abandon_filt:end,:); y=y-mean(y(1500:end,:));

    tempMax=mean(y(1:500,:))*(1-one_e); 
    tempMid=mean(y);                    
    tempMin=mean(y(1:500,:))*one_e;

    y_valid=y; y_valid(1:500,:)=NaN;
    [~,topPos]=min(abs(y_valid-tempMax));
    [~,midPos]=min(abs(y_valid-tempMid));
    [~,botPos]=min(abs(y_valid-tempMin));

    halfDist_1=midPos-topPos; % space between top ind and middle ind
    halfDist_2=botPos-midPos; 

    if sum(find(halfDist_1<0)) ~= 0
        disp('Error 2: Middle position is less than top position')
    end
    
    if sum(find(halfDist_2<0)) ~= 0
        disp('Error 3: Bottom position is less than middle position')
    end

    checkMid=cat(2,find(midPos>1500),find(midPos<700)); % traces are centered in the o-scope, so should be within 800-1400
    if ~isempty(checkMid)
        disp('Error 4: Bad middle position')
    end
    
    sweep_1=find(isoutlier(t(botPos)-t(topPos)));
    % sweep_1=sort(cat(2,find(halfDist_1>2*halfDist_2),find(halfDist_2>2*halfDist_1))); % looking for bad top/bot positions

    for ik=1:length(sweep_1)
        badInd=sweep_1(ik);

        % Case 1: Bot pos is further away than actual, but top pos is correct
        if (halfDist_2(badInd)>2*halfDist_1(badInd)) && (halfDist_1(badInd)<300) && (halfDist_1(badInd)>75) && (topPos(badInd)>600) % check if the top pos is reasonable, then use that as reference
            b_range=ceil(midPos(badInd)+1.9*halfDist_1(badInd)); % new end pos (range) to search for bot pos
            temp_y = y(:,badInd); temp_y(~ismember(temp_y,temp_y(midPos(badInd):b_range)))=NaN;
            temp_curve=-1.0*abs(temp_y-tempMin(badInd));
            [pk_val,temp_botPos,~,~]=findpeaks(temp_curve,'MinPeakWidth',8);
            if isempty(temp_botPos)
                disp("Error 5: No new min pos detected. Problem with: "+int2str(badInd))
                botPos(badInd)=b_range;
                break;
            end
            botPos(badInd)=temp_botPos(find(pk_val==max(pk_val)));

        elseif (halfDist_2(badInd)>2*halfDist_1(badInd)) && (halfDist_1(badInd)<300) && (topPos(badInd)>600) % mid pos is too close to top pos, so can't reach bottom value (i.e. halfDist_1 is invalid)
            b_range=ceil(midPos(badInd)+250); % new end pos (range) to search for bot pos
            temp_y = y(:,badInd); temp_y(~ismember(temp_y,temp_y(midPos(badInd):b_range)))=NaN;
            temp_curve=-1.0*abs(temp_y-tempMin(badInd));
            [pk_val,temp_botPos,~,~]=findpeaks(temp_curve,'MinPeakWidth',8);
            if isempty(temp_botPos)
                disp("Error 5: No new min pos detected. Problem with: "+int2str(badInd))
                break;
            end
            botPos(badInd)=temp_botPos(find(pk_val==max(pk_val)));
        end

        % Case 2: Top pos is further away than actual, but bot pos is correct
        if (halfDist_1(badInd)>2*halfDist_2(badInd)) && (halfDist_2(badInd)<300) && (halfDist_2(badInd)>75) && (botPos(badInd)<1500)
            b_range=ceil(midPos(badInd)-1.9*halfDist_2(badInd)); % new beginning pos (range) to search for top pos
            temp_y = y(:,badInd); temp_y(~ismember(temp_y,temp_y(b_range:midPos(badInd))))=NaN;
            temp_curve=-1.0*abs(temp_y-tempMax(badInd));
            [pk_val,temp_topPos,~,~]=findpeaks(temp_curve,'MinPeakWidth',8);
            if isempty(temp_topPos)
                disp("Error 6: No new max pos detected. Problem with: "+int2str(badInd))
                break;
            end
            topPos(badInd)=temp_topPos(find(pk_val==max(pk_val)));

        elseif (halfDist_1(badInd)>2*halfDist_2(badInd)) && (halfDist_2(badInd)<300) && (botPos(badInd)<1500)
            b_range=ceil(midPos(badInd)-250); % new beginning pos (range) to search for top pos
            temp_y = y(:,badInd); temp_y(~ismember(temp_y,temp_y(b_range:midPos(badInd))))=NaN;
            temp_curve=-1.0*abs(temp_y-tempMax(badInd));
            [pk_val,temp_topPos,~,~]=findpeaks(temp_curve,'MinPeakWidth',8);
            if isempty(temp_topPos)
                disp("Error 6: No new max pos detected. Problem with: "+int2str(badInd))
                break;
            end
            topPos(badInd)=temp_topPos(find(pk_val==max(pk_val)));
        end
    end
    
    %% Debugging Plots and Commands
    % [a1,a2,a3,a4]=findpeaks(temp_curve);
    % figure(); plot(temp_curve); findpeaks(temp_curve,'MinPeakWidth',8);
    % nC=1914; figure(); plot(y(:,nC)); hold on; xline(topPos(:,nC)); hold on; yline(tempMax(:,nC),'g'); hold on; xline(botPos(:,nC)); hold on; yline(tempMin(:,nC),'b');  hold on; yline(tempMid(:,nC),'r'); hold on; yline(mean(y(1:500,nC))); findpeaks(y(:,nC),'MinPeakWidth',20); title("Bot pos: "+int2str(botPos(nC))+" Top pos: "+int2str(topPos(nC)))
    %%

    %sweep_1c=find(isoutlier(t(botPos)-t(topPos)));

    decayTimes=t(botPos)-t(topPos);
    y_axis(ii,1:size(varargin{1,1},2))=d2./decayTimes;
end

mu=zeros(1,size(varargin{1,1},2));
for i=1:size(varargin{1,1},2)
    temp=y_axis(:,i).';
    slope=polyfit(ab,temp,1);
    mu(i)=slope(1);
end

% Try removing first position (0V) and recalc mobility 
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
% Try removing 2 points and recalc mobility
% permSet=1:1:length(varargin);
% permutations=nchoosek(permSet,2);
% badPos=find(mu>0);
% chosenPerm=zeros(1,length(badPos));
% for kk=1:length(badPos)
%     temp=y_axis(:,badPos(kk)).';
%     for ll=1:length(permutations)
%         slope=polyfit(ab(permutations(ll,:)),temp(permutations(ll,:)),1);
%         tempMu=slope(1);
%         if tempMu < 0
%             mu(badPos(kk))=tempMu;
%             chosenPerm(kk)=ll;
%             break;
%         end
%     end
% end

badPos=find(mu>0);
if ~isempty(badPos)
    disp("Error: Unable to calculate all positive mobilities for "+int2str(length(badPos))+" values.\n")
end
outInd=find(isoutlier(mu));
rejectInd=unique(cat(2,badPos,outInd));
disp("Total rejects: "+int2str(length(rejectInd)))

% mu(badPos)=NaN;
% mu=mu*(-1.0);





%% Calculating mobility via sigmoid fit (sigmoid(-x))
% Does not work. Mobility values will average out (across multiple biases for the same point).
% Might be due to the decay not actually being sigmoid. The fit for the
% tail end is poor.
% 
% d = 400e-9; % film thickness 
% d2=(d*100)^2; % square film thickness (cm)
% abandon=300;
% y_axis=zeros(length(varargin),size(varargin{1,1},2));
% for jj = 1:length(varargin)
%     tcData=varargin{1,jj}; tc=tcData-mean(tcData(2300:2500,:));
%     time=linspace(0,10*tS(jj),2500);
%     tcMF=medfilt1(tc,125); tcMF=tcMF(abandon:end,:);
%     ft = fittype('a/(b+exp(c*x+d))','independent',{'x'},'coefficients',{'a','b','c','d'});
%     decayTimes=zeros(1,size(varargin{1,1},2));
%     for ii=1:size(varargin{1,1},2)
%         [~,meanPos]=min(abs(tcMF(:,ii)-mean(tcMF(:,ii))));
%         tempTime=time-time(meanPos); tempTime=tempTime.';
%         tempTime=tempTime(abandon:end); 
%         fv=fit(tempTime,tcMF(:,ii),ft,'StartPoint',[0.0113,1.0,2.128e+06,0]);
%         tempTC=fv.a./(fv.b+exp(fv.c*tempTime+fv.d));
%         maxRange=max(tempTC)*0.9;
%         minRange=max(tempTC)*0.1;
%         [~,topRPos]=min(abs(tempTC-maxRange));
%         [~,botRPos]=min(abs(tempTC-minRange));
%         decayTimes(ii)=tempTime(botRPos)-tempTime(topRPos);
% 
% %         viewingDist=botRPos-topRPos; viewingDist=viewingDist+200;
% %         
% %         t=tempTime(abandon:end); y=tcMF(abandon:end,ii);
% %         tempMin=max(y)*0.1;
% %         [~,botPos]=min(abs(y-tempMin));
% %         viewingRange=botPos-viewingDist:botPos;
% %         y(setdiff(1:end,viewingRange))=NaN;
% %         tempMax=max(y)*0.9;
% %         [~,topPos]=min(abs(y-tempMax));
% %         decayTimes(ii)=t(botPos)-t(topPos);
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



