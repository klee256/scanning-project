
clc
clear all
close all

addpath(genpath(fileparts(pwd)));

load 'N57_7cleanJVzoomed'
load 'N57_7deltaVzoomed'

[trainingPack,matPack1]=visNeighborhood(N57_7cleanJVzoomed,N57_7deltaVzoomed,2,'n');

sep=[];
for i=1:9
    sep=cat(2,sep,repmat(i,28,1));
end
sep=sep.';
 
%% Combined Waterfall and Surf Visualization

figure();
% 
% % Vapor-wave 
sideColor=[122/255   134/255   252/255];
colormap(cool)
% 
% % Green/blue
% bg=linspace(0,1,256).';
% bg=cat(2,flip(bg),bg);
% bg(:,3)=zeros(256,1);
% bg=circshift(bg,1,2);
% 
% %sideColor=[0 0 1];
% sideColor=[0/255  81/255   174/255];
% colormap(bg)

% singleOne=763;
% for i=singleOne:singleOne

for i=1900:length(trainingPack)
    currents=trainingPack{i}(:,2:2:end); currents=currents.';
    voltages=trainingPack{i}(:,1:2:end); voltages=voltages.';
    h1=surf(voltages.',sep.',currents.');
    set(h1,'FaceAlpha',0.85)
    
    hold on
    h=waterfall(voltages,sep,currents);
    grid on
    set(h,'FaceColor','flat');
    set(h,'LineWidth',1.5)
    set(h,'EdgeColor',[0 0 0]);
    set(h,'FaceAlpha',0.93)
    set(h,'FaceVertexCData',sideColor)

    zlim([0 1])
    view(60,25)
    xL=xlabel('Scaled Voltage');
    xL.Position=xL.Position+[0 -0.6 0.15];
    xticks(0:0.2:1)
    yL=ylabel('N^{th} Neighbor');
    yL.Position=yL.Position+[0.25 0 0.15];
    yticks(1:1:9)
    zlabel('Scaled Current')
    %exportgraphics(gcf,'testAnimated.gif','Append',true);
    %title(i)
    pause(0.025)
    %pause(1)
    clf
end


%% Finding similar JV curves with different PL intensities
jvSimilar=[]; matSimilar=[];
c=N57_7cleanJVzoomed(:,2:2:end);  c=c.*(-1.0); c=rescale(c);
mat=rescale(N57_7deltaVzoomed);

s=size(N57_7cleanJVzoomed,2)/2;

bestVal=1000;
bestPos=[0,0];
for i=1:s
    tempJV=repmat(c(:,i),1,s);
    mseJV=(tempJV-c).^2; mseJV(:,i)=repmat(100,28,1); mseJV=sum(mseJV);
    
    tempMat=repmat(mat(i),1,s);
    mseMat=(tempMat-mat).^2; mseMat(i)=100; 
    
    [checkVal,checkPos]=min(mseJV+1./mseMat);
    if checkVal<bestVal
        bestVal=checkVal;
        bestPos=[i,checkPos];
    end
end
vv=N57_7cleanJVzoomed(:,1:2:end);
cc=N57_7cleanJVzoomed(:,2:2:end);

%orange=[252/255 115/255 7/255];
color1=[0 1 0];
%color1=[223/255 188/255 94/255];
%color2=[238/255 97/255 70/255];
color2=[1 0 1];

figure();
fSize=12;
plot(vv(:,bestPos(1)),cc(:,bestPos(1)).*-1.0,'Color',color1,'LineWidth',1.2)
hold on
plot(vv(:,bestPos(1)),cc(:,bestPos(2)).*-1.0,'Color',color2,'LineWidth',1.2)
legend("\DeltaV = "+num2str(N57_7deltaVzoomed(bestPos(1))*10^3)+" mV", ...
    "\DeltaV = "+num2str(N57_7deltaVzoomed(bestPos(2))*10^3)+" mV",'FontSize',fSize);
ax =gca;
ax.FontSize=fSize;
xlabel('voltage (V)','FontSize',fSize)
ylabel('current (A)','FontSize',fSize)
%axis square
grid on

jsc=N57_7cleanJVzoomed(1,2:2:end);
dim=sqrt(s);
dataMat=reshape(jsc,dim,dim);
[XX,YY]=meshgrid(linspace(0,dim*0.025,dim));
tickMark=0:0.4:1.6;

figure();
%contourf(XX,YY,dataMat)
pcolor(XX,YY,dataMat)
colormap('turbo')
shading interp
hold on
plot(XX(bestPos(1)), YY(bestPos(1)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color1)
hold on
plot(XX(bestPos(1)), YY(bestPos(1)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color1)
hold on
plot(XX(bestPos(2)), YY(bestPos(2)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color2)
hold on
plot(XX(bestPos(2)), YY(bestPos(2)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color2)
axis square
title('Short-circuit Current Map')
hc=colorbar;
hc.Label.String='I_{sc} (A)';
xticks(tickMark);
yticks(tickMark);
xlabel('Position (mm)'); ylabel('Position (mm)');

figure();
%contourf(XX,YY,reshape(N57_7deltaVzoomed*10^3,dim,dim))
pcolor(XX,YY,reshape(N57_7deltaVzoomed*10^3,dim,dim))
colormap('turbo')
shading interp
hold on
plot(XX(bestPos(1)), YY(bestPos(1)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color1)
hold on
plot(XX(bestPos(1)), YY(bestPos(1)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color1)
hold on
plot(XX(bestPos(2)), YY(bestPos(2)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color2)
hold on
plot(XX(bestPos(2)), YY(bestPos(2)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color2)
axis square
title('Transient Photovoltage Map')
hc=colorbar;
hc.Label.String='\DeltaV (mV)';
xticks(tickMark);
yticks(tickMark);
xlabel('Position (mm)'); ylabel('Position (mm)');


% figure();
% tickMark=0:0.4:1.6;
% 
% subplot(1,2,1);
% contourf(XX,YY,dataMat)
% colormap('turbo')
% hold on
% plot(XX(bestPos(1)), YY(bestPos(1)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color1)
% hold on
% plot(XX(bestPos(1)), YY(bestPos(1)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color1)
% hold on
% plot(XX(bestPos(2)), YY(bestPos(2)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color2)
% hold on
% plot(XX(bestPos(2)), YY(bestPos(2)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color2)
% axis square
% title('J_{sc} Map')
% hc=colorbar;
% hc.Label.String='current (A)';
% xticks(tickMark);
% yticks(tickMark);
% xlabel('Position (mm)'); ylabel('Position (mm)');
% 
% subplot(1,2,2);
% contourf(XX,YY,reshape(N57_7deltaVzoomed*10^3,dim,dim))
% colormap('turbo')
% hold on
% plot(XX(bestPos(1)), YY(bestPos(1)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color1)
% hold on
% plot(XX(bestPos(1)), YY(bestPos(1)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color1)
% hold on
% plot(XX(bestPos(2)), YY(bestPos(2)), 'o', 'LineWidth', 3, 'MarkerSize', 35,'Color',color2)
% hold on
% plot(XX(bestPos(2)), YY(bestPos(2)), '.', 'LineWidth', 0.5, 'MarkerSize', 10,'Color',color2)
% axis square
% title('\DeltaV Map')
% hc=colorbar;
% hc.Label.String='Transient Photovoltage (mV)';
% xticks(tickMark);
% yticks(tickMark);
% xlabel('Position (mm)'); ylabel('Position (mm)');

%% Surf Visualization
% figure();
% for i=1:length(trainingPack)
%     currents=trainingPack{i}(:,2:2:end); 
%     voltages=trainingPack{i}(:,1:2:end); 
%     h=surf(sep.',voltages,currents);
%     colormap([0.529   0.804    0.94])
%     %set(h,'EdgeColor','flat')
%     view(155,15)
%     drawnow
%     pause(0.1)
% end

%% The Neighborhood Visualization
% The window needs to be odd in order for it to work

% XX and YY are inverted idk why
s1=20; s2=37;
s3=24; s4=41;
smallXX=XX(s1:s2,s1:s2); smallYY=YY(s3:s4,s3:s4);
smallMat=dataMat(s1:s2,s3:s4);
diffX=smallXX(1,2)-smallXX(1,1); diffX=diffX-(diffX/2);


t=figure();
t.WindowState = 'maximized';
%colormap('gray')

subplot(1,3,1)
n=3;
spaces=[-n,0,n];
middle=floor(length(smallMat)/2);
pcolor(smallXX,smallYY,smallMat)
%hold on
%plot(0.6,0.8,'gx','MarkerSize', 55,'LineWidth', 3)
for i=1:length(spaces)
    for j=1:length(spaces)
        hold on
        plot(smallXX(middle+spaces(i),middle+spaces(i))+diffX,smallYY(middle+spaces(j),middle+spaces(j))+diffX,'ms','MarkerSize', 15,'MarkerFaceColor','m')
    end
end
axis square
title("n = "+n,'FontSize',20)
set(gca,'XTick',[], 'YTick', [])

subplot(1,3,2)
n=5;
spaces=[-n,0,n];
middle=floor(length(smallMat)/2);
pcolor(smallXX,smallYY,smallMat)
%hold on
%plot(0.6,0.8,'gx','MarkerSize', 55,'LineWidth', 3)
for i=1:length(spaces)
    for j=1:length(spaces)
        hold on
        plot(smallXX(middle+spaces(i),middle+spaces(i))+diffX,smallYY(middle+spaces(j),middle+spaces(j))+diffX,'ms','MarkerSize', 15,'MarkerFaceColor','m')
    end
end
axis square
title("n = "+n,'FontSize',20)
set(gca,'XTick',[], 'YTick', [])

subplot(1,3,3)
n=7;
spaces=[-n,0,n];
middle=floor(length(smallMat)/2);
pcolor(smallXX,smallYY,smallMat)
%hold on
%plot(0.6,0.8,'gx','MarkerSize', 55,'LineWidth', 3)
for i=1:length(spaces)
    for j=1:length(spaces)
        hold on
        plot(smallXX(middle+spaces(i),middle+spaces(i))+diffX,smallYY(middle+spaces(j),middle+spaces(j))+diffX,'ms','MarkerSize', 15,'MarkerFaceColor','m')
    end
end
axis square
title("n = "+n,'FontSize',20)
set(gca,'XTick',[], 'YTick', [])

% exportgraphics(t,'neighborhoodSizes.eps','BackgroundColor','none')

figure();
pcolor(XX,YY,dataMat)
axis square

%% Padding Visualization

% What happens at the corner?
map = [0 0 0.3
    0 0 0.4
    0 0 0.5
    0 0 0.6
    0 0 0.8
    1.0 1.0 1.0];
 
% pad=zeros(8,8);
% pad(5:end,4:end)=rescale(rand(4,5),0.5,1.0);
% n=2; spaces=[-n,0,n];
% middle=floor(length(pad)/2);
% 
% figure();
% pcolor(flipud(pad))
% for i=1:length(spaces)
%     for j=1:length(spaces)
%         hold on
%         %plot(1,1,'ms','MarkerSize', 19,'MarkerFaceColor','m')
%         plot(middle+spaces(j)+0.5,middle+spaces(i)+0.5,'ms','MarkerSize', 19,'MarkerFaceColor','m')
%     end
% end
% colormap(flipud(map))
% axis square

%% Padding Visualization (ALT 1)
% pad=zeros(7,7);
% pad(4:end,3:end)=rescale(rand(4,5),0.5,1.0);
% n=2; spaces=[-n,0,n];
% middle=floor(length(pad)/2);
% 
% figure();
% pcolor(flipud(pad))
% for i=1:length(spaces)
%     for j=1:length(spaces)
%         hold on
%         %plot(1,1,'ms','MarkerSize', 19,'MarkerFaceColor','m')
%         plot(middle+spaces(j)+0.5,middle+spaces(i)+1.5,'ms','MarkerSize', 19,'MarkerFaceColor','m')
%     end
% end
% colormap(flipud(map))
% axis square

%% Padding Visualization (ALT 2)
pad=zeros(8,8);
pad(5:end,4:end)=rescale(rand(4,5),0.5,1.0);
n=3; spaces=[-n,0,n];
middle=floor(length(pad)/2);

figure();
pcolor(flipud(pad))
for i=1:length(spaces)
    for j=1:length(spaces)
        hold on
        %plot(1,1,'ms','MarkerSize', 19,'MarkerFaceColor','m')
        plot(middle+spaces(j)+0.5,middle+spaces(i)+0.5,'ms','MarkerSize', 19,'MarkerFaceColor','m')
    end
end
colormap(flipud(map))
set(gca,'XTick',[], 'YTick', [])
axis square
