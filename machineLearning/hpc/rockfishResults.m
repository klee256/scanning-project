clc
clear all
close all

filePathName=matlab.desktop.editor.getActiveFilename;
[folderPath,~,~] = fileparts(filePathName);
cd(folderPath)
oldFolder=pwd;

versionN="4008";
% 
% cd(version)
% 
t=importdata(dir("*training*").name);
v=importdata(dir("*validation*").name);
% 
[~,best]=min(v);
% 
figure();
plot(t,'b')
hold on
plot(v,'r')
xlabel('Epochs')
axis square
grid on
title(versionN+" Training Curve"+"; Best: "+int2str(best)+"; Acc: "+num2str(v(best)))
% % saveas(gcf,version+".png")
% 
% figure();
% plot(v,'r')
% xlabel('Epochs')
% axis square
% grid on
% 
% % encodersAll=importdata(dir("allEncoders*").name);
% % encoderNet=encodersAll{best};
% % save("encoderNet"+version,'encoderNet');
%  
% cd(oldFolder)


%% 
% versionNet=1015:1018;
% 
% encoderNames=[];
% for i=1:length(versionNet)
%     encoderNames=cat(2,encoderNames,"encoder"+int2str(versionNet(i))+".mat");
% end
% 
% for i = 1:length(versionNet)
%     currentLook=int2str(versionNet(i));
%     cd(currentLook)
%     t=importdata(dir("*training*").name);
%     v=importdata(dir("*validation*").name);
% 
%     [~,best2]=min(t); t(best2)*100
%     [~,best1]=min(v); v(best1)*100
%     
% 
%     figure();
%     plot(t,'b')
%     hold on
%     plot(v,'r')
%     xlabel('Epochs')
%     axis square
%     grid on
%     title(currentLook+" Training Curve"+"; Best: "+int2str(best)+"; Acc: "+num2str(v(best)))
%     saveas(gcf,currentLook+".png")
% 
%     encodersAll=importdata(dir("allEncoders*").name);
%     encoderNet=encodersAll{best};
%     save(encoderNames(i),'encoderNet');
%     cd(oldFolder)
% end

