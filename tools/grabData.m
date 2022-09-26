% getData

clc
clear all
close all

oldFolder = pwd;
fileName={'1','2','3','4','5','6','7'};

jvData=[];
for i=1:length(fileName)
    cd(fileName{i})
    cd jv
    jvList=dir('*Inst*.csv');
    for j=1:length(jvList)
        name=jvList(j).name;
        data=readmatrix(name);
        jvData=cat(2,jvData,data(:,1:2));
    end
    cd(oldFolder)
end

tvData=[];
for i=1:length(fileName)
    cd(fileName{i})
    cd tv
    tvList=dir('*Inst*.csv');
    for j=1:length(tvList)
        name=tvList(j).name;
        data=readmatrix(name);
        tvData=cat(2,tvData,data(:,2));
    end
    cd(oldFolder)
end

tcData=[];
for i=1:length(fileName)
    cd(fileName{i})
    cd tc
    tcList=dir('*Inst*.csv');
    for j=1:length(tcList)
        name=tcList(j).name;
        data=readmatrix(name);
        tcData=cat(2,tcData,data(:,2));
    end
    cd(oldFolder)
end

plData=[]; 
cd pl
plList=dir('*PL.csv');
for j=1:length(plList)
    name=plList(j).name;
    data=readmatrix(name);
    plData=cat(2,plData,data);
end
cd(oldFolder)

% Single TC
% tcData=[];
% cd 1
% cd tc
% tcList=dir('*Inst*.csv');
% for j=1:length(tcList)
%     name=tcList(j).name;
%     data=readmatrix(name);
%     tcData=cat(2,tcData,data(:,2));
% end
% cd(oldFolder)

[~,topName]=fileparts(oldFolder);

tcName=append(topName,' tc','.xlsx');
writematrix(tcData,tcName)

tvName=append(topName,' tv','.xlsx');
writematrix(tvData,tvName)

jvName=append(topName,' jv','.xlsx');
writematrix(jvData,jvName)

plName=append(topName,' pl','.xlsx');
writematrix(plData,plName)

