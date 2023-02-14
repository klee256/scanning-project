% preproc.m

function [jv_s,mat_s] = standardize(jv,mat,savef)

% Error checking
% numSets=(nargin-1)/2;
% if mod(numSets,2) ~= 0
%     disp('Uneven inputs, missing JV set or materials parameters')
% end

% For material parameters:
% Remap equation: range [a,b], input x, output y
% y = a + [(x-min(x))/(max(x)-min(x))]*(b-a)

if savef=='y'
    doubleJV=[];
    for i=1:length(jv)
        doubleJV=cat(4,doubleJV,jv{i});
    end

    % 5th index is the location of the center point
    unproc_cV=squeeze(doubleJV(:,5,1,:)); v_mu=mean(unproc_cV,2); v_std=std(unproc_cV,1,2);
    unproc_cJ=squeeze(doubleJV(:,5,2,:)); j_mu=mean(unproc_cJ,2); j_std=std(unproc_cJ,1,2);

    % Standardization
    v_chnl=doubleJV(:,:,1,:); v_chnl=(v_chnl-v_mu)./v_std; v_chnl(isnan(v_chnl))=0;
    j_chnl=doubleJV(:,:,2,:); j_chnl=(j_chnl-j_mu)./j_std; j_chnl(isnan(j_chnl))=0;

    v_j=cat(3,v_chnl,j_chnl);

    jv_s=cell(1,size(unproc_cV,2));
    for i=1:size(unproc_cV,2)
        jv_s{i}=v_j(:,:,:,i);
    end

    % Gaussian (Normal/Standardization) parameters saved to a file
    gaussp.v_mu=v_mu;
    gaussp.v_std=v_std;
    gaussp.j_mu=j_mu;
    gaussp.j_std=j_std;
    save("gauss_param.mat","gaussp")

    max_mat=max(mat);
    min_mat=min(mat);
    mat_s=(mat-min_mat)./(max_mat-min_mat);
    matGauss.max=max_mat;
    matGauss.min=min_mat;
    save("matGauss_param.mat","matGauss")

elseif savef=='n'
    load gauss_param.mat
    doubleJV=[];
    for i=1:length(jv)
        doubleJV=cat(4,doubleJV,jv{i});
    end

    unproc_cV=squeeze(doubleJV(:,5,1,:));

    v_chnl=doubleJV(:,:,1,:); v_chnl=(v_chnl-gaussp.v_mu)./gaussp.v_std; v_chnl(isnan(v_chnl))=0;
    j_chnl=doubleJV(:,:,2,:); j_chnl=(j_chnl-gaussp.j_mu)./gaussp.j_std; j_chnl(isnan(j_chnl))=0;

    v_j=cat(3,v_chnl,j_chnl);

    jv_s=cell(1,size(unproc_cV,2));
    for i=1:size(unproc_cV,2)
        jv_s{i}=v_j(:,:,:,i);
    end

    load matGauss_param.mat
    mat_s=(mat-matGauss.min)./(matGauss.max-matGauss.min);

else
    disp('Error no parameter specified.')
end




%% Legacy

% set_JV=zeros(28,2*size(unproc_2,2));
% set_JV(:,1:2:end)=unproc_V; set_JV(:,2:2:end)=unproc_J;
% 
% min_mat=min(unproc_Mats);
% max_mat=max(unproc_Mats);
% 
% set_Mat=(unproc_Mats-min_mat)./(max_mat-min_mat);
% 
% if (max(set_Mat)~=1) || (min(set_Mat)~=0)
%     disp('Error: Material parameter rescale failed')
% end
% 
% if sum(isnan(set_JV))~=0
%     disp('Error: JV NaN')
% end
% 
% if sum(isnan(set_Mat))~=0
%     disp('Error: Material parameters NaN')
% end
% 
% params=[v_mu,v_std,j_mu,j_std,min_mat,max_mat];



