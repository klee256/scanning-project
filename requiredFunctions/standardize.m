% preproc.m
% format should be: usage %, jv_1, material parameter_1, jv_2, material parameter_2 ...
% output row vector params: v_mu,v_std,j_mu,j_std,min_mat,max_mat

function [jv_s,mat_s] = standardize(jv,mat)

% numSets=(nargin-1)/2;
% if mod(numSets,2) ~= 0
%     disp('Uneven inputs, missing JV set or materials parameters')
% end

doubleJV=[];
for i=1:length(jv)
    doubleJV=cat(4,doubleJV,jv{i});
end

% 5th index is the location of the center point
unproc_cV=squeeze(doubleJV(:,5,1,:)); v_mu=mean(unproc_cV,2); v_std=std(unproc_cV,1,2);
unproc_cJ=squeeze(doubleJV(:,5,2,:)); j_mu=mean(unproc_cJ,2); j_std=std(unproc_cJ,1,2);

% standardization
v_chnl=doubleJV(:,:,1,:); v_chnl=(v_chnl-v_mu)./v_std; v_chnl(isnan(v_chnl))=0;
j_chnl=doubleJV(:,:,2,:); j_chnl=(j_chnl-j_mu)./j_std; j_chnl(isnan(j_chnl))=0;

v_j=cat(3,v_chnl,j_chnl);

jv_s=cell(1,length(unproc_cV));
for i=1:length(unproc_cV)
    jv_s{i}=v_j(:,:,:,i);
end

% Remap equation: range [a,b], input x, output y
% y = a + [(x-min(x))/(max(x)-min(x))]*(b-a)

%mat_s=rescale(mat)-mean(rescale(mat));

mat_s=mat;



% Legacy

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



