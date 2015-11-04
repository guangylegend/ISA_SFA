function [Xall, MM, indices] = compute_raw_features_s2isa(params, data_file_names, size_idx)

% data_file_names : list of files
% XALL: descriptors for all video clips
% MM  : saliency measure of the descriptors
% indices: track start and end of each clip; also contains ds_sections,
% which tracks sections containing descriptors from differenct dense
% sampling offsets


indices = cell(1);

m = length(data_file_names);
Xall = 0;
switch params.feature.type
    case 'sisa'
        Xall = zeros(3000000, params.num_features, 'single');
end

MM = zeros(3000000, 1, 'single');

Xall_fill = 1;
% total_time = 0;
%global isa_network_all
%global sfa_network_all
load(params.isa_network_all_path_filename);
load(params.sfa_network_all_path_filename);
load(params.isa2_network_all_path_filename);
load(params.sfa2_network_all_path_filename);
for i=1:m
%   tic
    fprintf('%d ',i);
    M = loadclip_3dm([params.avipath{size_idx}, char(data_file_names(i)), '.avi'], params.fovea{params.num_layer}.spatial_size, 0, 0); 
    
    if(params.num_layer==1)
        [X_clip, motionmeasure, ds_sections] = transact_dense_samp(M,params,isa_network_all,sfa_network_all);
    elseif(params.num_layer==2)
        [X_clip, motionmeasure, ds_sections] = transact_dense_samp2(M,params,isa_network_all,sfa_network_all,isa2_network_all,sfa2_network_all);
    end
    %% code to filter out the features with more motion elements, by l1 norm of
    %% activations
            
    indices{i}.start = Xall_fill;

    for ds = 1:length(ds_sections)
       ds_sections(ds).start = ds_sections(ds).start + indices{i}.start - 1 ;
       ds_sections(ds).end = ds_sections(ds).end + indices{i}.start - 1 ;
    end
    
    Xall(Xall_fill:Xall_fill+size(X_clip,1)-1,:) = X_clip;
    
    MM(Xall_fill:Xall_fill+size(X_clip,1)-1,:) = motionmeasure;
    
    Xall_fill = Xall_fill + size(X_clip,1);    
    
    indices{i}.end = Xall_fill-1;

    indices{i}.ds_sections = ds_sections;
    
%   elapsed_time = toc;
%   total_time = total_time + elapsed_time;
%   average_time = total_time / i;
%   fprintf('number of features %d\n', size(X_clip, 1));
%   fprintf('elapsed = %f (seconds)\n', elapsed_time);
%   fprintf('ETA = %f (minutes)\n', average_time * (m-i) / 60);
  
    % print clip index
    %end    
end

Xall = Xall(1:Xall_fill-1,:);

MM = MM(1:Xall_fill-1);

end