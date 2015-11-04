function params = set_network_params()

%% parallel
params.par_num = 10;

%% layer 
params.num_layer = 2;
%% merge vedio
params.merge_clips = 100;
params.merge_clips2 = 100;
%% number of layers
%params.num_isa_layers = 2;

%% path param
params.vid_file_string = '/home/legend/AVIClips05/';
params.path = '/home/legend/ISA_SFA/';
params.isa_network_all_path_filename = [params.path,'/bases/isa_network_all.mat'];
params.isa2_network_all_path_filename = [params.path,'bases/isa2_network_all.mat'];
params.sfa_network_all_path_filename = [params.path,'/bases/sfa_network_all.mat'];
params.sfa2_network_all_path_filename = [params.path,'/bases/sfa2_network_all.mat'];
params.layer1_out_all_path_filename = [params.path,'/bases/layer1_out.mat'];
params.testid = 'kmeans_test';
%% fovea sizes
fovea{1}.spatial_size = 16;
fovea{1}.temporal_size = 10;
fovea{1}.dense_sample_size = 5;
fovea{2}.spatial_size = 20;
fovea{2}.temporal_size = 14;
fovea{2}.dense_sample_size = 5;

params.fovea = fovea;

%% convolutional strides
stride{1}.temporal_stride = 4;
params.stride = stride;

%% video sampling param
params.patches_per_clip = 200;
params.patches_per_clip2 = 200;
params.num_clips = 100;

%% network feature dimension and group size
% these are the default settings

params.pca_dim_l1 = 50; % number of feature dimensions network layer 1
params.group_size_l1 = 1; % group size

params.pca_dim_l2 = 40;
params.group_size_l2 = 2;

%params.pca_dim_l2 = 200; 
%params.group_size_l2 = 2; 

% results reported in the paper were obtained using group_size_l2 = 4 to 
% save memory on our clusters, but this does not affect results by a lot

params.bases_id{1} = 'layer1_base';

%params.bases_id{2} = sprintf('isa2layer_%dt%d_ts%dt%d_nf%d_gs%d_st%dt%d_l1_%s', ...
%           fovea{1}.spatial_size, fovea{2}.spatial_size, ...
%			  fovea{1}.temporal_size, fovea{2}.temporal_size, ...
%			  params.pca_dim_l2, params.group_size_l2, stride{1}.spatial_stride, stride{1}.temporal_stride, params.bases_id{1});

end


