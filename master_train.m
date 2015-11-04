%% open parallel
%parpool(2);
vid_file_string = '/home/legend/AVIClips05/';
path = '/home/legend/ISA_SFA/';
%% load parameters
params = set_network_params();

%% set paths
addpath(path);
addpath([path, '/activation_functions/']);
addpath([path, '/mmread/']);
addpath([path, '/tbox/']);
bases_path = [path, '/bases/']; %store trained filters/bases
unsup_train_data_path = [path, '/unsup_data/']; %a few Gb
data_path_file_name{1} = [unsup_train_data_path,'train_layer1_blks_mat'];
data_path_file_name{2} = [unsup_train_data_path,'train_layer2_blks_mat'];

%% extract unsup training data
%extract_unsupervised_training_data_hw2_layer1(vid_file_string, data_path_file_name{1}, params); 
%extract_unsupervised_training_data_hw2_layer2(vid_file_string, data_path_file_name{2}, params);
 %% train isa layer 1
 train_isa(data_path_file_name{1}, params);
 train_sfa_network(data_path_file_name{1},params);
 
 %% train isa layer 2
 train_isa2(data_path_file_name{2}, params);
 train_sfa2_network(params.layer1_out_all_path_filename,params);
 
 %% delete parallel
% delete(gcp('nocreate'));
