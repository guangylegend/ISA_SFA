%parpool(2);
path = './'; %current folder

% option: dense sampling parameter
dense_p = 1;
% 1 for non-overlap grid sampling of features
% 2 for 50% overlap grid sampling of features
%%

% option: use GPU with Jacket 
gpu = 0;
%%

%DEFAULT SET UP FOR HOLLYWOOD2

addpath(path);
addpath([path, '/activation_functions/']);
addpath([path, '/mmread/']);
addpath([path, '/tbox/']);
addpath([path, '/svm/']);
addpath([path, '/svm/chi/']);
addpath([path, '/svm/chi/pwmetric/']);
addpath([path, '/svm/libsvm-mat-3.0-1/']);
bases_path = [path, 'bases/'];


params = set_network_params();

%% number of layers
params.feature.num_layers = 2; %number of layers in the model; 2 is usually enought to give good results
params.feature.catlayers = 1; %whether or not concactenate features from different layers (if 0, only retain top activations)
params.feature.type = 'sisa'; %feature type, 'sisa' for stacked ISA


%% set params
% ***CHANGE TO DESIRED PATHS***
params.afspath = '/home/legend/';
params.savepath = '/home/legend/ISA_SFA/'; %path to save intermediate (before svm classification) results
% params.jacketenginepath = '/**/jacket/engine/'; % option to use Jacket GPU code for MATLAB http://www.accelereyes.com/

%path to videos
params.avipath{1} = [params.afspath, 'AVIClips05/']; 
%path to clipsets (label info)
params.infopath = [params.afspath, 'ClipSets/']; %path for video labels/info

%number of supervised train & test movies (for controlled testing)
params.num_movies =1000; %anything >900 = train/test all movies

%vector quantization
params.num_centroids = 3000; %for kmeans feature aggregation: recommend 1e4 centroids when using linear kernel, 3000 centroids when using chi-squared kernel
params.num_km_init = 3; %number of random kmeans initializations
params.num_km_samples = 1000; %number of kmeans samples *Per Centroid*
params.seed = 10; %random seed for kmeans

%norm binning & filtering (disregard, leave values)
params.usenormbin = 1; %
params.normbinexp = 1; %
params.filtermotion = 0; %

%svm kernel
params.kernel = 'chisq';

params.pydheight = 1;

params.norm_type = 1; %normalization type for svm classification
% [L1-norm]: recommended for chi-squared svm kernel

%evaluation method: 'ap' average precision (one-vs-all, multiple labels per datapoint), 
params.eval = 'ap';

params.num_vid_sizes = 1; %[void] option to use multiple video sizes
params.unscramble = 1; %set 1 Hollywood2: deal with multiple labels/clip

% -- dense sampling params --
% how densely we sample the videos. We commonly use: 1,1,1 non-overlap,
% 2,2,2 50% overlap
params.testds_sp_strides_per_cfovea_x = dense_p; 
params.testds_sp_strides_per_cfovea_y = dense_p; 
params.testds_tp_strides_per_cfovea = dense_p; 
% -- --

% set post activation threshholds
params.postact = set_postact(gpu);

%% ----------------- script to calculate intermediate params-----------------
%calcparams
if params.num_layer ==1
    params.num_features = params.pca_dim_l1/params.group_size_l1*params.fovea{1}.dense_sample_size;
elseif params.num_layer==2
    params.num_features = params.pca_dim_l2/params.group_size_l2*params.fovea{2}.dense_sample_size+...
        params.pca_dim_l2/params.group_size_l2*params.fovea{1}.dense_sample_size*((params.fovea{2}.temporal_size-...
        params.fovea{1}.temporal_size)/params.stride{1}.temporal_stride+1);
end
%% ---------------- make test summary ------------
%params.testid = maketestid(params, network);

%% execute
fprintf('\n--- Feature extraction and classification (Hollywood2) ---\n')
test_features_s2isa(params)
%