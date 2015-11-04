function [X_features, motion_measure, ds_sections] = transact_dense_samp(M,params,isa_network_all,sfa_network_all)
fovea = params.fovea{1};

ds_st_per_cf_x = params.testds_sp_strides_per_cfovea_x;
ds_st_per_cf_y = params.testds_sp_strides_per_cfovea_y;
ds_st_per_cf_t = params.testds_tp_strides_per_cfovea;

%ds_st_sp_x = params.testds_sp_stride_x;
%ds_st_sp_y = params.testds_sp_stride_y;
%ds_st_tp = params.testds_tp_stride;

[x, y, t] = size(M);
approx_num_samples = max(1, floor((x/fovea.spatial_size-2)*(y/fovea.spatial_size-2)*(t/fovea.temporal_size-2)));

ds_multiple = ds_st_per_cf_x*ds_st_per_cf_y*ds_st_per_cf_t;

%initialization of feature list
X_features = zeros(approx_num_samples*ds_multiple, params.num_features, 'single');    

%initialize motion measure (list)
motion_measure = zeros(approx_num_samples*ds_multiple, 1);


X_fill = 0;
ds_count = 1;
N = crop_video_blk(M, fovea.spatial_size, fovea.temporal_size);
num_blks = size(N,1)*size(N,2)*size(N,3)/fovea.spatial_size^2/fovea.temporal_size;
act_l1 =zeros(params.pca_dim_l1/params.group_size_l1*params.fovea{1}.dense_sample_size,num_blks);
num_samples = num_blks;
x_steps = size(N,1)/fovea.spatial_size;
y_steps = size(N,2)/fovea.spatial_size;
t_steps = size(N,3)/fovea.temporal_size;

batch_nums = 1;
batch_size = ceil(num_blks/batch_nums);

for batch_num = 0:batch_nums-1
    real_batch_size = min(batch_size,num_blks-batch_num*batch_size);
    X = cell(params.fovea{1}.temporal_size,1);
    for i=1:params.fovea{1}.temporal_size
       X{i} = zeros(params.fovea{1}.spatial_size^2,real_batch_size);
    end
    act_isa_l1 = cell(params.num_clips/params.merge_clips,params.fovea{1}.temporal_size);
    act_sfa_l1 = cell(params.num_clips/params.merge_clips,1);
    for batch_offset = 0:real_batch_size-1
        offset = batch_num*batch_size+batch_offset;
        x_offset = floor(offset/(y_steps*t_steps));
        y_offset = mod(floor(offset/t_steps),y_steps); 
        t_offset = mod(offset,t_steps);

        blk = N(x_offset*fovea.spatial_size+1:(x_offset+1)*fovea.spatial_size, ...
          y_offset*fovea.spatial_size+1:(y_offset+1)*fovea.spatial_size, ...
          t_offset*fovea.temporal_size+1:(t_offset+1)*fovea.temporal_size);

        for i=1:params.fovea{1}.temporal_size
            X{i}(:,batch_offset+1)= reshape(blk(:,:,i),params.fovea{1}.spatial_size^2,[]);
        end
    end        
    for i=1:params.num_clips/params.merge_clips
        for j=1:params.fovea{1}.temporal_size
            act_isa_l1{i,j} = activateISA(X{j}, isa_network_all{i,j}{1,1});                                
        end
    end 
    sfa_in = reshape_isa_out_to_sfa_in(act_isa_l1,params,real_batch_size,1);
    %% do sfa
    for i=1:params.num_clips/params.merge_clips
        for j=1:real_batch_size
            sfa_in{i}(:,:,j) = whitening(sfa_in{i}(:,:,j));
            act_sfa_l1{i}(:,:,j) = sfa_in{i}(:,:,j)*sfa_network_all{i};
        end
    end
    act = act_sfa_l1{1}; 
    act_l1(:,batch_num*batch_size+1:(batch_num)*batch_size+real_batch_size) = act;
end

X_features(X_fill+1:X_fill+num_samples, :) = act_l1';
motion_measure(X_fill+1:X_fill+num_samples, 1) = sum(abs(act_l1), 1)';  
ds_sections(ds_count).start = X_fill+1;                       
X_fill = X_fill + num_samples;            
ds_sections(ds_count).end = X_fill;
ds_count = ds_count + 1;

end
