function [X_features, motion_measure, ds_sections] = transact_dense_samp2(M,params,isa_network_all,sfa_network_all,isa2_network_all,sfa2_network_all)
fovea2 = params.fovea{2};
ds_st_per_cf_x = params.testds_sp_strides_per_cfovea_x;
ds_st_per_cf_y = params.testds_sp_strides_per_cfovea_y;
ds_st_per_cf_t = params.testds_tp_strides_per_cfovea;

%ds_st_sp_x = params.testds_sp_stride_x;
%ds_st_sp_y = params.testds_sp_stride_y;
%ds_st_tp = params.testds_tp_stride;

[x, y, t] = size(M);
approx_num_samples = max(1, floor((x/fovea2.spatial_size-2)*(y/fovea2.spatial_size-2)*(t/fovea2.temporal_size-2)));

ds_multiple = ds_st_per_cf_x*ds_st_per_cf_y*ds_st_per_cf_t;

%initialization of feature list
X_features = zeros(approx_num_samples*ds_multiple, params.num_features, 'single');    

%initialize motion measure (list)
motion_measure = zeros(approx_num_samples*ds_multiple, 1);
%% start dense sampling: load and calculate features for movies, starting with various offsets
N = crop_video_blk(M, fovea2.spatial_size, fovea2.temporal_size);
num_blks = size(N,1)*size(N,2)*size(N,3)/fovea2.spatial_size^2/fovea2.temporal_size;
num_samples = num_blks;
 
sp1_size = params.fovea{1}.spatial_size;
sp2_size = params.fovea{2}.spatial_size;
tp1_size = params.fovea{1}.temporal_size;
tp2_size = params.fovea{2}.temporal_size;

stride = params.stride{1}.temporal_stride;
x_sub_steps = (sp2_size-sp1_size)/stride+1;
y_sub_steps = (sp2_size-sp1_size)/stride+1;
t_sub_steps = (tp2_size-tp1_size)/stride+1;

l1_sp_size = params.pca_dim_l1/params.group_size_l1;
l1_tp_size = params.fovea{1}.dense_sample_size;
 
 x_steps = size(N,1)/fovea2.spatial_size;
 y_steps = size(N,2)/fovea2.spatial_size;
 t_steps = size(N,3)/fovea2.temporal_size;
 
 act_l1_pca_reduced = zeros(params.pca_dim_l2/params.group_size_l2*(l1_tp_size*t_sub_steps),num_blks);
 act_l2 =zeros(params.pca_dim_l2/params.group_size_l2*params.fovea{2}.dense_sample_size,num_blks);
 act_l1_abssum= zeros(1, num_blks);
 
 batch_nums = 1;
 batch_size = ceil(num_blks/batch_nums);
 for batch_num = 0:batch_nums-1
     real_batch_size = min(batch_size,num_blks-batch_num*batch_size);
     act_isa_l2 = cell(params.num_clips/params.merge_clips,l1_tp_size*t_sub_steps);
     act_sfa_l2 = cell(params.num_clips/params.merge_clips);
     isa2_in = cell(l1_tp_size*t_sub_steps,1);
     for i=1:params.fovea{1}.temporal_size
         isa2_in{i} = zeros(l1_sp_size*x_sub_steps*y_sub_steps,real_batch_size);
     end
    for batch_offset = 0:real_batch_size-1
     offset = batch_num*batch_size+batch_offset;
     isa2_in_tmp = zeros(l1_sp_size*x_sub_steps*y_sub_steps,l1_tp_size*t_sub_steps);
     x_offset = floor(offset/(y_steps*t_steps));
     y_offset = mod(floor(offset/t_steps),y_steps); 
     t_offset = mod(offset,t_steps);
     blk = N(x_offset*fovea2.spatial_size+1:(x_offset+1)*fovea2.spatial_size, ...
                y_offset*fovea2.spatial_size+1:(y_offset+1)*fovea2.spatial_size, ...
                t_offset*fovea2.temporal_size+1:(t_offset+1)*fovea2.temporal_size);
    %% calculate convolve l1 output for blk 
      for x_step=1:x_sub_steps
             for y_step=1:y_sub_steps
                    for t_step=1:t_sub_steps
                       %%...... l1 output for sub_blk..........%%
                        sub_blk = blk((x_step-1)*stride+1:(x_step-1)*stride+sp1_size,(y_step-1)*stride+1:(y_step-1)*stride+sp1_size,...
                            (t_step-1)*stride+1:(t_step-1)*stride+tp1_size);
                        layer = 1;
                         X = cell(params.fovea{layer}.temporal_size,1);
                         for ii=1:params.fovea{layer}.temporal_size
                            X{ii} = zeros(params.fovea{layer}.spatial_size^2,1);
                         end
                         act_isa_l1 = cell(params.num_clips/params.merge_clips,params.fovea{layer}.temporal_size);
                         act_sfa_l1 = cell(params.num_clips/params.merge_clips,1);
                          for ii=1:params.fovea{layer}.temporal_size
                               X{ii}= reshape(sub_blk(:,:,ii),params.fovea{layer}.spatial_size^2,[]);
                          end
                            for ii=1:params.num_clips/params.merge_clips
                                for jj=1:params.fovea{layer}.temporal_size
                                    act_isa_l1{ii,jj} = activateISA(X{jj}, isa_network_all{ii,jj}{1,1});                                
                                end
                            end
                            sfa_in = reshape_isa_out_to_sfa_in(act_isa_l1,params,1,1);
                    %% do sfa
                            for ii=1:params.num_clips/params.merge_clips
                            sfa_in{ii} = whitening(sfa_in{ii});
                            act_sfa_l1{ii} = sfa_in{ii}*sfa_network_all{ii};
                            end
                             sub_blk_act_l1 = act_sfa_l1{1};
                        %%........ l1 output for sub_blk..............%%
                       % add sub_blk_act_l1 to blk_act
                             sp_index = (y_step-1)*x_sub_steps+x_step;
                            isa2_in_tmp((sp_index-1)*l1_sp_size+1:sp_index*l1_sp_size,(t_step-1)*l1_tp_size+1:t_step*l1_tp_size) = ...
                                reshape(sub_blk_act_l1,l1_sp_size,l1_tp_size);
                    end
              end
      end
      act_l1_abssum(:, offset+1) = sum(reshape(abs(isa2_in_tmp),l1_sp_size*x_sub_steps*y_sub_steps*l1_tp_size*t_sub_steps,[]), 1); 
      % record pca reduced act_l1
      tmp = zeros(params.pca_dim_l2/params.group_size_l2*(l1_tp_size*t_sub_steps),1);
      for tt=1:t_step*l1_tp_size
        tmp((tt-1)*params.pca_dim_l2/params.group_size_l2+1:tt*params.pca_dim_l2/params.group_size_l2) = ...
           isa2_network_all{1,tt}{1,1}.V(1:params.pca_dim_l2/params.group_size_l2,:)*isa2_in_tmp(:,tt);
      end   
      act_l1_pca_reduced(:,offset+1) = tmp;
       for i=1:l1_tp_size*t_sub_steps
        % sigmoid 
        %isa2_in_tmp(:,i) = 1./(1+exp(-isa2_in_tmp(:,i)));
        isa2_in{i}(:,batch_offset+1) = isa2_in_tmp(:,i);
      end
      %%.........calculate convolve l1 output for blk .......... 
    end
      %% calculate l2 output for blk
       for ii=1:params.num_clips/params.merge_clips
            for jj=1:l1_tp_size*t_sub_steps
                act_isa_l2{ii,jj} = activateISA(isa2_in{jj}, isa2_network_all{ii,jj}{1,1});                                
            end
       end
       sfa2_in = reshape_isa_out_to_sfa_in(act_isa_l2,params,real_batch_size,2);
        %% do sfa
        for ii=1:params.num_clips/params.merge_clips
            for jj=1:real_batch_size
        sfa2_in{ii}(:,:,jj) = whitening(sfa2_in{ii}(:,:,jj));
        act_sfa_l2{ii}(:,:,jj) = sfa2_in{ii}(:,:,jj)*sfa2_network_all{ii};
            end
        end
        %act_sfa_l2 = find_slowest(act_sfa_l2);
        % sigmoid
        %act_sfa_l2 = 1./(1+exp(act_sfa_l2));
        %record act_l2
        act_l2(:,batch_num*batch_size+1:(batch_num)*batch_size+real_batch_size) = ...
            act_sfa_l2{1};
end
X_fill = 0;
ds_count = 1;
act = [act_l2;act_l1_pca_reduced];
X_features(X_fill+1:X_fill+num_samples, :) = act';
motion_measure(X_fill+1:X_fill+num_samples, 1) = act_l1_abssum';   
ds_sections(ds_count).start = X_fill+1;            
X_fill = X_fill + num_samples;
ds_sections(ds_count).end = X_fill;
ds_count = ds_count + 1;
end