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
sub_blk_nums = x_sub_steps*y_sub_steps*t_sub_steps;

l1_sp_size = params.pca_dim_l1/params.group_size_l1;
l1_tp_size = params.fovea{1}.dense_sample_size;
 
 x_steps = size(N,1)/fovea2.spatial_size;
 y_steps = size(N,2)/fovea2.spatial_size;
 t_steps = size(N,3)/fovea2.temporal_size;
 
 act_l1_pca_reduced = zeros(params.pca_dim_l2/params.group_size_l2*(l1_tp_size*t_sub_steps),num_blks);
 act_l2 =zeros(params.pca_dim_l2/params.group_size_l2*params.fovea{2}.dense_sample_size,num_blks);
 act_l1_abssum= zeros(1, num_blks);
 
batch_nums =5;
batch_size = ceil(num_blks/batch_nums);
for batch_num = 0:batch_nums-1
     real_batch_size = min(batch_size,num_blks-batch_num*batch_size);
     X = cell(params.fovea{1}.temporal_size,sub_blk_nums);
     for ii=1:params.fovea{1}.temporal_size
         for jj=1:sub_blk_nums
            X{ii,jj} = zeros(params.fovea{1}.spatial_size^2,1,real_batch_size);
         end
     end
     isa2_in = cell(l1_tp_size*t_sub_steps,1);
     for batch_offset = 0:real_batch_size-1
            x_offset = floor(batch_offset/(y_steps*t_steps));
            y_offset = mod(floor(batch_offset/t_steps),y_steps); 
            t_offset = mod(batch_offset,t_steps);
            blk = N(x_offset*fovea2.spatial_size+1:(x_offset+1)*fovea2.spatial_size, ...
                y_offset*fovea2.spatial_size+1:(y_offset+1)*fovea2.spatial_size, ...
                t_offset*fovea2.temporal_size+1:(t_offset+1)*fovea2.temporal_size);
            for x_step=1:x_sub_steps
                for y_step=1:y_sub_steps
                    for t_step=1:t_sub_steps
                        %% extract blk data
                        sub_blk_id = (x_step-1)*y_sub_steps*t_sub_steps+(y_step-1)*t_sub_steps+t_step;
                        sub_blk = blk((x_step-1)*stride+1:(x_step-1)*stride+sp1_size,(y_step-1)*stride+1:(y_step-1)*stride+sp1_size,...
                            (t_step-1)*stride+1:(t_step-1)*stride+tp1_size);
                          for ii=1:params.fovea{1}.temporal_size
                               X{ii,sub_blk_id}(:,batch_offset+1)= reshape(sub_blk(:,:,ii),params.fovea{1}.spatial_size^2,[]);
                          end
                          if(sub_blk_id == 1) 
                          end
                    end
                end
            end
     end
        %% calculate convolve l1 output for blk 
     act_isa_l1 = cell(params.num_clips/params.merge_clips,params.fovea{1}.temporal_size,sub_blk_nums);
     act_sfa_l1 = cell(params.num_clips/params.merge_clips,sub_blk_nums);
     for ii=1:params.num_clips/params.merge_clips
        for jj=1:params.fovea{1}.temporal_size
            for kk=1:sub_blk_nums
                act_isa_l1{ii,jj,kk} = activateISA(X{jj,kk}(:,:), isa_network_all{ii,jj}{1,1}); 
            end
        end
     end
     sfa_in = cell(sub_blk_nums,1);
     for kk=1:sub_blk_nums 
        sfa_in{kk} = reshape_isa_out_to_sfa_in(act_isa_l1(:,:,kk),params,real_batch_size,1);
     end
     %% do sfa
     for ii=1:params.num_clips/params.merge_clips
         for jj=1:sub_blk_nums
             for kk = 1:real_batch_size
                sfa_in{jj}{ii}(:,:,kk) = whitening(sfa_in{jj}{ii}(:,:,kk));
                act_sfa_l1{ii,jj}(:,kk) = sfa_in{jj}{ii}(:,:,kk)*sfa_network_all{ii};
             end
         end
     end
     
     %% reconstruct the l1 out block
     l1_out = zeros(l1_sp_size*x_sub_steps*y_sub_steps,l1_tp_size*t_sub_steps,real_batch_size);
     for ii=0:sub_blk_nums-1
         x_index = floor(ii/(y_sub_steps*t_sub_steps))+1;
         y_index = mod(floor(ii/t_sub_steps),y_sub_steps)+1; 
         t_index = mod(ii,t_sub_steps)+1;
         l1_out(((x_index-1)*y_sub_steps+y_index-1)*l1_sp_size+1:((x_index-1)*y_sub_steps+y_index)*l1_sp_size,...
            ((t_index-1)*l1_tp_size)+1:t_index*l1_tp_size,:) = reshape(act_sfa_l1{1,ii+1},l1_sp_size,l1_tp_size,[]);
     end
     % abssum
      act_l1_abssum(:, batch_num*batch_size+1:(batch_num)*batch_size+real_batch_size) = ...
          sum(reshape(abs(l1_out),l1_sp_size*x_sub_steps*y_sub_steps*l1_tp_size*t_sub_steps,[]), 1); 
      % record pca reduced act_l1
     for tt=1:t_sub_steps*l1_tp_size
          act_l1_pca_reduced((tt-1)*params.pca_dim_l2/params.group_size_l2+1:tt*params.pca_dim_l2/params.group_size_l2,...
              batch_num*batch_size+1: batch_num*batch_size+real_batch_size) = ...
              isa2_network_all{1,tt}{1,1}.V(1:params.pca_dim_l2/params.group_size_l2,:)*reshape(l1_out(:,tt,:),size(l1_out,1),size(l1_out,3),[]);
     end
    
       for i=1:l1_tp_size*t_sub_steps
        % sigmoid 
        l1_out(:,i,:) = 1./(1+exp(-l1_out(:,i,:)));
        isa2_in{i} = reshape(l1_out(:,i,:),size(l1_out,1),size(l1_out,3),[]);
      end
      %%.........calculate convolve l1 output for blk .......... 
      %% calculate l2 output for blk
      act_isa_l2 = cell(params.num_clips/params.merge_clips,l1_tp_size*t_sub_steps);
      act_sfa_l2 = cell(params.num_clips/params.merge_clips,1);
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
                act_sfa_l2{ii}(:,jj) = sfa2_in{ii}(:,:,jj)*sfa2_network_all{ii};
            end
        end
        act_sfa_l2 = act_sfa_l2{1};
        % sigmoid
        %act_sfa_l2 = 1./(1+exp(act_sfa_l2));
        %record act_l2
        act_l2(:,batch_num*batch_size+1:(batch_num)*batch_size+real_batch_size) = act_sfa_l2;
     
end
X_fill = 0;
ds_count = 1;
act = [act_l2;act_l1_pca_reduced];
%act = act_l1_pca_reduced;
X_features(X_fill+1:X_fill+num_samples, :) = act';
motion_measure(X_fill+1:X_fill+num_samples, 1) = act_l1_abssum';   
ds_sections(ds_count).start = X_fill+1;            
X_fill = X_fill + num_samples;
ds_sections(ds_count).end = X_fill;
ds_count = ds_count + 1;
end