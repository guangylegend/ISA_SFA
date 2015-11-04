function [] = train_isa2(train_data_filename, params)
fprintf('\n----------------------------------------\n');
fprintf('start training layer 1');

fprintf('------- load training patches: --------- \n');
load(train_data_filename);
all_train_blk = X;
load(params.isa_network_all_path_filename);
load(params.sfa_network_all_path_filename);

sp1_size = params.fovea{1}.spatial_size;
sp2_size = params.fovea{2}.spatial_size;
tp1_size = params.fovea{1}.temporal_size;
tp2_size = params.fovea{2}.temporal_size;
stride = params.stride{1}.temporal_stride;
x_steps = (sp2_size-sp1_size)/stride+1;
y_steps = (sp2_size-sp1_size)/stride+1;
t_steps = (tp2_size-tp1_size)/stride+1;
l1_sp_size = params.pca_dim_l1/params.group_size_l1;
l1_tp_size = params.fovea{1}.dense_sample_size;
%blk_act_l1 = zeros(l1_sp_size*x_steps*y_steps,l1_tp_size*t_steps);
isa2_in_tmp = cell(params.num_clips/params.merge_clips2,1);
%{
for i=1:params.num_clips/params.merge_clips2
    isa2_in_tmp{i} = zeros(l1_sp_size*x_steps*y_steps,params.merge_clips*params.patches_per_clip2,l1_tp_size*t_steps);
    for m=1:params.merge_clips2
        for j=1:params.patches_per_clip2
            fprintf('%d %d\n',m,j);
            for x_step=1:x_steps
                for y_step=1:y_steps
                    for t_step=1:t_steps
                        blk = all_train_blk{(i-1)*params.merge_clips2+m,j}((x_step-1)*stride+1:(x_step-1)*stride+sp1_size,(y_step-1)*stride+1:(y_step-1)*stride+sp1_size,...
                            (t_step-1)*stride+1:(t_step-1)*stride+tp1_size);
                        layer = 1;
                         X = cell(params.fovea{layer}.temporal_size,1);
                         for ii=1:params.fovea{layer}.temporal_size
                            X{ii} = zeros(params.fovea{layer}.spatial_size^2,1);
                         end
                         act_isa_l1 = cell(params.num_clips/params.merge_clips,params.fovea{layer}.temporal_size);
                         act_sfa_l1 = cell(params.num_clips/params.merge_clips,1);
                          for ii=1:params.fovea{layer}.temporal_size
                               X{ii}= reshape(blk(:,:,i),params.fovea{layer}.spatial_size^2,[]);
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
                            act = find_slowest(act_sfa_l1);
                             sub_blk_act_l1 = act;
                             sp_index = (y_step-1)*x_steps+x_step;
                            isa2_in_tmp{i}((sp_index-1)*l1_sp_size+1:sp_index*l1_sp_size,(m-1)*params.patches_per_clip2+j,(t_step-1)*l1_tp_size+1:t_step*l1_tp_size) = ...
                                reshape(sub_blk_act_l1,l1_sp_size,l1_tp_size);
                    end
                end
            end
        end
    end
end

isa2_in = cell(params.num_clips/params.merge_clips2,l1_tp_size*t_steps);
for i=1:params.num_clips/params.merge_clips2
    for j=1:l1_tp_size*t_steps
    isa2_in{i,j} = isa2_in_tmp{i}(:,:,j);
    % sigmoid
    isa2_in{i,j} = 1./(1+exp(isa2_in{i,j}));
    end
end
save(params.layer1_out_all_path_filename, 'isa2_in', '-v7.3');
%}
load(params.layer1_out_all_path_filename);
isa2_network_all = cell(params.num_clips/params.merge_clips2,l1_tp_size*t_steps);
for i=1:params.num_clips/params.merge_clips2
    for j=1:l1_tp_size*t_steps
        fprintf('train isa2 for video %d at time %d\n',i,j);
         isa2_in{i,j} = removeDC(isa2_in{i,j});
         [V,E,D] = pca(isa2_in{i,j}(:,:));
         Z = V(1:params.pca_dim_l2 , :)*isa2_in{i,j};
         isa2_network_all{i,j} = isa_est(Z,V(1:params.pca_dim_l2,:), params,2);
    end
end
save(params.isa2_network_all_path_filename, 'isa2_network_all', '-v7.3');
end
