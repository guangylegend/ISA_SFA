function [] = train_sfa_network(train_data_filename,params)
load(train_data_filename);
load(params.isa_network_all_path_filename);
isa_output = cell(params.num_clips/params.merge_clips,params.fovea{1}.temporal_size);
%% activate ISA output
for i=1:params.num_clips/params.merge_clips
    for j=1:params.fovea{1}.temporal_size
       isa_output{i,j} = activateISA(X{i,j},isa_network_all{i,j}{1,1});
    end
end
 %% reshape IS output
 %{
 Y = cell(100,1);
 for i=1:params.num_clips
     Y{i} = zeros(params.pca_dim_l1*params.fovea{1}.temporal_size,params.patches_per_clip);
     for j=1:params.fovea{1}.temporal_size
         Y{i}((j-1)*params.pca_dim_l1+1:j*params.pca_dim_l1,:) = isa_output{i,j}(:,:);
     end
 end
 %}
 sfa_in = reshape_isa_out_to_sfa_in(isa_output,params,params.patches_per_clip*params.merge_clips,1);
 sfa_network_all = cell(params.num_clips/params.merge_clips,1);
 %% train sfa
 for i=1:params.num_clips/params.merge_clips
     fprintf('%d',i);
     sfa_network_all{i} = train_SFA(sfa_in{i});
 end
 save(params.sfa_network_all_path_filename,'sfa_network_all');
end

