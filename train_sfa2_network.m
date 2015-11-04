function [] = train_sfa2_network(train_data_filename,params)
load(train_data_filename);
load(params.isa2_network_all_path_filename);
isa2_out_tp_size = ((params.fovea{2}.temporal_size-params.fovea{1}.temporal_size)/...
    params.stride{1}.temporal_stride+1)*params.fovea{1}.dense_sample_size;
isa2_output = cell(params.num_clips/params.merge_clips2,isa2_out_tp_size);
%% activate ISA output
for i=1:params.num_clips/params.merge_clips
    for j=1:isa2_out_tp_size
       isa2_output{i,j} = activateISA(isa2_in{i,j},isa2_network_all{i,j}{1,1});
    end
end
 %% reshape IS output
 sfa2_in = reshape_isa_out_to_sfa_in(isa2_output,params,params.patches_per_clip2*params.merge_clips2,2);
 sfa2_network_all = cell(params.num_clips/params.merge_clips,1);
 %% train sfa
 for i=1:params.num_clips/params.merge_clips
     fprintf('%d',i);
     sfa2_network_all{i} = train_SFA(sfa2_in{i});
 end
save(params.sfa2_network_all_path_filename, 'sfa2_network_all', '-v7.3');
end

