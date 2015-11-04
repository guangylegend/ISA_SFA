function sfa_in = reshape_isa_out_to_sfa_in(isa_output,params,num_blks,layer)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if(layer==1)
    pca_dim = params.pca_dim_l1;
    group_size = params.group_size_l1;
    merge_clips = params.merge_clips;
    tp_size = params.fovea{layer}.temporal_size;
elseif(layer==2)
    pca_dim = params.pca_dim_l2;
     group_size = params.group_size_l2;
     merge_clips = params.merge_clips2;
     tp_size = ((params.fovea{2}.temporal_size-params.fovea{1}.temporal_size)/...
    params.stride{1}.temporal_stride+1)*params.fovea{1}.dense_sample_size;
end
sfa_in = cell(params.num_clips/merge_clips,1);
 x_size = pca_dim/group_size*params.fovea{layer}.dense_sample_size;
 y_size = tp_size+1-params.fovea{layer}.dense_sample_size;
 z_size = num_blks;
 for i=1:params.num_clips/merge_clips
     sfa_in{i} = zeros(x_size,y_size,z_size);
      for y=1:y_size
          for x=1:params.fovea{layer}.dense_sample_size
              sfa_in{i}((x-1)*pca_dim/group_size+1:x*pca_dim/group_size,y,:) = isa_output{i,y+x-1};
          end
      end
 end
end

