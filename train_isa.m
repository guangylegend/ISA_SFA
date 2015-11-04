%train stacked ISA

%this function trains bases for layers of stacked isa model

function [] = train_isa(train_data_filename, params)

fprintf('\n----------------------------------------\n');
fprintf('start training layer 1');

fprintf('------- load training patches: --------- \n');
load(train_data_filename);
isa_network_all = cell(params.num_clips/params.merge_clips,params.fovea{1}.temporal_size);
for i=1:params.num_clips/params.merge_clips
    for j=1:params.fovea{1}.temporal_size
    fprintf('train ISA for video %d at time %d',i,j)
    %fprintf('Removing DC component\n')

    X{i,j} = removeDC(X{i,j});
    %fprintf('Doing PCA and whitening on data or prev layer activations\n')

    [V,E,D] = pca(X{i,j});

    Z = V(1:params.pca_dim_l1 , :)*X{i,j};

   % save_filename = [bases_path, network.isa{params.layer}.bases_id, '.mat'];

    %fprintf('saving bases at %s\n', save_filename);

    isa_network_all{i,j} = isa_est(Z,V(1:params.pca_dim_l1,:), params,1);
    end
end
    save(params.isa_network_all_path_filename, 'isa_network_all', '-v7.3');
end
