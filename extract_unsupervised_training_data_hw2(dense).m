function [] = extract_unsupervised_training_data_hw2(train_video_dir, result_filename, params)

spatial_size = params.fovea{1}.spatial_size;
temporal_size = params.fovea{1}.temporal_size;
num_patches = params.patches_per_clip;
dirlist = dir([train_video_dir, 'actioncliptrain*']);
num_clips = length(dirlist);
train_filenames = cell(num_clips, 1); 
for i = 1 : num_clips
    train_filenames{i} = dirlist(i).name;
end

X = sample_video_blks(train_video_dir, train_filenames, spatial_size, temporal_size, num_patches);
save(result_filename, 'X', 'spatial_size', 'temporal_size', '-v7.3');
end

function X = sample_video_blks(path, filenames,sp_size, tp_size, num_perclip)
num_clips = length(filenames);

X = cell(num_clips,tp_size);
for i=1:num_clips
    for j=1:tp_size
    X{i,j} = zeros(sp_size^2, num_perclip);
    end
end

%margin = 5;

for i = 17: num_clips
    counter = 1;
    filename = [path, filenames{i}];
    fprintf('loading clip: %s\n', filename);
    M = loadclip_3dm(filename, sp_size, 0, 0);
    stride = 2;
    N = crop_video_blk(M, sp_size*stride, tp_size*stride);
    %[dimx, dimy, dimt] = size(N);
    num_blks = size(N,1)*size(N,2)*size(N,3)/sp_size^2/tp_size/stride^3;
    if num_blks < 200;
        stride = 1;
        N = crop_video_blk(M, sp_size*stride, tp_size*stride);
        num_blks = size(N,1)*size(N,2)*size(N,3)/sp_size^2/tp_size/stride^3;
    end
    Y = cell(tp_size,1);
    for j = 1:tp_size
        Y{j} = zeros(sp_size^2,num_blks);
    end
for x_offset = 0:stride:size(N,1)/sp_size-stride
    for y_offset = 0:stride:size(N,2)/sp_size-stride
        for t_offset = 0:stride:size(N,3)/tp_size-stride
            B = N(x_offset*sp_size+1:(x_offset+1)*sp_size, ...
                y_offset*sp_size+1:(y_offset+1)*sp_size, ...
                t_offset*tp_size+1:(t_offset+1)*tp_size);
            for j=1:tp_size
                %X{i,j}(:, counter)= reshape(B(:,:,i),sp_size^2,[]);
                 Y{j}(:,counter) = reshape(B(:,:,j),sp_size^2,[]);
            end
             counter = counter + 1;
        end
    end
end
      %par
      for j=1:tp_size
          fprintf('do pca on vedio %d at time %d',i,j);
            [V,E,D] = pca(Y{j}');
            X{i,j} = (V(1:num_perclip,:)*Y{j}')';
      end
end

end
