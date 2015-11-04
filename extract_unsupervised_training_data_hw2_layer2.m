function [] = extract_unsupervised_training_data_hw2_layer2(train_video_dir, result_filename, params)
%% return a num_clips*num_perclip cell, each element is a 3-D block
sp_size = params.fovea{2}.spatial_size;
tp_size = params.fovea{2}.temporal_size;
num_patches = params.patches_per_clip2;
dirlist = dir([train_video_dir, 'actioncliptrain*']);
num_clips = length(dirlist);
rand = randperm(num_clips);
train_filenames = cell(num_clips, 1); 
for i = 1 : num_clips
    train_filenames{i} = dirlist(rand(i)).name;
end

X = sample_video_blks(train_video_dir, train_filenames, params, num_patches);
save(result_filename, 'X', 'sp_size', 'tp_size', '-v7.3');
end

function X = sample_video_blks(path, filenames,params, num_perclip)
num_clips = length(filenames);
sp_size = params.fovea{2}.spatial_size;
tp_size = params.fovea{2}.temporal_size;
X = cell(num_clips,num_perclip);
for i=1:num_clips
    for j=1:num_perclip
           X{i,j} = zeros(sp_size,sp_size,tp_size);
    end
end

margin = 5;

for i = 1 : num_clips
        filename = [path, filenames{i}];
        fprintf('loading clip: %s\n', filename);
        M = loadclip_3dm(filename, sp_size, 0, 0);
    
        [dimx, dimy, dimt] = size(M);
    
        for j = 1 : num_perclip
            %(NOTE) fix the error 
            x_pos = randi([1+margin, dimx-margin-sp_size+1]);
            y_pos = randi([1+margin, dimy-margin-sp_size+1]);
            t_pos = randi([1, dimt-tp_size+1]);
            
            X{i,j} = M(x_pos: x_pos+sp_size-1, y_pos: y_pos+sp_size-1, t_pos: t_pos+tp_size-1);
        end
end

end