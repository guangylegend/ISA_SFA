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