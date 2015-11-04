% This demo script solves the same problem as SFATK_DEMO, but in a much
% more complicated way ;) It illustrate how to perform SFA on very long
% data sets.

% divide 2*pi in 5000 parts
T = 5000;

% we have two input signals
input_dim = 2;
% we don't want to reduce the input dimension
pp_dim = input_dim;
% we are only interested in the most slowly-varying signal
sfa_range = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Slow Feature Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create an SFA object and get a reference to it
hdl = sfa2_create(pp_dim, sfa_range, 'PCA');

% cycle over the two SFA steps
for step_name = {'preprocessing', 'expansion'},
  % cycle over your data (generate 100 independent data chunks)
  for i = 1:100,
    %% here you have to generate, load or cut part of your data set %%

    % in this case we generate a small data chunk
    t0 = rand*2*pi; t1 = t0+pi/8;
    t = linspace(t0,t1,T/16);
    x1 = sin(t)+cos(11*t).^2;
    x2 = cos(11*t);
    x = [x1; x2]';
 
    % update the current step
    sfa_step(hdl, x, step_name{1});
  end
end
% close the algorithm
sfa_step(hdl, [], 'sfa');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test on a whole data set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute the input signal
t = linspace(0,2*pi,T);
x1 = sin(t)+cos(11*t).^2;
x2 = cos(11*t);
x = [x1; x2]';
% execute the learned function
y = sfa_execute(hdl,x);
figure; clf; plot(y);
title('output of the most slowly-varying function');

% free the SFA object
sfa_clear(hdl);
