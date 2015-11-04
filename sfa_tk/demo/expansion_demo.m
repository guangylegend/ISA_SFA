% This demo script shows how to perform SFA on user-defined function
% spaces. The user has to overwrite the functions EXPANSION and
% XP_DIM. In this example the two functions are located in the
% subdirectory sfa_tk/demo/expansion_demo/ . Further information can be
% found on the online documentation.

% init graphics
figure; clf; set(gcf, 'Position', [618 173 575 727]);

% !!! The following lines installs the user-defined EXPANSION and XP_DIM
% functions, defined in the expansion_demo/ subdirectory.
% You might need to change this line if you don't start the script from
% the demo/ directory.
addpath expansion_demo/

% create the input signal
T = 5000;
t = linspace(0, 2*pi, T);

x1 = -sin(t)+2*cos(11*t).^4;
x2 = cos(11*t);
x = [x1; x2]';

% show the input signal
clf; subplot(3,1,1); plot(x);
title('input signals');

% linear SFA won't be able to recover the slowest source signal sin(t),
% since it is nonlinearly mixed
fprintf('\nLINEAR SFA\n\n');
y = sfa1(x);
subplot(3,1,2); plot(y(:,1));
title('signal exctracted by linear SFA');

% the user-defined expansion function contains a to-the-fourth nonlinearity,
% which allows SFA to reconstruct the slowest source signal sin(t).
% if the user-defined functions are really used, appropriate messages
% should have been printed on the screen.
fprintf('\nEXPANDED SFA\n\n');
y = sfa2(x);
subplot(3,1,3); plot(y(:,1));
title(['signal exctracted by nonlinear SFA in a', ...
      ' user-defined function space']);

% !!! Remove the user-defined EXPANSION and XP_DIM functions, defined in
% the expansion_demo/ subdirectory. You might need to change this line
% if you don't start the script from the demo/ directory
rmpath expansion_demo/

