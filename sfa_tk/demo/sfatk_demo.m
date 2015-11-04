% see Wiskott, L. and Sejnowski, T.J. (2002), "Slow Feature Analysis:
% Unsupervised Learning of Invariances", Neural Computation, 14(4):715-770,
% Figure 2

msg = ['This demo reproduces an example from Wiskott, L. and Sejnowski,' ...
       ' T.J. (2002), "Slow Feature Analysis: Unsupervised Learning of' ...
       ' Invariances", Neural Computation, 14(4):715-770, Figure 2\n\n'];

fprintf(msg);


% init graphics
figure; clf; set(gcf, 'Position', [89 477 866 338]);

% create the input signal
T = 5000;
t = linspace(0, 2*pi, T);

x1 = sin(t)+cos(11*t).^2;
x2 = cos(11*t);
x = [x1; x2]';

% plot the input signal
subplot(1,3,1); plot(x2,x1);
set(gca, 'Xlim', [-1.2, 1.2], 'YLim', [-1.2,2.2]);
title('input signal x(t)');
xlabel('x_2(t)'); ylabel('x_1(t)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Slow Feature Analysis
%
% create a SFA object
hdl = sfa2_create(2, xp_dim(2), 'PCA');
% perform the preprocessing step
sfa_step(hdl, x, 'preprocessing');
% perform the expansion step
sfa_step(hdl, x, 'expansion');
% close the algorithm
sfa_step(hdl, [], 'sfa');
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute the output signal
y = sfa_execute(hdl, x);

% it would have been quicker (but less instructive) to write:
% [y, hdl] = sfa2(x);

% plot the output of the slowest varying function
subplot(1,3,2); plot(t, y(:,1));
set(gca, 'Xlim', [0,2*pi], 'PlotBoxAspectRatio', [1,1,1])
title('output of the slowest varying function');
xlabel('t'); ylabel('y_1(t)');

% plot the contours of the slowest varying function
n = 100;
[X1,X2] = meshgrid(linspace(-1.2,2.2,n),linspace(-1.2,1.2,n));
testpts = [X1(:) X2(:)];

out = sfa_execute(hdl, testpts);
out = reshape(out(:,1),n,n);

subplot(1,3,3); contourf(X2,X1,out,10);
title('contours of the slowest varying function g_1(x)');
xlabel('x_2'); ylabel('x_1');
