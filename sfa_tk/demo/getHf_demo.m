% This demo script shows how to use the SFA_GETHF function.

% init graphics
figure; clf; set(gcf, 'Position', [618 173 575 727]);

% create the input signal
T = 5000;
t = linspace(0, 2*pi, T);

x1 = sin(t)+cos(11*t).^2+2;
x2 = cos(11*t);
x = [x1; x2]';

% perform slow feature analysis using the default function space,
% i.e. the space of all polynoms of degree 2. The variable HDL contains a
% reference to the Matlab structure that contains the slow-varying
% functions.
[y, hdl] = sfa2(x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the slow-varying functions as quadratic forms

%%%%% To get the activity of the most slowly-varying function on X
%%%%% it's the same to do like this:

y = sfa_execute(hdl, x, 0, 1);
subplot(4,1,1); plot(y);
title('output signal obtained by sfa\_execute');

%%%%% or like this:

% obtain the first function as a quadratic form _in the input space_
% (i.e. WHERE==3)
[H,f,c] = sfa_getHf(hdl, 1, 3);
% The quadratic form is defined as 1/2 * x'*H*x + f'*x + c . If you take
% a look at the coefficients of H, f, and c you see that the learned
% function is 1.4142*(x2^2 - x1 + 2), which corresponds to -1.4142*sin(t),
% a sinus function normalized to have unit variance.

% evaluate the quadratic form on the data
y = zeros(T,1);
for i=1:T,
  v = x(i,:)';
  y(i) = 0.5* v'*H*v + f'*v + c;
end
subplot(4,1,2); plot(y);
title(['output signal obtained with the quadratic form in the input' ...
       ' space']);

%%%%% or like this:

% obtain the first function as a quadratic form _in the input mean-free
% space_ (i.e. WHERE==2)
[H,f,c] = sfa_getHf(hdl, 1, 2);

% compute the mean-free input
x_meanfree = x - repmat(mean(x), T, 1);
% evaluate the quadratic form on the mean-free data
y = zeros(T,1);
for i=1:T,
  v = x_meanfree(i,:)';
  y(i) = 0.5* v'*H*v + f'*v + c;
end
subplot(4,1,3); plot(y);
title(['output signal obtained with the quadratic form in the input mean-free' ...
       ' space']);


%%%%% or even like this:
% obtain the first function as a quadratic form _in the whitened space_
% (i.e. WHERE==0). It is sometimes useful to work in this space,
% especially if you reduced the dimensionality of the input during
% whitening.
[H,f,c] = sfa_getHf(hdl, 1, 0);

% get the whitening matrix
global SFA_STRUCTS
W = SFA_STRUCTS{hdl}.W0;
% compute the whitened input
x_white = x_meanfree*W';
% evaluate the quadratic form on the whitened data
y = zeros(T,1);
for i=1:T,
  v = x_white(i,:)';
  y(i) = 0.5* v'*H*v + f'*v + c;
end
subplot(4,1,4); plot(y);
title(['output signal obtained with the quadratic form in the whitened' ...
       ' space']);

% free the SFA object
sfa_clear(hdl);
