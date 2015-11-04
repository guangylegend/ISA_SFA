function [X,MM] = normalized(X,MM)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
min_X = min(X);
max_X = max(X);
for i=1:size(X,1)
    X(i) = (X(i)-min_X)/(max_X-min_X);
    MM(i) = (MM(i)-min_X*size(X,1))/(max_X-min_X);
end
end

