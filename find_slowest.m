function rec = find_slowest(input)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
min = sum_deviration(input{1});
rec = input{1};
for i=2:size(input,1)
    tmp = sum_deviration(input{i});
    if tmp<min
        min = tmp;
        rec = input{i};
    end
end
end

function A = sum_deviration(input)
A=0;
for i=1:size(input,1)-1
    A = A+(input(i+1)-input(i))^2;
end
end

