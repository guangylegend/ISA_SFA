function V = train_SFA(X)
    [dim,t,n] = size(X);
    input = zeros(t,n*(dim-1));
    for i = 1:n
        tmp = X(:,:,i);
        tmp = whitening(tmp);
        %Z = Y./sqrt((size(X,1)-1)); 
        Z = tmp';
        dev = zeros(t,dim-1);
        for j = 1:dim-1
            dev(:,j) = Z(:,j+1)-Z(:,j);
        end
        input(:,(i-1)*(dim-1)+1:i*(dim-1)) = dev;
    end     
    M = input*input';
    %M = M./size(M,1);
    [V,D] = eig(M);
    V = -V;
    V = V(:,1);
    %Y = -V'*input;
    %Y = Y';

end

