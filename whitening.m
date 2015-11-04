function Y = whitening(X)
    X=X-ones(size(X,1),1)*mean(X);
    [U,D,V]=svd(cov(X));
    eps = 1e-4;
    A=U*inv(diag(sqrt(diag(D)+eps)));
    Y=X*A;
end
