function eigenvecs = pca_fun(X, d)

% Implementation of PCA
% input:
%   X - N*D data matrix, each row as a data sample
%   d - target dimensionality, d <= D
% output:
%   eigenvecs: D*d matrix
%
% usage:
%   eigenvecs = pca_fun(X, d);
%   projection = X*eigenvecs;

    N = size(X, 1);
    
    % make the mean zero everywhere
    X = bsxfun(@minus, X, mean(X));
    
    % compute the covariance matrix
    C_X = (X' * X) / N;
    
    % sort and return top eigenvalue eigenvectors
    [eigenvecs D] = eigs(double(C_X), d);
    %[D,idx] = sort(diag(D), 'descend');
    %eigenvecs = V(:, idx(1:d));
  