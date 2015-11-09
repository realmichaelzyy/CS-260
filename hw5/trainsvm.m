function [w,b] = trainsvm(train_data, train_label, C)
% Train linear SVM (primal form)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  C: tradeoff parameter (on slack variable side)
%
% Output:
%  w: feature vector (column vector)
%  b: bias term
%

    N = size(train_data, 1);
    D = size(train_data, 2);
    
    % initialize minimization goals
    xi = zeros(N, 1);
    w = zeros(D, 1);
    b = 0;
    
    % since we have to fit into matlab quadprog form, make a vector x
    % that will be [ w ; xi ; b ] = ( D + N + 1 ) X 1
    x = [ w ; xi ; b ];
    
    % H will be a (D + N + 1) X (D + N + 1) matrix, but we only want
    % it to save the "w" part of x, so we set the rest to zero
    H = [ eye(D) zeros(D, N+1) ; zeros(N+1, D+N+1) ];
    
    % f will represent the xi sum, so we make it a vector of D zeros, N
    % ones, 1 zero
    f = C * [ zeros(D, 1) ; ones(N, 1) ; 0 ];
    
    % to fit constraint to matlab quadprog form, we again need to make a
    % matrix A: (D + N + 1) X (D + N + 1)
    % each row is [ (y_i * x_i^T) eye(D+N+1) y_i ]  
    % TODO: if this doesn't work, recheck the minus?
    A = -[ bsxfun(@times, train_data, train_label) eye(N) train_label ];
    
    % upper bound is -1's
    ub = -1 * ones(N, 1);
    
    % lower bound on xi's is 0, no lower bounds on the rest
    lb = [ -inf * ones(D, 1) ; zeros(N, 1), ; -inf ];
    
    [x, fval, exitval, output] = quadprog(H, f, A, ub, [], [], lb, []);
    
    w = x(1:D);
    b = x(N+D+1);