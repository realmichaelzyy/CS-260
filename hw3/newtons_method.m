function [ costs, weights, b ] = newtons_method( features, labels, iterations, regularization, debug )
% Performs newtons method with the specified options
% Arguments:
%   1. features - an NxD matrix where each row is a training example of D
%   features
%   2. labels - an Nx1 matrix of labels
%   4. iterations
%   5. regularization - coeffecient for regularization (lambda). If less
%   than 0, do not perform any regularization
%   6. debug - if true, then print out the cost after each iteration

N = size(features, 1);
D = size(features, 2);
weights = zeros(D, 1);
b = 0.1;
costs = zeros(iterations, 1);

for i = 1:iterations
    % calculate sigma
    sigma = 1 ./ (1 + exp(-(b + (features * weights))));
    % bound sigma at [1e-16, 1 - 1e-16]
    sigma = max(sigma, 1e-16);
    sigma = min(sigma, 1 - 1e-16);
    % calculate costs, while lower bounding sigma and 1-sigma at 1e-16
    cost_vec = ((1 - labels) .* log(1 - sigma)) + (labels .* log(sigma));
    cost = -sum(cost_vec);
    if (regularization > 0)
        cost = cost + regularization * norm(weights);
    end
    costs(i) = cost;

    if (debug)
        disp(sprintf('Iteration %d', i));
        disp(sprintf('Cost: %0.16f', cost));
    end
    
    % update weights
    de_dw = sum(bsxfun(@times, (sigma - labels), features));
    de_dsigma = sigma .* (1 - sigma);
    hessian = features' * diag(de_dsigma) * features;
    if (regularization > 0)
        hessian = hessian + (eye(size(hessian)) * 2 * regularization);
    end
    if (regularization > 0)
        weights = weights - pinv(hessian) * (de_dw' + (2 * regularization * weights));
    else
        weights = weights - pinv(hessian) * de_dw';
    end

    % update b 
    de_db = sum(sigma - labels);
    b = b - (de_db / sum(de_dsigma));
end


end

