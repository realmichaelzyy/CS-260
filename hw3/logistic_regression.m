function [ costs, weights, b ] = logistic_regression( features, labels, step_size, iterations, regularization, debug )
% Performs logistic regression with the specified options
% Arguments:
%   1. features - an NxD matrix where each row is a training example of D
%   features
%   2. labels - an Nx1 matrix of labels
%   3. step_size
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
    de_db = sum(sigma - labels);
    b = b - (step_size * de_db);
    regularization = max(0, regularization);
    weights = weights - step_size * (de_dw' + 2 * regularization * weights);
end

end

