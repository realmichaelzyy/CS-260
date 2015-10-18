function [ costs, weights ] = logistic_regression( features, labels, step_size, iterations, regularization, debug )
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
weights = ones(D, 1);
costs = zeros(iterations, 1);

for i = 1:iterations
    % calculate sigma
    sigma = 1 ./ (1 + exp(-(features * weights)));
    % calculate costs, while lower bounding sigma and 1-sigma at 1e-16
    cost_vec = ((1 - labels) .* log(max(1 - sigma, 1e-16))) + (labels .* log(max(sigma, 1e-16)));
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
    weights = weights - (step_size * de_dw');
    if (regularization > 0)
        weights = weights - (2 * regularization * weights);
    end
end

end

