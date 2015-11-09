function [ cost ] = cross_entropy ( features, labels, weights, b)
% returns the cross-entropy value of features and labels with the
% given trained weights and bias

    % calculate sigma
    sigma = 1 ./ (1 + exp(-(b + (features * weights))));
    % bound sigma at [1e-16, 1 - 1e-16]
    sigma = max(sigma, 1e-16);
    sigma = min(sigma, 1 - 1e-16);
    % calculate costs, while lower bounding sigma and 1-sigma at 1e-16
    cost_vec = ((1 - labels) .* log(1 - sigma)) + (labels .* log(sigma));
    cost = -sum(cost_vec);
end

