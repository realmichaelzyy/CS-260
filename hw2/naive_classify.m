function class = naive_classify(p_ki, priors, x)
max_prob = -Inf;
class = 0;
for c = 1:size(priors, 1)
    feature_probs = x .* log(p_ki(c, :));
    feature_probs = feature_probs + ((1 - x) .* log(1 - p_ki(c, :)));
    prob = sum(feature_probs) + log(priors(c));
    if prob > max_prob
        max_prob = prob;
        class = c;
    end
end

