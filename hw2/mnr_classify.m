function [ accu ] = mnr_classify(X, Y)
    features = mnrfit(X, Y);
    probs = mnrval(features, X);
    [m, i] = max(probs, [], 2);
    accu = sum(i == Y) / size(Y, 1);
end

