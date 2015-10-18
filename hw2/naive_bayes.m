
function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 

N = size(train_data, 1);
D = size(train_data, 2);
table = tabulate(train_label);
% K*1 vector of priors
priors = table(:, 3) / 100;
K = size(priors, 1);
% K*D vector of conditional probs
p_ki = zeros(K, D);
for i = 1:N
    p_ki(train_label(i), :) = p_ki(train_label(i), :) + train_data(i, :);
end
% divide by the class frequencies
p_ki = bsxfun(@rdivide, p_ki, table(:, 2));
% replace unknown probs with 0.1
p_ki(~p_ki) = 0.01;
%disp(p_ki);

new_accurate = 0;
for i = 1:size(new_data, 1)
    if (naive_classify(p_ki, priors, new_data(i, :)) == new_label(i))
        new_accurate = new_accurate + 1;
    end
end

train_accurate = 0;
for i = 1:N
    if (naive_classify(p_ki, priors, train_data(i, :)) == train_label(i))
        train_accurate = train_accurate + 1;
    end
end

new_accu = new_accurate / size(new_data, 1);
train_accu = train_accurate / N;
% CS260 2015 Fall, Homework 2
