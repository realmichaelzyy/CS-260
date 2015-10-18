% CS260 2015 Fall, Homework 1
1;

function class = classify_vec(features, labels, x, k)
  % first column tracks the distance, second column tracks the class
  diffs = features .- x ;
  dists = sqrt(sum(diffs .^ 2, 2)) ;
  top_k = horzcat(dists, labels) ;
  top_k = sortrows(top_k, 1)(1:k, :) ;
  class = mode(top_k(:, 2));
end

function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)
% k-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  k: number of nearest neighbors
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)

  n = rows(train_data) ;
  training_correct = 0 ;
  % for each training example
  for i = 1:n
    % remove the current example
    removed_train_data = [train_data(1:i-1, :) ; train_data(i+1:end, :)] ;
    removed_train_label = [train_label(1:i-1, :) ; train_label(i+1:end, :)] ;
    % classify example
    class = classify_vec(removed_train_data, removed_train_label, train_data(i, :), k) ;
    if (class == train_label(i))
      training_correct += 1 ;
    endif
  endfor

  n = rows(new_data) ;
  new_correct = 0 ;
  % for each new example
  for i = 1:n
    class = classify_vec(train_data, train_label, new_data(i, :), k) ;
    if (class == new_label(i))
      new_correct += 1 ;
    endif
  endfor

  train_accu = training_correct / rows(train_data);
  new_accu = new_correct / rows(new_data);
end

