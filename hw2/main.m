
%printf('Loading data...\n');
[train_data, train_labels] = one_hot('car_train.data', 950);
[test_data, test_labels] = one_hot('car_test.data', 389);
[valid_data, valid_labels] = one_hot('car_valid.data', 389);

% 4a - naive bayes 
[new_accu, train_accu] = naive_bayes(train_data, train_labels, test_data, test_labels)
[valid_accu, train_accu] = naive_bayes(train_data, train_labels, valid_data, valid_labels)

% 4b - decision tree's TODO
% for criterion = {'gdi' , 'deviance' }
%     disp(sprintf('Criterion: %s', criterion{1}));
%     for min_leaf = 1:10
%         % pruning off for training
%         tree = fitctree(train_data, train_labels, 'Prune', 'off', 'SplitCriterion', criterion{1}, 'minleaf', min_leaf);
%         train_accu = sum(predict(tree, train_data) == train_labels) / size(train_labels, 1);
%         % pruning on for test/validation
%         tree = fitctree(train_data, train_labels, 'Prune', 'on', 'SplitCriterion', criterion{1}, 'minleaf', min_leaf);
%         valid_accu = sum(predict(tree, valid_data) == valid_labels) / size(valid_labels, 1);
%         test_accu = sum(predict(tree, test_data) == test_labels) / size(test_labels, 1);
%         disp(sprintf('Min Leaf: %d. Accuracy: %f', min_leaf, accuracy));
%         %disp('\hline');
%         %disp(sprintf('%s & %d & %f & %f & %f \\\\', criterion{1}, min_leaf, train_accu, valid_accu, test_accu));
%     end
% end

% 4b - mnr fit
% train_accu = mnr_classify(train_data, train_labels)
% valid_accu = mnr_classify(valid_data, valid_labels)
% new_accu = mnr_classify(test_data, test_labels)