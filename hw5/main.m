% 5.1 - preprocess data
[train_features, train_labels, test_features, test_labels] = ...
    preprocess(true);

% 5.3 - cross validation for SVM
C = [ 4^-6 ; 4^-4 ; 4^-2 ; 4 ; 4^2 ];
num_folds = size(C, 1);
indices = crossvalind('Kfold', size(train_features, 1), num_folds);
accus = [];
for i = 1:num_folds
    val = (indices == i); train = ~val;
    % ADD time measurements
    [w, b] = trainsvm(train_features(train, :), train_labels(train, :), C(i));
    accu = testsvm(train_features(val, :), train_labels(val, :), w, b); 
    accus = [ accus ; C(i) accu ];
end

disp('5 fold cross validation for Linear SVM')
disp('======================================');
for i = 1:num_folds
    disp(sprintf('C = %.4f | Accuracy = %.8f', accus(i, 1), accus(i, 2)));
end
[a, max_ind] = max(accus);
bestC = accus(max_ind(2), 1);
[w, b] = trainsvm(train_features, train_labels, bestC);
disp(' ');
disp(sprintf('Best: C = %.4f | Test Accuracy = %.8f', bestC, ...
             testsvm(test_features, test_labels, w, b)));