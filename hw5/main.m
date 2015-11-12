% 5.1 - preprocess data
[train_features, train_labels, test_features, test_labels] = ...
    preprocess(true);

% 5.3 - cross validation for SVM
C = [ 4^-6 ; 4^-5 ; 4^-4 ; 4^-3 ; 4^-2 ; 4^-1 ; 4^0 ; 4^1 ; 4^2 ];
num_folds = 5;
accus = [];
for c = 1:size(C, 1)
    indices = crossvalind('Kfold', size(train_features, 1), num_folds);
    avg_accs = [];
    avg_times = [];
    for i = 1:num_folds
        val = (indices == i); train = ~val;
        % ADD time measurements
        t = cputime;
        [w, b] = trainsvm(train_features(train, :), train_labels(train, :), C(c));
        e = cputime-t;
        accu = testsvm(train_features(val, :), train_labels(val, :), w, b); 
        avg_accs = [ avg_accs ; accu ];
        avg_times = [ avg_times ; e ];
    end
    accus = [ accus ; C(c) mean(avg_times) mean(avg_accs) ];
end

disp('5 fold cross validation for Linear SVM')
disp('======================================');
for i = 1:size(C, 1)
    disp(sprintf('%0.6f & %0.6f & %0.6f \\\\', accus(i, 1), accus(i, 3), accus(i, 2)));
    disp('\hline');
end
[a, max_ind] = max(accus);
bestC = accus(max_ind(3), 1);
[w, b] = trainsvm(train_features, train_labels, bestC);
disp(' ');
disp(sprintf('Best: C = %.4f | Test Accuracy = %.8f', bestC, ...
             testsvm(test_features, test_labels, w, b)));