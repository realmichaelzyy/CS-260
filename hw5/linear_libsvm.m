[train_features, train_labels, test_features, test_labels] = ...
    preprocess(false);

C = [ 4^-6 ; 4^-4 ; 4^-2 ; 4^0 ; 4^2 ];
accus = [];

for i = 1:size(C, 1)
    opts = sprintf('-v 5 -c %f -q', C(i));
    model = svmtrain(train_labels, train_features, opts);
    accus = [ accus ; C(i) model ];
end

disp(' ');
disp('5 fold cross validation for Linear LibSVM')
disp('======================================');
for i = 1:size(C, 1)
    disp(sprintf('C = %.4f | Accuracy = %.8f', accus(i, 1), accus(i, 2)));
end
[a, max_ind] = max(accus);
bestC = accus(max_ind(2), 1);

opts = sprintf('-c %f -q', C(i));
model = svmtrain(train_labels, train_features, opts);
[predicted, test_accu, decision_values] = ...
    svmpredict(test_labels, test_features, model);
disp(sprintf('Best C = %.4f', bestC));