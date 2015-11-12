[train_features, train_labels, test_features, test_labels] = ...
    preprocess(false);

C = [ 4^-6 ; 4^-5 ; 4^-4 ; 4^-3 ; 4^-2 ; 4^-1 ; 4^0 ; 4^1 ; 4^2 ];
accus = [];

for i = 1:size(C, 1)
    opts = sprintf('-v 5 -c %f -q', C(i));
    t = cputime;
    model = svmtrain(train_labels, train_features, opts);
    e = cputime-t;
    accus = [ accus ; C(i) model e ];
end

disp(' ');
disp('5 fold cross validation for Linear LibSVM')
disp('======================================');
for i = 1:size(C, 1)
    disp(sprintf('%0.6f & %0.6f & %0.6f \\\\', accus(i, 1), accus(i, 3), accus(i, 2)));
    disp('\hline');
end
[a, max_ind] = max(accus);
bestC = accus(max_ind(2), 1);

opts = sprintf('-c %f -q', C(i));
model = svmtrain(train_labels, train_features, opts);
[predicted, test_accu, decision_values] = ...
    svmpredict(test_labels, test_features, model);
disp(sprintf('Best C = %.4f', bestC));