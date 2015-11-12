[train_features, train_labels, test_features, test_labels] = ...
    preprocess(false);

C = [ 4^-4 ; 4^-3 ; 4^-2 ; 4^-1 ; 4^0 ; 4 ; 4^2 ; 4^3 ;
      4^4 ;  4^5  ; 4^6  ; 4^7 ];
D = [ 1 ; 2 ; 3 ];
accus = [];

for i = 1:size(C, 1)
    for j = 1:size(D, 1)
        opts = sprintf('-t 1 -v 5 -c %f -d %d -q', C(i), D(j));
        t = cputime;
        model = svmtrain(train_labels, train_features, opts);
        e = cputime-t;
        accus = [ accus ; C(i) D(j) model e ];
    end
end

disp(' ');
disp('Polynomial Kernel SVM')
disp('======================================');
for i = 1:size(accus, 1)
    disp(sprintf('%0.6f & %0.6f & %0.6f & %0.6f \\\\', accus(i, 1), accus(i, 2), accus(i, 4), accus(i, 3)));
    disp('\hline');
end
[a, max_ind] = max(accus);

opts = sprintf('-t 1 -c %f -d %d -q', accus(max_ind(3), 1), accus(max_ind(3), 2));
model = svmtrain(train_labels, train_features, opts);
[predicted, test_accu, decision_values] = ...
    svmpredict(test_labels, test_features, model);
disp(sprintf('Best: C = %.4f, D = %d', accus(max_ind(3), 1), accus(max_ind(3), 2))); 
disp(sprintf('Test Accuracy: %0.8f', test_accu(1)));
