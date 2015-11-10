[train_features, train_labels, test_features, test_labels] = ...
    preprocess(false);

C = [ 4^-4 ; 4^-3 ; 4^-2 ; 4^-1 ; 4^0 ; 4 ; 4^2 ; 4^3 ]; %...
      %4^4 ;  4^5  ; 4^6  ; 4^7 ];
accus = [];

% polynomial
D = [ 1 ; 2 ; 3 ];
for i = 1:size(C, 1)
    for j = 1:size(D, 1)
        opts = sprintf('-t 1 -v 5 -c %f -d %d -q', C(i), D(j));
        disp(opts);
        model = svmtrain(train_labels, train_features, opts);
        accus = [ accus ; C(i) D(j) model ];
    end
end

disp(' ');
disp('Kernel SVM')
disp('======================================');
for i = 1:size(accus, 1)
    disp(sprintf('C = %.4f | D = %d | Accuracy = %.8f', accus(i, 1), accus(i, 2), accus(i, 3)));
end
[a, max_ind] = max(accus);
best = accus(max_ind(3), 1);

disp(sprintf('Best: C = %.4f, D = %d', accus(max_ind(3), 1), accus(max_ind(3), 2)));
