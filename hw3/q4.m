disp('Loading data...');
[spam_train_f, spam_train_l, spam_test_f, spam_test_l] = bag_of_words();
[ion_train_f, ion_train_l, ion_test_f, ion_test_l] = ionosphere_load();

l2_norms = ones(2, 5);
steps = { 0.001, 0.01, 0.05, 0.1, 0.5 };
plotStyle = {'b-o', 'g-o', 'r-o', 'c-o', 'k-o'};
legendInfo = cell(5, 1);

% plot for ham/spam
hold on
for i = 1:length(steps)
    [costs, weights] = logistic_regression(spam_train_f, spam_train_l, steps{i}, 50, 0.1, false); 
    plot(1:length(costs), costs, plotStyle{i});
    legendInfo{i} = num2str(steps{i});
end
legend(legendInfo);
title('EmailSpam Classification with Regularization');
xlabel('Number Iterations');
ylabel('Cross-Entropy');
hold off
print('ham_reg_01', '-dpng');
clf

% plot for ion
hold on
for i = 1:length(steps)
    [costs, weights] = logistic_regression(ion_train_f, ion_train_l, steps{i}, 50, 0.1, false); 
    plot(1:length(costs), costs, plotStyle{i});
    legendInfo{i} = num2str(steps{i});
end
legend(legendInfo);
title('Ionosphere Classification with Regularization');
xlabel('Number Iterations');
ylabel('Cross-Entropy');
hold off
print('ion_reg_01', '-dpng');

% L2 norm at step size 0.01
for lambda = 0:0.05:0.5
    [costs, weights] = logistic_regression(spam_train_f, spam_train_l, 0.01, 50, lambda, false); 
    [costs2, weights2] = logistic_regression(ion_train_f, ion_train_l, 0.01, 50, lambda, false);
    disp(sprintf('%0.2f & %0.4f & %0.4f \\\\', lambda, norm(weights), norm(weights2)));
    disp('\hline');
end