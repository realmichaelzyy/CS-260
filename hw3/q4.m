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
print('ham_reg_many', '-dpng');
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
print('ion_reg_many', '-dpng');
clf

% L2 norm at step size 0.01
for lambda = 0:0.05:0.5
    [costs, weights] = logistic_regression(spam_train_f, spam_train_l, 0.01, 50, lambda, false); 
    [costs2, weights2] = logistic_regression(ion_train_f, ion_train_l, 0.01, 50, lambda, false);
    disp(sprintf('%0.2f & %0.4f & %0.4f \\\\', lambda, norm(weights), norm(weights2)));
    disp('\hline');
end

% 10 plots of train/test curves for different reg coeffecients
for s = 1:length(steps)
    step_size = steps{s};
    x = [];
    spam_train = [];
    spam_test = [];
    ion_train = [];
    ion_test = [];
    for lambda = 0:0.05:0.5
        x = [ x lambda ];
        [train_costs, weights] = logistic_regression(spam_train_f, spam_train_l, step_size, 50, lambda, false);
        spam_train = [ spam_train train_costs(length(train_costs)) ];
        [test_costs, weights] = logistic_regression(spam_test_f, spam_test_l, step_size, 50, lambda, false);
        spam_test = [ spam_test test_costs(length(test_costs)) ];
        [train_costs, weights] = logistic_regression(ion_train_f, ion_train_l, step_size, 50, lambda, false);
        ion_train = [ ion_train train_costs(length(train_costs)) ];
        [test_costs, weights] = logistic_regression(ion_test_f, ion_test_l, step_size, 50, lambda, false);
        ion_test = [ ion_test test_costs(length(test_costs)) ];
    end
    
    hold on
    plot(x, spam_train, 'r-o');
    plot(x, spam_test, 'b-o');
    legend({'train', 'test'});
    title(strcat('EmailSpam Cross-Entropy with step size = ', num2str(step_size)));
    xlabel('Regularization Coeffecient');
    ylabel('Cross-Entropy');
    hold off
    print(strcat('email_reg_', num2str(s)), '-dpng');
    clf 
    
    hold on
    plot(x, ion_train, 'r-o');
    plot(x, ion_test, 'b-o');
    legend({'train', 'test'});
    title(strcat('Ionosphere Cross-Entropy with step size = ', num2str(step_size)));
    xlabel('Regularization Coeffecient');
    ylabel('Cross-Entropy');
    hold off
    print(strcat('ion_reg_', num2str(s)), '-dpng');
    clf 

end