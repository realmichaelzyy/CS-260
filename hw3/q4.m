% Q4) Logistic Regression with Regularization

disp('Loading data...');
[spam_train_f, spam_train_l, spam_test_f, spam_test_l] = bag_of_words();
[ion_train_f, ion_train_l, ion_test_f, ion_test_l] = ionosphere_load();

l2_norms = ones(2, 5);
steps = { 0.001, 0.01, 0.05, 0.1, 0.5 };

% plot for ham/spam
hold on
for i = 1:length(steps)
    [costs, weights, b] = logistic_regression(spam_train_f, spam_train_l, steps{i}, 50, 0.1, false); 
    l2_norms(1, i) = norm(weights);
    subplot(length(steps), 1, i);
    plot(1:length(costs), costs);
    title(sprintf('step size = %0.2f', steps{i}));
end
suptitle('EmailSpam Classification')
xlabel('Number Iterations');
ylabel('Cross-Entropy');
hold off
print('emailspam_grad_reg', '-dpng', '-r100');
clf

% plot for ion
hold on
for i = 1:length(steps)
    [costs, weights, b] = logistic_regression(ion_train_f, ion_train_l, steps{i}, 50, 0.1, false); 
    l2_norms(2, i) = norm(weights);
    subplot(length(steps), 1, i);
    plot(1:length(costs), costs);
    title(sprintf('step size = %0.2f', steps{i}));
end
suptitle('Ionosphere Classification')
xlabel('Number Iterations');
ylabel('Cross-Entropy');
hold off
print('ion_grad_reg', '-dpng', '-r100');
clf

% L2 norm at step size 0.01
for lambda = 0:0.05:0.5
    [costs, weights, b] = logistic_regression(spam_train_f, spam_train_l, 0.01, 50, lambda, false); 
    [costs2, weights2, b] = logistic_regression(ion_train_f, ion_train_l, 0.01, 50, lambda, false);
    % uncomment to display L2 Norms in latex table format
    %disp(sprintf('%0.2f & %0.4f & %0.4f \\\\', lambda, norm(weights), norm(weights2)));
    %disp('\hline');
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
        [train_costs, spam_weights, b] = logistic_regression(spam_train_f, spam_train_l, step_size, 50, lambda, false);
        spam_train = [ spam_train cross_entropy(spam_train_f, spam_train_l, spam_weights, b) ];
        spam_test = [ spam_test cross_entropy(spam_test_f, spam_test_l, spam_weights, b) ];
        [train_costs, ion_weights, b] = logistic_regression(ion_train_f, ion_train_l, step_size, 50, lambda, false);
        ion_train = [ ion_train cross_entropy(ion_train_f, ion_train_l, ion_weights, b) ];
        ion_test = [ ion_test cross_entropy(ion_test_f, ion_test_l, ion_weights, b) ];
    end
    
    hold on
    plot(x, spam_train);
    plot(x, spam_test, 'LineStyle', '--');
    legend({'train', 'test'});
    title(strcat('EmailSpam Cross-Entropy with step size = ', num2str(step_size)));
    xlabel('Regularization Coeffecient');
    ylabel('Cross-Entropy');
    hold off
    print(strcat('email_reg_', num2str(s)), '-dpng');
    clf 
    
    hold on
    plot(x, ion_train);
    plot(x, ion_test, 'LineStyle', '--');
    legend({'train', 'test'});
    title(strcat('Ionosphere Cross-Entropy with step size = ', num2str(step_size)));
    xlabel('Regularization Coeffecient');
    ylabel('Cross-Entropy');
    hold off
    print(strcat('ion_reg_', num2str(s)), '-dpng');
    clf 

end