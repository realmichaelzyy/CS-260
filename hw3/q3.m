% Q3) Logistic Regression without optimization

disp('Loading data...');
[spam_train_f, spam_train_l, spam_test_f, spam_test_l] = bag_of_words();
[ion_train_f, ion_train_l, ion_test_f, ion_test_l] = ionosphere_load();

l2_norms = ones(2, 5);
steps = { 0.001, 0.01, 0.05, 0.1, 0.5 };

% plot for ham/spam
hold on
for i = 1:length(steps)
    [costs, weights] = logistic_regression(spam_train_f, spam_train_l, steps{i}, 50, -1, false); 
    l2_norms(1, i) = norm(weights);
    subplot(length(steps), 1, i);
    plot(1:length(costs), costs);
    title(sprintf('step size = %0.2f', steps{i}));
end
suptitle('EmailSpam Classification')
xlabel('Number Iterations');
ylabel('Cross-Entropy');
hold off
print('emailspam_grad', '-dpng', '-r100');
clf

% plot for ion
hold on
for i = 1:length(steps)
    [costs, weights] = logistic_regression(ion_train_f, ion_train_l, steps{i}, 50, -1, false); 
    l2_norms(2, i) = norm(weights);
    subplot(length(steps), 1, i);
    plot(1:length(costs), costs);
    title(sprintf('step size = %0.2f', steps{i}));
end
suptitle('Ionosphere Classification')
xlabel('Number Iterations');
ylabel('Cross-Entropy');
hold off
print('ion_grad', '-dpng', '-r100');

% uncomment the following to display L2 norms
% disp(l2_norms);