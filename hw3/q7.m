% Q7) Newton's Method with Regularization

disp('Loading data...');
[spam_train_f, spam_train_l, spam_test_f, spam_test_l] = bag_of_words();
[ion_train_f, ion_train_l, ion_test_f, ion_test_l] = ionosphere_load();

x = [];
spam_train = [];
spam_test = [];
spam_norms = [];
ion_train = [];
ion_test = [];
ion_norms = [];
for lambda = 0:0.05:0.5
    x = [ x lambda ];
    [train_costs, weights, b] = newtons_method(spam_train_f, spam_train_l, 50, lambda, false);
    spam_train = [ spam_train cross_entropy(spam_train_f, spam_train_l, weights, b) ];
    spam_test = [ spam_test cross_entropy(spam_test_f, spam_test_l, weights, b) ];
    spam_norms = [ spam_norms norm(weights) ];
    [train_costs, weights, b] = newtons_method(ion_train_f, ion_train_l, 50, lambda, false);
    ion_train = [ ion_train cross_entropy(ion_train_f, ion_train_l, weights, b) ];
    ion_test = [ ion_test cross_entropy(ion_test_f, ion_test_l, weights, b) ];
    ion_norms = [ ion_norms norm(weights) ];
    disp(sprintf('%d done', lambda));
end

hold on
plot(x, spam_train);
plot(x, spam_test, 'LineStyle', '--');
legend({'train', 'test'});
title('EmailSpam Cross-Entropy with Regularization');
xlabel('Regularization Coeffecient');
ylabel('Cross-Entropy');
hold off
print('email_newtons_reg', '-dpng');
clf 

hold on
plot(x, ion_train);
plot(x, ion_test, 'LineStyle', '--');
legend({'train', 'test'});
title('Ionosphere Cross-Entropy with Regularization');
xlabel('Regularization Coeffecient');
ylabel('Cross-Entropy');
hold off
print('ion_newtons_reg', '-dpng');
clf 

% Uncomment the following to display L2 norms and test cross-entropy data
% for i = 1:length(spam_train)
%     disp('\hline');
%     disp(sprintf('%d & %f & %f & %f & %f \\\\', x(i), ... 
%          spam_norms(i), ...
%          spam_test(i), ...
%          ion_norms(i), ... 
%          ion_test(i)));
% end