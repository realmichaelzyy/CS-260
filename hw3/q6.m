% Q6) Newton's Method, without Regularization

disp('Loading data...');
[spam_train_f, spam_train_l, spam_test_f, spam_test_l] = bag_of_words();
[ion_train_f, ion_train_l, ion_test_f, ion_test_l] = ionosphere_load();

% Plot Cross Entropy vs Iterations for Ionosphere
[costs, weights, b] = newtons_method(ion_train_f, ion_train_l, 50, -1, false); 
plot(1:50, costs);
title('Ionosphere Newtons Method');
xlabel('Number Iterations');
ylabel('Cross-Entropy');
print('ion_newtons_iterations', '-dpng');
%disp(sprintf('Ionosphere L2 Norm: %f', norm(weights)));
%disp(sprintf('Ionosphere Test Cross Entropy: %f', cross_entropy(ion_test_f, ion_test_l, weights, b)));
clf

% Plot Cross Entropy vs Iterations for EmailSpam
[costs, weights, b] = newtons_method(spam_train_f, spam_train_l, 50, -1, false); 
plot(1:50, costs);
title('EmailSpam Newtons Method');
xlabel('Number Iterations');
ylabel('Cross-Entropy');
print('emailspam_newtons_iterations', '-dpng');
%disp(sprintf('EmailSpam L2 Norm: %f', norm(weights)));
%disp(sprintf('EmailSpam Test Cross Entropy: %f', cross_entropy(spam_test_f, spam_test_l, weights, b)));
clf