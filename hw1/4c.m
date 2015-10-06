1;

load one_hot.m ;
source("knn_classify.m") ;

printf("Loading data...\n");
[train_data, train_labels] = one_hot("car_train.data", 950);
[test_data, test_labels] = one_hot("car_test.data", 389);
[valid_data, valid_labels] = one_hot("car_valid.data", 389);

for k = 1:2:23
  printf("Using k = %d\n", k);
  fflush(stdout);
  [valid_accu, train_accu] = knn_classify(train_data, train_labels, valid_data, valid_labels, k);
  printf("Training Accuracy: %f, Validation Accuracy: %f\n", train_accu, valid_accu);
  fflush(stdout);
endfor

[test_accu, train_accu] = knn_classify(train_data, train_labels, test_data, test_labels, 9);
printf("Test Accuracy when k=9: %f\n", test_accu);
