function [train_features, train_labels, test_features, test_labels] = preprocess(print)

    if (print)
        disp('Preprocessing data');
        disp('==================');
    end

    TRAIN_FILENAME = 'splice_train.mat';
    TEST_FILENAME = 'splice_test.mat';

    raw_train = load(strcat('data/', TRAIN_FILENAME));
    raw_test = load(strcat('data/', TEST_FILENAME));
    
    % subtract mean and divide by the standard deviation column-wise
    train_features = (raw_train.data - ...
                      repmat(mean(raw_train.data), size(raw_train.data, 1), 1)) ...
                     ./ repmat(std(raw_train.data), size(raw_train.data, 1), 1);
    % test features use mean and std dev of the training features
    test_features = (raw_test.data - ...
                      repmat(mean(raw_train.data), size(raw_test.data, 1), 1)) ...
                     ./ repmat(std(raw_train.data), size(raw_test.data, 1), 1);
                 
    train_labels = raw_train.label;
    test_labels = raw_test.label;
    
    if (print)
        means = mean(raw_train.data);
        stddevs = std(raw_train.data);
        disp(sprintf('3rd feature: %0.4f mean, %0.4f standard deviation', means(2), stddevs(2)));
        disp(sprintf('10th feature: %0.4f mean, %0.4f standard deviation', means(9), stddevs(9)));
        disp(' ');
    end