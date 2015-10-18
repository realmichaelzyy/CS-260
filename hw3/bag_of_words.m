function [ train_features, train_labels, test_features, test_labels ] = bag_of_words()
% Returns bag-of-words features and labels for the spam/ham dataset

    % open and read dict.dat
    tokenIndx = containers.Map;
    indx = 1;
    fid = fopen('spam/dic.dat','r') ;
    next_line = fgetl(fid) ;
    while ~isequal(next_line, -1)
        tokenIndx(next_line) = indx;
        indx = indx + 1;
        next_line = fgetl(fid);
    end
    
    [train_features, train_labels] = featurize_dir(tokenIndx, 'spam/train/');
    [test_features, test_labels] = featurize_dir(tokenIndx, 'spam/test/');

    % display top three words
    word_counts = sum(train_features) + sum(test_features);
    word_counts = [word_counts ; 1:length(word_counts)];
    word_counts = sortrows(word_counts', 1)';
    for i = 1:3
        wordIndx = word_counts(2, length(word_counts)-i+1);
        count = word_counts(1, length(word_counts)-i+1);
        word = 'not found';
        for j = tokenIndx.keys()
            if tokenIndx(j{1}) == wordIndx
                word = j{1};
                disp(word);
                disp(count);
                continue;
            end
        end
    end
end

