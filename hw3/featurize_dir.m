function [ features, labels ] = featurize_dir(tokenIndx, subdir)
% Creates a feature matrix of NxD and label vector of Nx1 from the 
% specified directory. N is the amount of text files in the directory
% and D is the number of tokens, or length(tokenIndx).
% Arguments
%   1. tokenIndx - Map of tokens to the feature index
%   2. dir - path to the directory you want to featurize

    D = length(tokenIndx);
    
    % ham
    files_dir = strcat(subdir, 'ham/');
    files = dir(strcat(files_dir, '*.txt'));
    ham = zeros(length(files), D+1);
    for k = 1:length(files)
        ham(k, 1:D) = featurize(tokenIndx, strcat(files_dir, files(k).name));
        ham(k, D+1) = 1;
    end
    
    % spam
    files_dir = strcat(subdir, 'spam/');
    files = dir(strcat(files_dir, '*.txt'));
    spam = zeros(length(files), D+1);
    for k = 1:length(files)
        spam(k, 1:D) = featurize(tokenIndx, strcat(files_dir, files(k).name));
        spam(k, D+1) = 0;
    end
    
   	ordered = [ spam ; ham ];
    % shuffle the training examples
    % http://stackoverflow.com/questions/5444248/random-order-of-rows-matlab
    shuffled = ordered(randperm(size(ordered,1)),:);
    
    features = shuffled(:, 1:D);
    labels = shuffled(:, D+1);
end

