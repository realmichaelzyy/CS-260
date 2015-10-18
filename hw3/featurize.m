function [ features ] = featurize( tokenIndx, filename )
% Given a dictionary of token indexes and a filename, 
% tokenizes the content and outputs a feature vector
% of word counts

    features = zeros(1, length(tokenIndx));
    text = fileread(filename);
    % split on delimiters whitespace and '.', ',', '?'
    tokens = strsplit(text, {' ', '.', ',', '?'});
    for t = tokens
        if tokenIndx.isKey(t{1})
            features(tokenIndx(t{1})) = features(tokenIndx(t{1})) + 1;
        end
    end
end

