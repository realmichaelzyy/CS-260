function [train_features, train_labels, test_features, test_labels] = ionosphere_load()

train_features = zeros(269, 34);
train_labels = zeros(269, 1);
fid = fopen('ionosphere/ionosphere_train.dat', 'r') ;
next_line = fgetl(fid) ;
line = 1;
while ~isequal(next_line, -1)
    train_features(line, :) = str2num(next_line(1:length(next_line)-2));
    if (next_line(length(next_line-1)) == 'g')
        train_labels(line) = 1;
    else
        train_labels(line) = 0;
    end    
    line = line + 1;
    next_line = fgetl(fid) ;
end

test_features = zeros(82, 34);
test_labels = zeros(82, 1);
fid = fopen('ionosphere/ionosphere_test.dat', 'r') ;
next_line = fgetl(fid) ;
line = 1;
while ~isequal(next_line, -1)
    test_features(line, :) = str2num(next_line(1:length(next_line)-2));
    if (next_line(length(next_line-1)) == 'g')
        test_labels(line) = 1;
    else
        test_labels(line) = 0;
    end    
    line = line + 1;
    next_line = fgetl(fid) ;
end


end

