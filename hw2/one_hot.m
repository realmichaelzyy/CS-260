function [features, labels] = one_hot(filename, num_examples)
  warning('off')
  raw_features = 7 ;
  % 3 * 4 + 3 * 3 = 21 binary features
  bin_features = 21 ;

  fid = fopen(filename,'r') ;
  next_line = fgetl(fid) ;
  raw_array = cell(num_examples, raw_features) ;
  features = zeros(num_examples, bin_features) ; 
  labels = zeros(num_examples, 1) ;
  line_index = 1 ;
  while ~isequal(next_line, -1)
    raw_line = strsplit(next_line, ',') ;
    % lord forgive me
    switch (raw_line{1})
      case 'vhigh' 
        features(line_index, 1) = 1 ;
      case 'high' 
        features(line_index, 2) = 1 ;
      case 'med' 
        features(line_index, 3) = 1 ;
      case 'low' 
        features(line_index, 4) = 1 ;
    end
    switch (raw_line{2})
      case 'vhigh' 
        features(line_index, 5) = 1 ;
      case 'high' 
        features(line_index, 6) = 1 ;
      case 'med' 
        features(line_index, 7) = 1 ;
      case 'low' 
        features(line_index, 8) = 1 ;
    end
    switch (raw_line{3})
      case '2' 
        features(line_index, 9) = 1 ;
      case '3' 
        features(line_index, 10) = 1 ;
      case '4' 
        features(line_index, 11) = 1 ;
      case '5more' 
        features(line_index, 12) = 1 ;
    end
    switch (raw_line{4})
      case '2' 
        features(line_index, 13) = 1 ;
      case '4' 
        features(line_index, 14) = 1 ;
      case 'more' 
        features(line_index, 15) = 1 ;
    end
    switch (raw_line{5})
      case 'small' 
        features(line_index, 16) = 1 ;
      case 'med' 
        features(line_index, 17) = 1 ;
      case 'big' 
        features(line_index, 18) = 1 ;
    end
    switch (raw_line{6})
      case 'low' 
        features(line_index, 19) = 1 ;
      case 'med' 
        features(line_index, 20) = 1 ;
      case 'high' 
        features(line_index, 21) = 1 ;
    end
    switch (raw_line{7})
      case 'unacc' 
        labels(line_index) = 1 ;
      case 'acc' 
        labels(line_index) = 2 ;
      case 'good' 
        labels(line_index) = 3 ;
      case 'vgood' 
        labels(line_index) = 4 ;
    end
    line_index = line_index + 1 ;
    next_line = fgetl(fid) ;
  end
end
