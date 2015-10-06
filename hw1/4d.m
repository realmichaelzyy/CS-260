1;

warning("off");
source("knn_classify.m") ;
printf("Loading data...\n");
boundary = importdata("boundary.mat");

grid = zeros(10000, 5);

for k = [1, 5, 15, 20]
  for i = 1:100
    for j = 1:100
      point = [i/100.0 j/100.0]; 
      class = classify_vec(boundary.features, boundary.labels, point, k);
      if (class == 1)
        grid(i*100 + j, :) = [ point 1 1 1 ];
      else
        grid(i*100 + j, :) = [ point 0 0 0 ];
      endif
    endfor
  endfor
  scatter(grid(:, 1), grid(:, 2), [], grid(:, 3:5));
  print(sprintf("%dk.png", k));
  printf("Done with %d\n", k);
  fflush(stdout);
endfor
