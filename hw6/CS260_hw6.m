data = load('face_data.mat');

% construct image matrix
N = size(data.image, 2);
X = [];
for i = 1:N
    pixels = size(data.image{i}, 1) * size(data.image{i}, 2);
    X = [ X ; reshape(data.image{i}, 1, pixels) ];
end

% shuffle images
r = randperm(N);
X = X(r, :);
Y = data.personID(r)';

% Save top eigenfaces as files
eigenfaces = pca_fun(X, 200);
for p = 1:5
    scale_save_img(reshape(eigenfaces(:, p), 50, 50), ...
                   sprintf('eigenface_%d.png', p));
end

D = [ 20, 50, 100, 200 ]; 
Cbase = 2;
C = 10;
train_test_ratio = 0.75

for idx = 1:numel(D)
    d = D(idx)
    P = pca_fun(X, d);
    Xp = X * P;

    split = floor(N * train_test_ratio);
    trainf = double(Xp(1:split, :));
    trainl = Y(1:split);
    testf = double(Xp(split:N, :));
    testl = Y(split:N);

    accus = [];
    for c = 0:C
        opts = sprintf('-v 5 -c %d -q', pow2(Cbase, c));
        model = svmtrain(trainl, trainf, opts);
        accus = [ accus ; pow2(Cbase, c) model ];
    end

    [a, max_ind] = max(accus);
    bestC = accus(max_ind(2), 1);

    opts = sprintf('-c %d -q', bestC);
    model = svmtrain(trainl, trainf, opts);
    [predicted, test_accu, decision_values] = ...
        svmpredict(testl, testf, model);
    %disp(sprintf('D: %d, C = %.4f, Accu = %.6f', d, bestC, test_accu(1)));
    disp(sprintf('%d & %d & %.6f \\\\', d, bestC, test_accu(1)));
end
