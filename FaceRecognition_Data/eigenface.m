clear 

M = 35;                                 % number of images

%LOAD IMAGMES AND CONVERT IMAGES TO VECTOR 
D = './ALL';                            % file path
S = dir(fullfile(D, '*.TIF'));          % all files in the folder matching the name pattern
figure('Name', 'All trained faces');
for i = 1 : numel(S)
       F = fullfile(D, S(i).name);
       Image = imread(F);
       subplot(5,7,i), imshow(Image);   % display the image in one canvas
       X(i, :) = Image(:);              % each image is stored as row in matrix X
end

image_mean = mean(X);                   % mean image   %%If A is a matrix, then mean(A) returns a row vector containing the mean of each column.
X = double(X);


%COVARIANCE MATRIX
S = (X - image_mean)' * (X - image_mean);
%S = cov(X);


%CALCULATE EIGEN VALUES AND VECTOR
[V, D] = eig(S);
d = diag(D);
[out, idx] = sort(d, 'descend');          % sort in descending order
idx = idx(1:100);
                           
W = V(:, idx);                            % pick top 7 Eigenvectors    choose  K=7


% CONSTRUCT DATABASE USING FA
DA = './FA';                               % file path
SA = dir(fullfile(DA, '*.TIF'));           % all files in the folder matching the name pattern
figure('Name', 'All Database Faces');
for i = 1 : numel(SA)
       F = fullfile(DA, SA(i).name);
       Image = imread(F);
       subplot(3,4,i), imshow(Image);      % display the image in one canvas
       FA(i, :) = Image(:);                % each image is stored as row in matrix X
end

C_A = W' * (double(FA) - image_mean)';


% TESTING USING FB
DB = './FB';                               % file path
SB = dir(fullfile(DB, '*.TIF'));           % all files in the folder matching the name pattern
figure('Name', 'Testing Faces');
for i = 1 : numel(SB)
       F = fullfile(DB, SB(i).name);
       Image = imread(F);
       subplot(4,6,i), imshow(Image);      % display the image in one canvas
       FB(i, :) = Image(:);                % each image is stored as row in matrix X
end

C_B = W' * (double(FB) - image_mean)';


% GET PREDICT LABELS
count = 0;
for i = 1:23
    x = C_B(:, i);
    dist = sum((C_A - x).^2, 1);
    [min_d, idx] = min(dist);
    if SB(i).name(7:11) == SA(idx).name(7:11)
        count = count + 1;
    end
%     src = int32(reshape(FB(i, :), 32, 32));
%     tgt = int32(reshape(FA(idx, :), 32, 32));
%     subplot(1,2,1), imshow(src, [0, 255]);
%     subplot(1,2,2), imshow(tgt, [0, 255]);
%     pause(0.5);
end


%COMPUTE ACCURACY
accuracy = (count/23)*100;
info = ['Accuracy = ', num2str(accuracy)];
disp(info)