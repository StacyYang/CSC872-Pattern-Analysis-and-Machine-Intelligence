clear
close all
clc



%%%%LOAD IMAGES%%%%
DirectoryF = './Female'; 
femaleFiles = dir(fullfile(DirectoryF, '*.TIF'));
Mf = length(femaleFiles);                               %number of female images
figure('Name', 'Training female images');
for i=1:Mf
    femaleImageFile = fullfile(DirectoryF, femaleFiles(i).name);
    femaleImage = imread(femaleImageFile);
    subplot(6,9,i), imshow(femaleImage);
    Xf(i,:)= femaleImage(:);                            %each image is stored as row in matrix Xf
end
    
DirectoryM = './Male'; 
maleFiles = dir(fullfile(DirectoryM, '*.TIF'));
Mm = length(maleFiles);                                 %number of male images
figure('Name', 'Training male images');
for i=1:Mm
    maleImageFile = fullfile(DirectoryM, maleFiles(i).name);
    maleImage = imread(maleImageFile);
    subplot(5,9,i), imshow(maleImage);
    Xm(i, :) = maleImage(:);                            %each image is stored as row in matrix Xm
end

total_Num = Mf + Mm;

%%%%PERFORM PCA ON THE ENTIRE DATA SET%%%%
X = [Xf; Xm];
mean_vec = mean(X);                                     %mean
X = double(X);                      

S = (X - mean_vec)'*(X - mean_vec);                     %covariance matrix
[V,D] = eig(S);
eig_val = diag(D);                                      %calculate eigen values 
[sorted_eig_val, idx] = sort(eig_val, 'descend');       %sort eigen values in descending order
sorted_eig_vector = V(:, idx);
total_eig_vals = sum(eig_val);

%%%%FIND A SUBSET WITH K TOP PCs THAT CAPTURE MOST OF DATA%%%%
counter = 1;
threshold = 0.95;
sum_eigvals = sorted_eig_val(1);
while sum_eigvals < total_eig_vals * threshold
    counter = counter +1;
    sum_eigvals = sum_eigvals + sorted_eig_val(counter);

end
top_k_eig_vector = sorted_eig_vector(:, 1:counter);


%%%%Form PCA MODEL%%%%
FX = top_k_eig_vector' * (X - mean_vec)';
FXf = FX(:, 1:Mf);                                      %projected female image data points
FXm = FX(:, Mf+1 : total_Num);                          %projected male image data points

%%%%COMPUTE Sw AND Sb%%%%
u = mean(FX, 2);
uf = mean(FXf, 2);
um = mean(FXm, 2);

x = FXf(:, 1);
temp = x - uf;
Sw = temp * temp';
for i = 2:Mf
    x = FXf(:,i);
    temp = x -uf;
    Sw = Sw + temp * temp';
end

for i = 1: Mm
    x = FXm(:, i); 
    temp = x -um;
    Sw = Sw + temp * temp';
end

Sbf = (uf - u)*(uf - u)';
Sbm = (um - u)*(um - u)';
Sb = Mf * Sbf + Mm * Sbm;

%%%%SOLVE A GENERALIZED EIGEN-VALUE PROBLEM WITH Sw and Sb%%%%
S_lda = inv(Sw) * Sb;
[vLDA, dLDA] = eigs(S_lda, 1);                                 % find the largest magnitude eigenvalue

%%%%COMPUTE DISCRIMINANT SLOPE AND INTERCEPT%%%%
w = vLDA' * top_k_eig_vector';
b = vLDA' * (uf + um)/2;

%%%%TEST%%%%
MISCLASSIFIED = [];
result_f = [];
result_m = [];
for i = 1 : Mf
    y = X(i, :);
    result = w * (y - mean_vec)' - b;
    if result < 0
        result_f = [result_f, 1];
    end
    if result > 0
        MISCLASSIFIED = [MISCLASSIFIED, i];
        result_f = [result_f, -1];
    end
end


for i = Mf+1 : total_Num
    y = X(i, :);
    result = w * (y - mean_vec)' - b;
    if result > 0
        result_m = [result_m, -1];
    end
    if result < 0
        MISCLASSIFIED = [MISCLASSIFIED, i];
        result_m = [result_m, 1];
    end
end


%%%%DISPLAY MISCLASSIFIED IMAGES%%%%
length = size(MISCLASSIFIED);
length = length(2);
figure('Name', 'All misclassified images');
colormap('gray');
for i = 1:length
    img = X(MISCLASSIFIED(i), :);
    img = reshape(img, 32, 32);
    subplot(1, length, i);
    imshow(img, []);
end

